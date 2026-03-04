"""Gemini API client module.

Handles uploading a video file to the Google File API and querying
the Gemini 1.5 Flash model for safety-violation analysis.
"""

import time

from google import genai
from google.genai import types


def analyse_video(api_key: str, video_path: str, safety_rules: str) -> str:
    """Upload *video_path* to the Gemini File API and return the model's analysis.

    Args:
        api_key:      Google Gemini API key.
        video_path:   Absolute path to the MP4 file on disk.
        safety_rules: Plain-text safety rules to check against.

    Returns:
        The model's raw text response describing any detected violations.

    Raises:
        google.genai.errors.APIError: On any API-level failure.
        RuntimeError: If the uploaded file fails to become active.
    """
    client = genai.Client(api_key=api_key)

    # Upload the video and wait for processing
    video_file = client.files.upload(
        file=video_path,
        config=types.UploadFileConfig(mime_type="video/mp4"),
    )

    # Poll until the file is active (may take several seconds for large files)
    max_wait_seconds = 120
    elapsed = 0
    poll_interval = 5
    while video_file.state == types.FileState.PROCESSING:
        if elapsed >= max_wait_seconds:
            raise RuntimeError(
                "The uploaded video is still processing after "
                f"{max_wait_seconds} seconds. Please try again."
            )
        time.sleep(poll_interval)
        elapsed += poll_interval
        video_file = client.files.get(name=video_file.name)

    if video_file.state != types.FileState.ACTIVE:
        raise RuntimeError(
            f"Video upload failed with state: {video_file.state}"
        )

    prompt = (
        "You are an aviation ground-handling safety inspector.\n\n"
        "The following safety rules apply:\n"
        f"{safety_rules}\n\n"
        "Carefully review the video and identify any violations of these rules. "
        "For each violation found, provide:\n"
        "  - Violation: A short title (e.g. 'Person too close to engine')\n"
        "  - Timestamp: Approximate time in the video (e.g. '00:12')\n"
        "  - Description: A one-sentence explanation of the hazard\n"
        "  - Severity: Low / Medium / High\n\n"
        "If no violations are found, state 'No violations detected.'\n"
        "Respond in British English."
    )

    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=[video_file, prompt],
    )

    # Clean up the uploaded file to avoid quota build-up
    try:
        client.files.delete(name=video_file.name)
    except Exception:  # noqa: BLE001 – best-effort cleanup, never fatal
        pass

    return response.text
