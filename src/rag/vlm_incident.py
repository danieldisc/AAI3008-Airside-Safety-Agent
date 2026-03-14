from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any


def _load_safety_agent_class(airside_src_dir: Path):
    module_path = airside_src_dir / "vlm_agent.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Could not find VLM module at {module_path}")

    spec = importlib.util.spec_from_file_location("airside_vlm_agent", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Python module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "SafetyAgent"):
        raise ImportError("SafetyAgent class not found in vlm_agent.py")
    return module.SafetyAgent


def _normalize_time(value: float) -> float:
    return round(float(value), 2)


def _slugify_stem(stem: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_")
    return cleaned.lower() or "incident"


def _derive_observations(logs: list[dict]) -> list[dict]:
    observations: list[dict] = []

    if not logs:
        return observations

    ordered = sorted(logs, key=lambda item: int(item.get("frame_index", 0)))

    has_engine_active = any(bool(item.get("propeller_active")) for item in ordered)
    has_person = any(bool(item.get("person_detected")) for item in ordered)

    if has_engine_active:
        first_active = next(
            int(item.get("frame_index", 0))
            for item in ordered
            if bool(item.get("propeller_active"))
        )
        observations.append(
            {
                "t_start": _normalize_time(first_active),
                "t_end": _normalize_time(first_active + 1),
                "text": "Aircraft engine/propeller appears active during ground operation.",
            }
        )

    if has_engine_active and has_person:
        first_overlap = next(
            int(item.get("frame_index", 0))
            for item in ordered
            if bool(item.get("propeller_active")) and bool(item.get("person_detected"))
        )
        observations.append(
            {
                "t_start": _normalize_time(first_overlap),
                "t_end": _normalize_time(first_overlap + 1),
                "text": "Ground personnel visible while aircraft engine/propeller appears active.",
            }
        )

    in_segment = False
    seg_start = 0
    prev_frame = 0

    for item in ordered:
        frame_idx = int(item.get("frame_index", 0))
        violation = bool(item.get("danger_zone_violation"))

        if violation and not in_segment:
            in_segment = True
            seg_start = frame_idx

        if in_segment and not violation:
            observations.append(
                {
                    "t_start": _normalize_time(seg_start),
                    "t_end": _normalize_time(frame_idx),
                    "text": "Ground crew enters aircraft engine intake hazard and exclusion zone while engine/propeller appears active.",
                }
            )
            in_segment = False

        prev_frame = frame_idx

    if in_segment:
        observations.append(
            {
                "t_start": _normalize_time(seg_start),
                "t_end": _normalize_time(prev_frame + 1),
                "text": "Ground crew remains within aircraft engine intake hazard and exclusion zone while engine/propeller appears active.",
            }
        )

    return observations


def _derive_audio_and_uncertainty(analyst_text: str) -> tuple[list[dict], list[str], list[str], list[str]]:
    text = (analyst_text or "").lower()

    audio_items: list[dict] = []
    visual_signals: list[str] = []
    audio_signals: list[str] = []
    uncertainty_notes: list[str] = []

    if any(token in text for token in ("warning", "beep", "alarm", "stand clear", "announcement")):
        audio_items.append({"t": 0.0, "text": "Warning cue detected in analyst narrative."})
        audio_signals.append("warning_beep_present")

    if "no clear stop signal" in text or "no stop signal" in text:
        visual_signals.append("no_stop_signal_detected")

    if any(token in text for token in ("unclear", "uncertain", "not visible", "cannot confirm")):
        uncertainty_notes.append("Some safety cues are uncertain due to camera angle or occlusion.")

    return audio_items, visual_signals, audio_signals, uncertainty_notes


def _build_incident_payload(video_path: Path, clip_id: str | None, logs: list[dict], analyst_text: str) -> dict[str, Any]:
    observations = _derive_observations(logs)

    audio_items, visual_from_text, audio_from_text, uncertainty_notes = _derive_audio_and_uncertainty(analyst_text)

    any_violation = any(bool(item.get("danger_zone_violation")) for item in logs)
    any_engine_active = any(bool(item.get("propeller_active")) for item in logs)
    any_person = any(bool(item.get("person_detected")) for item in logs)

    visual_signals = list(visual_from_text)
    if any_violation:
        visual_signals.append("danger_zone_violation_detected")
    if any_engine_active and any_person:
        visual_signals.append("personnel_near_active_engine")

    if not observations:
        observations = [
            {
                "t_start": 0.0,
                "t_end": 0.0,
                "text": "No clear safety violation observed in sampled frames.",
            }
        ]

    unique_visual = list(dict.fromkeys(visual_signals))
    unique_audio_signals = list(dict.fromkeys(audio_from_text))

    violation_frames = sum(1 for item in logs if bool(item.get("danger_zone_violation")))
    confidence = 0.62
    if logs:
        ratio = violation_frames / max(len(logs), 1)
        confidence = min(0.92, max(0.55, 0.62 + (0.34 * ratio) + (0.05 if any_violation else 0.0)))

    return {
        "clip_id": clip_id or f"inc_{_slugify_stem(video_path.stem)}",
        "observations": observations,
        "audio": audio_items,
        "scene": {
            "aircraft_type": "unknown",
            "location": "airport ramp/gate",
            "visibility": "unknown",
        },
        "signals": {
            "visual": unique_visual,
            "audio": unique_audio_signals,
        },
        "uncertainty_notes": uncertainty_notes,
        "confidence": round(confidence, 2),
    }


def build_video_analysis_artifacts(
    video_path: Path,
    engine: str = "Gemini",
    clip_id: str | None = None,
    airside_src_dir: Path | None = None,
) -> tuple[dict[str, Any], str, list[dict]]:
    repo_root = Path(__file__).resolve().parent.parent
    legacy_workspace_root = repo_root.parent

    if airside_src_dir is not None:
        resolved_src_dir = airside_src_dir
    else:
        default_src_dir = repo_root / "src"
        legacy_src_dir = legacy_workspace_root / "AAI3008-Airside-Safety-Agent" / "src"
        resolved_src_dir = default_src_dir if default_src_dir.exists() else legacy_src_dir

    SafetyAgent = _load_safety_agent_class(resolved_src_dir)
    agent = SafetyAgent()

    full_logs, analyst_text = agent.analyze_pipeline(str(video_path), engine=engine)
    logs = full_logs if isinstance(full_logs, list) else []
    payload = _build_incident_payload(video_path=video_path, clip_id=clip_id, logs=logs, analyst_text=analyst_text)
    return payload, analyst_text, logs


def build_incident_payload_from_video(
    video_path: Path,
    engine: str = "Gemini",
    clip_id: str | None = None,
    airside_src_dir: Path | None = None,
) -> dict:
    payload, _, _ = build_video_analysis_artifacts(
        video_path=video_path,
        engine=engine,
        clip_id=clip_id,
        airside_src_dir=airside_src_dir,
    )
    return payload
