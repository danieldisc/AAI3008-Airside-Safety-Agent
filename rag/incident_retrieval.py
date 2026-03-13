from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import DEFAULT_TOP_K, INDEX_DIR
from rag.query_index import query_index


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def expand_retrieval_query(claim_text: str) -> str:
    lowered = claim_text.lower()
    expansions = []

    if any(token in lowered for token in ["engine", "intake", "blast", "hazard"]):
        expansions.extend(["engine ingestion zone", "jet blast area", "ramp clearance"])
    if any(token in lowered for token in ["marshaller", "signal", "stop", "hand-signal"]):
        expansions.extend(["manual marshalling", "STOP signal", "aircraft movement danger"])
    if any(token in lowered for token in ["warning", "beep", "audio"]):
        expansions.extend(["safety communication", "warning signal", "ground crew alert"])
    if any(token in lowered for token in ["wing", "wingtip", "tow", "towed", "marshaller"]):
        expansions.extend(["wingwalker", "aircraft movement", "imminent danger stop signal"])
    if any(token in lowered for token in ["fuel", "fueling", "refuel", "beacon", "anti-collision"]):
        expansions.extend(["fueling safety", "anti-collision lights", "engines running", "fire prevention"])
    if any(token in lowered for token in ["pushback", "tail", "exhaust", "thrust", "heat distortion"]):
        expansions.extend(["jet blast area", "remain clear", "ground personnel distance"])

    if not expansions:
        return claim_text

    return f"{claim_text} {' '.join(expansions)}"


def claim_queries_from_incident(payload: dict) -> list[dict]:
    claims: list[dict] = []

    observations = payload.get("observations", [])
    for idx, observation in enumerate(observations, start=1):
        claim_text = normalize_text(str(observation.get("text", "")))
        if not claim_text:
            continue

        t_start = observation.get("t_start")
        t_end = observation.get("t_end")
        query = expand_retrieval_query(claim_text)

        claims.append(
            {
                "claim_id": f"obs_{idx}",
                "claim_type": "observation",
                "time_window": {"t_start": t_start, "t_end": t_end},
                "claim_text": claim_text,
                "retrieval_query": query,
            }
        )

    signals = payload.get("signals", {})
    for signal_type in ("visual", "audio"):
        values = signals.get(signal_type, [])
        for idx, value in enumerate(values, start=1):
            signal_text = normalize_text(str(value).replace("_", " "))
            if not signal_text:
                continue

            query = expand_retrieval_query(signal_text)
            claims.append(
                {
                    "claim_id": f"{signal_type}_{idx}",
                    "claim_type": f"signal_{signal_type}",
                    "time_window": None,
                    "claim_text": signal_text,
                    "retrieval_query": query,
                }
            )

    audio_events = payload.get("audio", [])
    for idx, audio_item in enumerate(audio_events, start=1):
        if isinstance(audio_item, dict):
            text = audio_item.get("text", "")
            t_stamp = audio_item.get("t")
        else:
            text = audio_item
            t_stamp = None

        claim_text = normalize_text(str(text))
        if not claim_text:
            continue

        claims.append(
            {
                "claim_id": f"audio_event_{idx}",
                "claim_type": "audio_observation",
                "time_window": {"t": t_stamp} if t_stamp is not None else None,
                "claim_text": claim_text,
                "retrieval_query": expand_retrieval_query(claim_text),
            }
        )

    uncertainty_notes = payload.get("uncertainty_notes", [])
    for idx, note in enumerate(uncertainty_notes, start=1):
        note_text = normalize_text(str(note))
        if not note_text:
            continue
        claims.append(
            {
                "claim_id": f"uncertainty_{idx}",
                "claim_type": "uncertainty_note",
                "time_window": None,
                "claim_text": note_text,
                "retrieval_query": expand_retrieval_query(note_text),
            }
        )

    return claims


def retrieve_for_incident(payload: dict, index_dir: Path, top_k: int) -> dict:
    clip_id = payload.get("clip_id", "unknown_clip")
    claims = claim_queries_from_incident(payload)

    claim_results = []
    for claim in claims:
        hits = query_index(index_dir=index_dir, query=claim["retrieval_query"], top_k=top_k)
        claim_results.append(
            {
                **claim,
                "evidence": hits,
            }
        )

    return {
        "clip_id": clip_id,
        "retrieval_top_k": top_k,
        "claims": claim_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run manual retrieval per incident claim from VLM JSON"
    )
    parser.add_argument(
        "--incident-json",
        type=Path,
        default=Path("sample.json"),
        help="Path to VLM-style incident JSON",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file path; prints to stdout if omitted",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with args.incident_json.open("r", encoding="utf-8") as f:
        incident_payload = json.load(f)

    result = retrieve_for_incident(
        payload=incident_payload,
        index_dir=args.index_dir,
        top_k=args.top_k,
    )

    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"Wrote retrieval output to {args.output}")
