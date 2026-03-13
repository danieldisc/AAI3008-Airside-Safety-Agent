from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_report(
    incident_payload: dict,
    retrieval_payload: dict,
    violations_payload: dict,
    teachable_payload: dict,
) -> dict:
    mapped_violations = violations_payload.get("mapped_violations", [])
    coaching_items = teachable_payload.get("coaching_items", [])

    high_count = sum(1 for item in mapped_violations if str(item.get("severity", "")).lower() == "high")
    medium_count = sum(1 for item in mapped_violations if str(item.get("severity", "")).lower() == "medium")
    low_count = sum(1 for item in mapped_violations if str(item.get("severity", "")).lower() == "low")

    overall_risk = "low"
    if high_count >= 1:
        overall_risk = "high"
    elif medium_count >= 1:
        overall_risk = "medium"

    claim_summary = []
    claims = retrieval_payload.get("claims", [])
    for claim in claims:
        claim_id = claim.get("claim_id")
        mapped = [v for v in mapped_violations if v.get("claim_id") == claim_id]
        claim_summary.append(
            {
                "claim_id": claim_id,
                "claim_type": claim.get("claim_type"),
                "claim_text": claim.get("claim_text"),
                "time_window": claim.get("time_window"),
                "matched_violations": [
                    {
                        "violation_code": v.get("violation_code"),
                        "severity": v.get("severity"),
                        "confidence": v.get("confidence"),
                        "citations": v.get("citations", []),
                    }
                    for v in mapped
                ],
            }
        )

    immediate_actions = []
    for coaching in coaching_items:
        if coaching.get("severity") == "high":
            immediate_actions.extend(coaching.get("corrective_actions", [])[:2])
    immediate_actions = list(dict.fromkeys(immediate_actions))[:6]

    return {
        "clip_id": incident_payload.get("clip_id", retrieval_payload.get("clip_id", "unknown_clip")),
        "stage": "llm4_structured_report",
        "scene": incident_payload.get("scene", {}),
        "confidence": incident_payload.get("confidence"),
        "summary": {
            "overall_risk": overall_risk,
            "violation_count": len(mapped_violations),
            "severity_breakdown": {
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
            "top_findings": [
                {
                    "violation_code": item.get("violation_code"),
                    "label": item.get("violation_label"),
                    "severity": item.get("severity"),
                    "confidence": item.get("confidence"),
                }
                for item in mapped_violations[:5]
            ],
        },
        "claims": claim_summary,
        "teachable_moments": coaching_items,
        "recommended_immediate_actions": immediate_actions,
        "audit": {
            "retrieval_top_k": retrieval_payload.get("retrieval_top_k"),
            "total_claims": len(claims),
            "mapped_claims": len(mapped_violations),
            "unmapped_claims": len(violations_payload.get("unmapped_claims", [])),
            "source_artifacts": {
                "incident_input": "sample.json",
                "retrieval_output": "outputs/retrieval_sample.json",
                "violations_output": "outputs/violations_sample.json",
                "teachable_output": "outputs/teachable_sample.json",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble final structured incident report payload")
    parser.add_argument("--incident", type=Path, default=Path("sample.json"))
    parser.add_argument("--retrieval", type=Path, default=Path("outputs/retrieval_sample.json"))
    parser.add_argument("--violations", type=Path, default=Path("outputs/violations_sample.json"))
    parser.add_argument("--teachable", type=Path, default=Path("outputs/teachable_sample.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report_sample.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    incident_payload = load_json(args.incident)
    retrieval_payload = load_json(args.retrieval)
    violations_payload = load_json(args.violations)
    teachable_payload = load_json(args.teachable)

    result = build_report(
        incident_payload=incident_payload,
        retrieval_payload=retrieval_payload,
        violations_payload=violations_payload,
        teachable_payload=teachable_payload,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote final report payload to {args.output}")
