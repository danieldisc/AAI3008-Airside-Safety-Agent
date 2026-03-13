from __future__ import annotations

import argparse
import json
from pathlib import Path


GUIDANCE_LIBRARY = {
    "ENG_INGESTION_ZONE_BREACH": {
        "teachable_moment": "Engine intake and blast zones are immediate life-safety boundaries. Personnel must never enter these areas while engines are running or start-up is imminent.",
        "corrective_actions": [
            "Stop ramp movement in the affected stand and clear all personnel from danger zones.",
            "Conduct a targeted safety brief on intake/blast boundaries before next turnaround.",
            "Verify stand markings and visual hazard boundary cues are clearly visible.",
        ],
        "prevention_controls": [
            "Pre-task hazard briefing with engine-danger map",
            "Supervisor confirmation of zone clearance before movement",
            "Recurring competency checks for ramp crew",
        ],
    },
    "NO_STOP_SIGNAL_ON_DANGER": {
        "teachable_moment": "When danger is detected during aircraft movement, STOP signaling is non-negotiable and must be immediate, clear, and acknowledged.",
        "corrective_actions": [
            "Re-brief marshalling team on mandatory STOP signal triggers.",
            "Run a short drill for imminent-danger STOP hand-signal execution.",
            "Require explicit cockpit/crew acknowledgement loop for stop commands.",
        ],
        "prevention_controls": [
            "Marshaller pre-position and line-of-sight check",
            "Standardized hand-signal refresher at shift start",
            "Spot audits on marshalling protocol adherence",
        ],
    },
    "SAFETY_ALERT_RESPONSE_GAP": {
        "teachable_moment": "Warning cues (audio/visual) are only effective when teams execute a defined response immediately.",
        "corrective_actions": [
            "Review the local alarm-response checklist with all involved personnel.",
            "Introduce a call-and-response protocol for active warnings.",
            "Document response timing in post-operation safety log.",
        ],
        "prevention_controls": [
            "Alarm-to-action checklist cards",
            "Communication role assignment per turnaround",
            "Periodic drills with timed response targets",
        ],
    },
    "NO_WING_WALKER_COVERAGE": {
        "teachable_moment": "Aircraft towing requires complete wing-walker coverage to protect clearance margins and prevent wingtip strikes.",
        "corrective_actions": [
            "Hold tow movement until required wing-walker positions are manned.",
            "Re-brief towing team on wing-walker responsibilities and stop triggers.",
            "Apply a pre-movement checklist confirming wingtip clearance monitoring.",
        ],
        "prevention_controls": [
            "Mandatory wing-walker assignment log",
            "Tow-start supervisor sign-off",
            "Wingtip clearance drill during recurrent training",
        ],
    },
    "JET_BLAST_PROXIMITY": {
        "teachable_moment": "Jet blast zones can rapidly move personnel and equipment; blast-area clearance must be enforced before and during thrust changes.",
        "corrective_actions": [
            "Clear non-essential personnel and loose equipment from blast-risk areas.",
            "Reconfirm pushback exclusion zones before applying thrust.",
            "Pause operation when blast indicators (heat distortion/debris movement) are observed.",
        ],
        "prevention_controls": [
            "Blast-zone floor marking verification",
            "Pushback role brief with distance checks",
            "Equipment securing checklist",
        ],
    },
    "FUELING_WITH_POSSIBLE_ENGINE_ACTIVITY": {
        "teachable_moment": "Fueling must not proceed under unsafe aircraft energy states; beacon/engine cues require immediate verification and control.",
        "corrective_actions": [
            "Suspend fueling until engine/beacon safety state is confirmed.",
            "Coordinate flight deck-ground confirmation on aircraft safe fueling status.",
            "Apply fire-prevention controls and enforce stand safety perimeter.",
        ],
        "prevention_controls": [
            "Fueling start authorization checklist",
            "Beacon/engine state cross-check protocol",
            "Refueling safety audit sampling",
        ],
    },
}


def default_guidance() -> dict:
    return {
        "teachable_moment": "Safety-critical anomalies require immediate pause, role clarity, and controlled restart only after risk is mitigated.",
        "corrective_actions": [
            "Pause operation and verify hazard controls.",
            "Re-brief involved roles on procedure before resuming.",
        ],
        "prevention_controls": [
            "Role-based refresher training",
            "Supervisor verification checkpoints",
        ],
    }


def priority_rank(severity: str) -> int:
    table = {"high": 1, "medium": 2, "low": 3}
    return table.get(severity.lower(), 3)


def build_coaching_payload(violations_payload: dict) -> dict:
    mapped_violations = violations_payload.get("mapped_violations", [])
    coaching_items = []

    for violation in mapped_violations:
        code = str(violation.get("violation_code", ""))
        severity = str(violation.get("severity", "medium"))
        guidance = GUIDANCE_LIBRARY.get(code, default_guidance())

        coaching_items.append(
            {
                "claim_id": violation.get("claim_id"),
                "violation_code": code,
                "severity": severity,
                "priority": priority_rank(severity),
                "teachable_moment": guidance["teachable_moment"],
                "corrective_actions": guidance["corrective_actions"],
                "prevention_controls": guidance["prevention_controls"],
                "supporting_citations": violation.get("citations", [])[:3],
            }
        )

    coaching_items.sort(key=lambda item: (item["priority"], item.get("claim_id") or ""))

    return {
        "clip_id": violations_payload.get("clip_id", "unknown_clip"),
        "stage": "llm3_teachable_moment",
        "coaching_items": coaching_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate teachable moments and corrective actions from mapped violations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/violations_sample.json"),
        help="Input mapped violations JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/teachable_sample.json"),
        help="Output teachable-moment JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        violations_payload = json.load(f)

    result = build_coaching_payload(violations_payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote teachable payload to {args.output}")
