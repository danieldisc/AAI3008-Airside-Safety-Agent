from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ViolationRule:
    code: str
    label: str
    severity: str
    trigger_keywords: tuple[str, ...]
    evidence_keywords: tuple[str, ...]
    action_tags: tuple[str, ...]


DEFAULT_RULES = (
    ViolationRule(
        code="ENG_INGESTION_ZONE_BREACH",
        label="Personnel entered engine ingestion/blast danger area",
        severity="high",
        trigger_keywords=("engine", "intake", "hazard", "blast"),
        evidence_keywords=("engine intake", "blast area", "danger area", "remain clear"),
        action_tags=("stop_operation", "retrain_ground_crew", "mark_hazard_boundary"),
    ),
    ViolationRule(
        code="NO_STOP_SIGNAL_ON_DANGER",
        label="Missing or ineffective STOP marshalling signal during danger",
        severity="high",
        trigger_keywords=("stop", "marshaller", "signal", "continues forward"),
        evidence_keywords=("stop signal", "manual marshalling", "imminent danger", "stop the aircraft"),
        action_tags=("marshaller_refresher", "signal_protocol_enforcement"),
    ),
    ViolationRule(
        code="SAFETY_ALERT_RESPONSE_GAP",
        label="Warning signal present without adequate safety response",
        severity="medium",
        trigger_keywords=("warning", "beep", "audio"),
        evidence_keywords=("warning", "safety communication", "alert"),
        action_tags=("alarm_response_drill", "communication_protocol_review"),
    ),
)

MIN_RETRIEVAL_SCORE = 0.14
MIN_RULE_SCORE = 0.26
MIN_TRIGGER_HITS = 1
DEFAULT_RULE_PACK_DIR = Path(__file__).resolve().parent.parent / "rule_packs"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword.strip():
        return keyword in text
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None


def _rule_from_dict(payload: dict) -> ViolationRule:
    return ViolationRule(
        code=str(payload["code"]),
        label=str(payload["label"]),
        severity=str(payload["severity"]),
        trigger_keywords=tuple(str(item).lower() for item in payload.get("trigger_keywords", [])),
        evidence_keywords=tuple(str(item).lower() for item in payload.get("evidence_keywords", [])),
        action_tags=tuple(str(item) for item in payload.get("action_tags", [])),
    )


def load_rules(rule_pack_dir: Path | None = None) -> tuple[ViolationRule, ...]:
    directory = rule_pack_dir or DEFAULT_RULE_PACK_DIR
    if not directory.exists():
        return DEFAULT_RULES

    loaded_rules: list[ViolationRule] = []
    for file_path in sorted(directory.glob("*.json")):
        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        entries = content.get("rules", []) if isinstance(content, dict) else content
        if not isinstance(entries, list):
            continue

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            required = {"code", "label", "severity"}
            if not required.issubset(entry.keys()):
                continue
            loaded_rules.append(_rule_from_dict(entry))

    if not loaded_rules:
        return DEFAULT_RULES

    deduped: dict[str, ViolationRule] = {}
    for rule in loaded_rules:
        deduped[rule.code] = rule
    return tuple(deduped.values())


def score_rule(claim_text: str, evidence_text: str, rule: ViolationRule) -> float:
    claim_norm = normalize_text(claim_text)
    evidence_norm = normalize_text(evidence_text)

    trigger_hits = sum(1 for token in rule.trigger_keywords if contains_keyword(claim_norm, token))
    evidence_hits = sum(1 for token in rule.evidence_keywords if contains_keyword(evidence_norm, token))

    trigger_score = trigger_hits / max(len(rule.trigger_keywords), 1)
    evidence_score = evidence_hits / max(len(rule.evidence_keywords), 1)
    return 0.65 * trigger_score + 0.35 * evidence_score


def keyword_hits(text: str, keywords: tuple[str, ...]) -> int:
    text_norm = normalize_text(text)
    return sum(1 for token in keywords if contains_keyword(text_norm, token))


def is_actionable_claim(claim: dict) -> bool:
    claim_type = str(claim.get("claim_type", ""))
    claim_text = normalize_text(str(claim.get("claim_text", "")))

    if claim_type == "uncertainty_note":
        return False

    if claim_type == "audio_observation":
        return any(contains_keyword(claim_text, token) for token in ("warning", "beep", "alarm", "stand clear", "stop"))

    if claim_type == "observation":
        risk_tokens = (
            "walk",
            "stands",
            "tow",
            "towed",
            "wingtip",
            "close",
            "without",
            "no ",
            "engine",
            "fuel",
            "beacon",
            "warning",
            "pushback",
            "thrust",
            "exhaust",
            "heat distortion",
            "signal",
            "continues",
        )
        return any(contains_keyword(claim_text, token.strip()) for token in risk_tokens)

    return True


def filter_evidence_for_rule(evidence_items: list[dict], rule: ViolationRule) -> list[dict]:
    filtered = []
    min_retrieval_score = MIN_RETRIEVAL_SCORE if rule.severity == "high" else 0.08
    min_evidence_hits = 1 if rule.severity == "high" else 0

    for item in evidence_items:
        text = str(item.get("text", ""))
        retrieval_score = float(item.get("score", 0.0))
        evidence_keyword_hits = keyword_hits(text, rule.evidence_keywords)
        if retrieval_score < min_retrieval_score:
            continue
        if evidence_keyword_hits < min_evidence_hits:
            continue
        filtered.append(item)

    filtered.sort(key=lambda obj: float(obj.get("score", 0.0)), reverse=True)
    return filtered


def pick_primary_evidence(evidence_items: list[dict]) -> tuple[str, list[str]]:
    citations: list[str] = []
    excerpts: list[str] = []
    for item in evidence_items[:3]:
        citation = str(item.get("citation", ""))
        text = str(item.get("text", "")).strip()
        if citation:
            citations.append(citation)
        if text:
            excerpts.append(text[:260])
    return " | ".join(excerpts), list(dict.fromkeys(citations))


def calibrate_confidence(rule_score: float, supporting_evidence_count: int, avg_retrieval_score: float) -> float:
    confidence = 0.2 + (0.45 * rule_score) + (0.25 * min(supporting_evidence_count, 3) / 3) + (0.1 * avg_retrieval_score)
    if supporting_evidence_count <= 1:
        confidence = min(confidence, 0.74)
    return round(min(0.88, confidence), 3)


def apply_severity_guardrail(original_severity: str, supporting_evidence_count: int, avg_retrieval_score: float) -> str:
    if original_severity != "high":
        return original_severity
    if supporting_evidence_count <= 1 and avg_retrieval_score < 0.22:
        return "medium"
    return original_severity


def map_claim_to_violation(claim: dict, rules: tuple[ViolationRule, ...]) -> dict | None:
    claim_text = str(claim.get("claim_text", ""))
    evidence = claim.get("evidence", [])
    if not claim_text or not evidence:
        return None
    if not is_actionable_claim(claim):
        return None

    claim_text_norm = normalize_text(claim_text)
    ranked_candidates: list[tuple[ViolationRule, float, list[dict]]] = []

    for rule in rules:
        trigger_hits = sum(1 for token in rule.trigger_keywords if contains_keyword(claim_text_norm, token))
        if trigger_hits < MIN_TRIGGER_HITS:
            continue

        filtered_evidence = filter_evidence_for_rule(evidence[:5], rule)
        if not filtered_evidence:
            continue

        evidence_text = "\n".join(str(item.get("text", "")) for item in filtered_evidence[:3])
        avg_retrieval_score = sum(float(item.get("score", 0.0)) for item in filtered_evidence[:3]) / max(len(filtered_evidence[:3]), 1)
        rule_score = score_rule(claim_text, evidence_text, rule) + (0.12 * avg_retrieval_score)

        if rule.severity == "high" and trigger_hits < 2 and avg_retrieval_score < 0.24:
            continue

        ranked_candidates.append((rule, rule_score, filtered_evidence))

    if not ranked_candidates:
        return None

    ranked_candidates.sort(key=lambda item: item[1], reverse=True)
    best_rule, best_score, best_evidence = ranked_candidates[0]
    if best_score < MIN_RULE_SCORE:
        return None

    rationale_excerpt, citations = pick_primary_evidence(best_evidence)
    avg_retrieval_score = sum(float(item.get("score", 0.0)) for item in best_evidence[:3]) / max(len(best_evidence[:3]), 1)
    confidence = calibrate_confidence(
        rule_score=best_score,
        supporting_evidence_count=len(best_evidence),
        avg_retrieval_score=avg_retrieval_score,
    )
    severity = apply_severity_guardrail(
        original_severity=best_rule.severity,
        supporting_evidence_count=len(best_evidence),
        avg_retrieval_score=avg_retrieval_score,
    )

    return {
        "violation_code": best_rule.code,
        "violation_label": best_rule.label,
        "severity": severity,
        "confidence": confidence,
        "claim_id": claim.get("claim_id"),
        "claim_type": claim.get("claim_type"),
        "claim_text": claim_text,
        "rationale": rationale_excerpt,
        "citations": citations,
        "guardrail": {
            "supporting_evidence_count": len(best_evidence),
            "avg_retrieval_score": round(avg_retrieval_score, 3),
            "rule_score": round(best_score, 3),
        },
        "action_tags": list(best_rule.action_tags),
    }


def map_retrieval_payload(payload: dict, rule_pack_dir: Path | None = None) -> dict:
    rules = load_rules(rule_pack_dir)
    claims = payload.get("claims", [])
    mapped_violations = []
    unmapped_claims = []

    for claim in claims:
        mapped = map_claim_to_violation(claim, rules)
        if mapped is None:
            unmapped_claims.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "claim_text": claim.get("claim_text"),
                    "reason": "no_rule_above_threshold",
                }
            )
            continue
        mapped_violations.append(mapped)

    best_by_code: dict[str, dict] = {}
    for item in mapped_violations:
        code = str(item.get("violation_code", ""))
        if not code:
            continue
        existing = best_by_code.get(code)
        if existing is None or float(item.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
            best_by_code[code] = item

    deduped_mapped = sorted(best_by_code.values(), key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    return {
        "clip_id": payload.get("clip_id", "unknown_clip"),
        "stage": "llm2_claim_to_violation",
        "mapped_violations": deduped_mapped,
        "unmapped_claims": unmapped_claims,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rule-based mapper for claim-to-violation with citations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/retrieval_sample.json"),
        help="Input retrieval payload JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/violations_sample.json"),
        help="Output mapped violations JSON",
    )
    parser.add_argument(
        "--rule-pack-dir",
        type=Path,
        default=DEFAULT_RULE_PACK_DIR,
        help="Directory containing JSON rule packs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        retrieval_payload = json.load(f)

    result = map_retrieval_payload(retrieval_payload, rule_pack_dir=args.rule_pack_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote violation mapping to {args.output}")
