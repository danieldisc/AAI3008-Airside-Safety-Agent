from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import INDEX_DIR
from rag.incident_retrieval import retrieve_for_incident
from rag.llm2_mapper import DEFAULT_RULE_PACK_DIR, map_retrieval_payload
from rag.llm3_teachable import build_coaching_payload
from rag.llm4_report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full incident pipeline: retrieval -> violations -> teachable -> report"
    )
    parser.add_argument("--incident-json", type=Path, default=None)
    parser.add_argument(
        "--video-path",
        type=Path,
        default=None,
        help="Optional video path to generate incident JSON via real VLM",
    )
    parser.add_argument(
        "--vlm-engine",
        type=str,
        default="Gemini",
        choices=("Gemini", "OpenAI"),
        help="VLM backend when --video-path is provided",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Optional clip id override for VLM-generated incident payload",
    )
    parser.add_argument(
        "--airside-src-dir",
        type=Path,
        default=None,
        help="Optional path to AAI3008-Airside-Safety-Agent/src",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--rule-pack-dir", type=Path, default=DEFAULT_RULE_PACK_DIR)
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild manual index before running retrieval",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def run_pipeline(
    incident_json_path: Path,
    video_path: Path | None,
    vlm_engine: str,
    clip_id: str | None,
    airside_src_dir: Path | None,
    output_dir: Path,
    top_k: int,
    index_dir: Path,
    rule_pack_dir: Path,
    rebuild_index: bool,
) -> dict:
    if rebuild_index:
        from rag.build_index import build_index

        print("[1/5] Rebuilding index...")
        build_index(index_dir)

    if video_path is not None:
        print("[2/5] Running VLM incident extraction...")
        from rag.vlm_incident import build_video_analysis_artifacts

        incident_payload, analyst_report_text, _observer_logs = build_video_analysis_artifacts(
            video_path=video_path,
            engine=vlm_engine,
            clip_id=clip_id,
            airside_src_dir=airside_src_dir,
        )
        write_json(output_dir / "incident_from_vlm.json", incident_payload)
        write_text(output_dir / "analyst_report.txt", analyst_report_text)
    else:
        if incident_json_path is None:
            raise ValueError("Provide --incident-json <path> or --video-path <path>")
        print("[2/5] Loading incident JSON...")
        with incident_json_path.open("r", encoding="utf-8") as f:
            incident_payload = json.load(f)

    print("[3/5] Running retrieval...")
    retrieval_payload = retrieve_for_incident(
        payload=incident_payload,
        index_dir=index_dir,
        top_k=top_k,
    )

    print("[4/5] Mapping violations + teachable moments...")
    violations_payload = map_retrieval_payload(retrieval_payload, rule_pack_dir=rule_pack_dir)
    teachable_payload = build_coaching_payload(violations_payload)

    print("[5/5] Building final report...")
    report_payload = build_report(
        incident_payload=incident_payload,
        retrieval_payload=retrieval_payload,
        violations_payload=violations_payload,
        teachable_payload=teachable_payload,
    )

    write_json(output_dir / "retrieval_sample.json", retrieval_payload)
    write_json(output_dir / "violations_sample.json", violations_payload)
    write_json(output_dir / "teachable_sample.json", teachable_payload)
    write_json(output_dir / "report_sample.json", report_payload)

    print("Pipeline complete.")
    if video_path is not None:
        print(f"- Incident JSON: {output_dir / 'incident_from_vlm.json'}")
        print(f"- Analyst report: {output_dir / 'analyst_report.txt'}")
    print(f"- Retrieval: {output_dir / 'retrieval_sample.json'}")
    print(f"- Violations: {output_dir / 'violations_sample.json'}")
    print(f"- Teachable: {output_dir / 'teachable_sample.json'}")
    print(f"- Report: {output_dir / 'report_sample.json'}")
    return report_payload


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        incident_json_path=args.incident_json,
        video_path=args.video_path,
        vlm_engine=args.vlm_engine,
        clip_id=args.clip_id,
        airside_src_dir=args.airside_src_dir,
        output_dir=args.output_dir,
        top_k=args.top_k,
        index_dir=args.index_dir,
        rule_pack_dir=args.rule_pack_dir,
        rebuild_index=args.rebuild_index,
    )
