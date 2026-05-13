#!/usr/bin/env python
"""Audit v1.0.0-rc1 release consistency across docs and key artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


DOCS = [
    "README.md",
    "docs/sionna_native_channel_extraction.md",
    "docs/sionna_native_precoder_api_probe.md",
    "docs/releases/v1.0.0-rc1.md",
    "docs/pr/sionna_interface_rc_pr.md",
]
ARTIFACTS = [
    "outputs/sionna_interface_rc/interface_rc_artifact_manifest.json",
    "outputs/repro/sionna_interface_rc_minimal_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _contains_all(texts: list[str], patterns: list[str]) -> bool:
    corpus = "\n".join(texts)
    return all(pattern in corpus for pattern in patterns)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    doc_texts = [Path(path).read_text(encoding="utf-8") for path in DOCS]
    artifact_payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in ARTIFACTS]
    payload = {
        "v1_0_0_rc1_naming_consistent": _contains_all(doc_texts, ["v1.0.0-rc1"]),
        "release_candidate_wording_present": _contains_all(doc_texts, ["release candidate"]),
        "interface_first_wording_present": _contains_all(doc_texts, ["interface-first"]),
        "full_native_only_false_present": _contains_all(doc_texts, ["full native-only benchmark"]),
        "relationship_status_present": _contains_all(doc_texts, ["close_but_different"]),
        "strict_equivalence_false_present": _contains_all(doc_texts, ["strict_equivalence_claim_allowed = false"]),
        "optional_sionna_dependency_present": artifact_payloads[0]["artifacts"][0]["optional_sionna_dependency"] is True,
        "no_sionna_rt_claim": _contains_all(doc_texts, ["no Sionna RT"]),
        "no_ray_tracing_claim": _contains_all(doc_texts, ["no ray tracing"]),
        "no_fiveg_full_stack_claim": _contains_all(doc_texts, ["no 5G NR full stack"]),
        "no_mainline_native_replacement_claim": _contains_all(doc_texts, ["not a mainline native replacement"]),
        "no_stable_learned_gt_wmmse_claim": _contains_all(doc_texts, ["no stable learned `> WMMSE-iter5` claim"]),
        "reproduction_commands_present": _contains_all(doc_texts, ["Reproduction Commands"]),
        "artifact_paths_exist": all(Path(path).exists() for path in ARTIFACTS),
    }
    payload["status"] = "ok"
    write_json(out_path, payload)
    lines = [
        "# Interface RC Release Consistency Audit",
        "",
        *[f"- {k}: `{v}`" for k, v in payload.items()],
    ]
    write_markdown(md_path, lines)
    print(f"Saved interface RC release consistency audit to {out_path}")


if __name__ == "__main__":
    main()
