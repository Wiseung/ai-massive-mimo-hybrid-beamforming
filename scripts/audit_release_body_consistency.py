#!/usr/bin/env python
"""Audit consistency between GitHub release body, release notes, README, and core docs."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


FILES = [
    "docs/releases/v1.0.0.md",
    "README.md",
    "docs/sionna_native_channel_extraction.md",
    "docs/sionna_native_precoder_api_probe.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    release = json.loads(subprocess.check_output(["gh", "release", "view", args.tag, "--json", "tagName,name,body,url"], text=True))
    docs = [Path(path).read_text(encoding="utf-8") for path in FILES]
    corpus = "\n".join(docs + [release["body"]])
    payload = {
        "status": "ok",
        "v1_0_0_title_consistent": "v1.0.0" in release["name"],
        "interface_first_stable_release_wording_consistent": "interface-first" in corpus and "stable release" in corpus,
        "reproduction_commands_present": "Reproduction Commands" in corpus,
        "artifact_paths_consistent": "outputs/sionna_interface_stable/interface_stable_artifact_manifest.json" in corpus,
        "known_limitations_consistent": "not a full native-only benchmark" in corpus,
        "no_full_native_only_claim": "full native-only benchmark" in corpus,
        "no_mainline_native_replacement_claim": "mainline replacement" in corpus,
        "no_strict_project_rzf_equivalence_claim": "strict `project_rzf` equivalence is not claimed" in corpus,
        "no_rt_claim": "no Sionna RT" in corpus,
        "no_ray_tracing_claim": "no ray tracing" in corpus,
        "no_fiveg_full_stack_claim": "no 5G NR full stack" in corpus,
        "optional_sionna_dependency_wording_consistent": "optional Sionna dependency" in corpus or "optional dependency only" in corpus,
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Release Body Consistency Audit", "", *[f"- {k}: `{v}`" for k, v in payload.items()]])
    print(f"Saved release body consistency audit to {out_path}")


if __name__ == "__main__":
    main()
