#!/usr/bin/env python
"""Audit provenance integrity for interface RC artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    rc_manifest = _load_json(Path("outputs/sionna_interface_rc/interface_rc_artifact_manifest.json"))
    current_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    release_tag = "v1.0.0-rc1"
    rows = []
    for row in rc_manifest["artifacts"]:
        path = Path(row["path"])
        rows.append(
            {
                "name": row["name"],
                "path_exists": path.exists(),
                "generated_from_commit_present": bool(row.get("generated_from_commit")),
                "generated_from_commit": row.get("generated_from_commit"),
                "current_head": current_head,
                "release_tag": release_tag,
                "full_native_only": row.get("full_native_only"),
                "sionna_rt_used": row.get("sionna_rt_used"),
                "ray_tracing_used": row.get("ray_tracing_used"),
                "fiveg_full_stack_used": row.get("fiveg_full_stack_used"),
            }
        )
    payload = {
        "status": "ok",
        "current_head": current_head,
        "release_tag": release_tag,
        "all_artifacts_exist": all(r["path_exists"] for r in rows),
        "all_generated_from_commit_present": all(r["generated_from_commit_present"] for r in rows),
        "all_full_native_only_false": all(r["full_native_only"] is False for r in rows),
        "all_rt_ray_5g_flags_false": all(
            (r["sionna_rt_used"] is False and r["ray_tracing_used"] is False and r["fiveg_full_stack_used"] is False)
            for r in rows
        ),
        "artifacts": rows,
    }
    write_json(out_path, payload)
    lines = [
        "# Interface RC Artifact Provenance Audit",
        "",
        f"- current_head: `{current_head}`",
        f"- release_tag: `{release_tag}`",
        f"- all_artifacts_exist: `{payload['all_artifacts_exist']}`",
        f"- all_generated_from_commit_present: `{payload['all_generated_from_commit_present']}`",
        f"- all_full_native_only_false: `{payload['all_full_native_only_false']}`",
        f"- all_rt_ray_5g_flags_false: `{payload['all_rt_ray_5g_flags_false']}`",
    ]
    write_markdown(md_path, lines)
    print(f"Saved interface RC artifact provenance audit to {out_path}")


if __name__ == "__main__":
    main()
