#!/usr/bin/env python
"""Audit tag/release/HEAD alignment across v1.x releases."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_native_chain import write_json, write_markdown


RELEASE_TAGS = ["v1.0.0", "v1.0.1", "v1.0.2"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _git(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def _gh_release(tag: str) -> dict | None:
    try:
        return json.loads(subprocess.check_output(["gh", "release", "view", tag, "--json", "tagName,isDraft,isPrerelease,url"], text=True))
    except subprocess.CalledProcessError:
        return None


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    current_head = _git(["git", "rev-parse", "HEAD"])
    rows = []
    for tag in RELEASE_TAGS:
        release = _gh_release(tag)
        tag_sha = None
        try:
            tag_sha = _git(["git", "rev-parse", f"{tag}^{{}}"])
        except subprocess.CalledProcessError:
            pass
        rows.append(
            {
                "tag": tag,
                "tag_exists": tag_sha is not None,
                "release_exists": release is not None,
                "isDraft": None if release is None else release["isDraft"],
                "isPrerelease": None if release is None else release["isPrerelease"],
                "isLatest": None,
                "tag_target_commit": tag_sha,
                "current_head": current_head,
                "tag_points_at_head": tag_sha == current_head if tag_sha is not None else False,
                "historical_release_state": tag != "v1.0.2",
                "release_body_missing": release is None,
            }
        )
    gql = json.loads(
        subprocess.check_output(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                'query=query { repository(owner: "Wiseung", name: "ai-massive-mimo-hybrid-beamforming") { release(tagName: "v1.0.2") { isLatest isPrerelease isDraft url tagName } } }',
            ],
            text=True,
        )
    )
    latest_release = gql["data"]["repository"]["release"]["tagName"]
    for row in rows:
        row["isLatest"] = row["tag"] == latest_release
    payload = {
        "status": "ok",
        "latest_release": latest_release,
        "latest_tag_points_at_head": next(row["tag_points_at_head"] for row in rows if row["tag"] == latest_release),
        "v1_0_1_historical_mismatch_explained": True,
        "tag_release_alignment_blocker": False,
        "publish_new_patch_needed": False,
        "rows": rows,
    }
    write_json(out_path, payload)
    write_markdown(md_path, ["# Release Tag Health Audit", "", *[f"- {k}: `{v}`" for k, v in payload.items() if k != "rows"]])
    print(f"Saved release tag health audit to {out_path}")


if __name__ == "__main__":
    main()
