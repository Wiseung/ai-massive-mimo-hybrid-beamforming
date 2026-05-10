#!/usr/bin/env python
"""Check whether the optional Sionna environment is available."""

from __future__ import annotations

from _bootstrap import add_src_to_path

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info, format_sionna_env_lines


def main() -> None:
    info = collect_sionna_env_info()
    for line in format_sionna_env_lines(info):
        print(line)


if __name__ == "__main__":
    main()
