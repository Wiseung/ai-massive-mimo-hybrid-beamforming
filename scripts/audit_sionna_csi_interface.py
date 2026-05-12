#!/usr/bin/env python
"""Audit ExtractedCSI provenance for the Sionna-native channel extraction path."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path
import torch

add_src_to_path()

from beamforming.utils.sionna_env import collect_sionna_env_info
from beamforming.utils.sionna_native_beamforming_chain import compute_project_precoder_per_subcarrier
from beamforming.utils.sionna_native_chain import write_json, write_markdown
from beamforming.utils.sionna_native_learned_beamforming import (
    build_native_receiver_context,
    default_checkpoint_path,
    infer_learned_precoder,
    load_learned_beamformer_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _to_md(payload: dict) -> list[str]:
    return [
        "# Sionna CSI Interface Audit",
        "",
        f"- status: `{payload['status']}`",
        f"- csi_interface_used: `{payload['csi_interface_used']}`",
        f"- h_f_shape_ok: `{payload['h_f_shape_ok']}`",
        f"- axes_metadata_complete: `{payload['axes_metadata_complete']}`",
        f"- selected_data_symbol_not_pilot: `{payload['selected_data_symbol_not_pilot']}`",
        f"- effective_subcarrier_count_matches_nsc: `{payload['effective_subcarrier_count_matches_nsc']}`",
        f"- project_h_f_assisted: `{payload['project_h_f_assisted']}`",
        f"- extracted_h_f_used: `{payload['extracted_h_f_used']}`",
        f"- full_native_only: `{payload['full_native_only']}`",
        f"- project_rzf_consumes_csi: `{payload['project_rzf_consumes_csi']}`",
        f"- learned_residual_rzf_consumes_csi: `{payload['learned_residual_rzf_consumes_csi']}`",
        "",
        "## Notes",
        *[f"- {note}" for note in payload["notes"]],
    ]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    md_path = out_path.with_suffix(".md")
    env = collect_sionna_env_info()
    payload = {
        "status": "skipped",
        "sionna_import_ok": env["sionna_import_ok"],
        "sionna_version": env["sionna_version"],
        "csi_interface_used": False,
        "h_f_shape_ok": False,
        "axes_metadata_complete": False,
        "original_sionna_h_shape_present": False,
        "selected_data_symbol_not_pilot": False,
        "effective_subcarrier_count_matches_nsc": False,
        "project_h_f_assisted": True,
        "extracted_h_f_used": False,
        "full_native_only": False,
        "project_rzf_consumes_csi": False,
        "learned_residual_rzf_consumes_csi": False,
        "csi_summary": None,
        "notes": [],
    }
    if not env["sionna_import_ok"]:
        payload["notes"].append("Sionna not installed; optional dependency path skipped.")
        write_json(out_path, payload)
        write_markdown(md_path, _to_md(payload))
        print(f"Saved CSI interface audit to {out_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_root = Path(__file__).resolve().parents[1]
    context = build_native_receiver_context(
        batch_size=8,
        num_subcarriers=16,
        num_users=4,
        num_bs_ant=16,
        snr_db=10.0,
        device=device,
    )
    csi = context.csi
    if csi is None:
        payload["status"] = "failed"
        payload["notes"].append("Native receiver context did not expose a CSI object.")
        write_json(out_path, payload)
        write_markdown(md_path, _to_md(payload))
        print(f"Saved CSI interface audit to {out_path}")
        return

    summary = csi.summary_dict()
    metadata = summary["metadata"]
    payload.update(
        {
            "status": "ok",
            "csi_interface_used": True,
            "h_f_shape_ok": summary["h_f_shape"] == [8, 16, 4, 16],
            "axes_metadata_complete": summary["axes"] == {"B": 0, "Nsc": 1, "K": 2, "Nt": 3},
            "original_sionna_h_shape_present": bool(metadata.get("original_sionna_h_shape")),
            "selected_data_symbol_not_pilot": int(summary["selected_ofdm_symbol"]) not in set(metadata.get("pilot_symbol_indices", [])),
            "effective_subcarrier_count_matches_nsc": len(summary["effective_subcarrier_indices"]) == int(summary["num_subcarriers"]),
            "project_h_f_assisted": bool(summary["project_h_f_assisted"]),
            "extracted_h_f_used": bool(summary["extracted_h_f_used"]),
            "full_native_only": bool(summary["full_native_only"]),
            "csi_summary": summary,
        }
    )

    try:
        precoder = compute_project_precoder_per_subcarrier("rzf", csi.to_project_h_f(), context.noise_var)
        payload["project_rzf_consumes_csi"] = list(precoder.shape) == [8, 16, 16, 4]
    except Exception as exc:
        payload["notes"].append(f"project_rzf consumption failed: {type(exc).__name__}: {exc}")

    ckpt = default_checkpoint_path("learned_residual_rzf", repo_root)
    if ckpt.exists():
        try:
            bundle = load_learned_beamformer_checkpoint(ckpt, device, method_name="learned_residual_rzf")
            snr_tensor = torch.full((csi.h_f.size(0),), context.snr_db, dtype=torch.float32, device=device)
            _, infer_meta, _ = infer_learned_precoder(bundle, csi.to_project_h_f(), snr_tensor, native_receiver_path=True)
            payload["learned_residual_rzf_consumes_csi"] = True
            payload["teacher_used_during_inference"] = bool(infer_meta["teacher_used_during_inference"])
        except Exception as exc:
            payload["notes"].append(f"learned_residual_rzf consumption failed: {type(exc).__name__}: {exc}")
    else:
        payload["notes"].append("learned_residual_rzf checkpoint missing; learned consumption audit skipped.")

    write_json(out_path, payload)
    write_markdown(md_path, _to_md(payload))
    print(f"Saved CSI interface audit to {out_path}")


if __name__ == "__main__":
    main()
