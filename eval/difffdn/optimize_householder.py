#!/usr/bin/env python3
"""Run tiny DiffFDN optimization and export a JSON preset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from difffdn.difffdn_tiny import optimize_householder  # type: ignore
else:
    from .difffdn_tiny import optimize_householder


def _parse_delay_csv(text: str) -> List[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    values = [int(p) for p in parts]
    if len(values) != 4:
        raise ValueError(f"Expected exactly 4 delay samples, got {values}")
    if any(v <= 0 for v in values):
        raise ValueError(f"Delay samples must be positive, got {values}")
    return values


def _to_float_list(x: torch.Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", default="householder_prime_rt60_2p8")
    parser.add_argument("--matrix-type", choices=["householder", "hadamard"], default="householder")
    parser.add_argument("--sr", type=float, default=48000.0)
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--delay-samples", default="1499,2377,3217,4421")
    parser.add_argument("--rt60", type=float, default=2.8)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--alpha-density", type=float, default=0.0)
    parser.add_argument("--freq-bins-per-step", type=int, default=256)
    parser.add_argument("--learn-io", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out-dir", default="eval/out/presets")
    parser.add_argument("--out-json", default=None, help="Optional explicit output path")
    parser.add_argument("--history-json", default=None, help="Optional history output path")
    args = parser.parse_args()

    delay_samples = _parse_delay_csv(args.delay_samples)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    result = optimize_householder(
        sr=args.sr,
        nfft=args.nfft,
        delay_samples=delay_samples,
        rt60=args.rt60,
        matrix_type=args.matrix_type,
        steps=args.steps,
        lr=args.lr,
        alpha_density=args.alpha_density,
        learn_io=args.learn_io,
        freq_bins_per_step=args.freq_bins_per_step,
        seed=args.seed,
        dtype=dtype,
        device="cpu",
        log_every=max(args.steps // 8, 1),
    )

    preset = {
        "config_id": args.config_id,
        "sr": int(args.sr),
        "nfft": int(args.nfft),
        "delay_samples": [int(v) for v in delay_samples],
        "rt60": float(args.rt60),
        "gains": _to_float_list(result.gains),
        "matrix_type": args.matrix_type,
        "u": _to_float_list(result.u),
        "b": _to_float_list(result.b),
        "cL": _to_float_list(result.c_l),
        "cR": _to_float_list(result.c_r),
        "losses": {
            **result.losses,
            "alpha_density": float(args.alpha_density),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "freq_bins_per_step": int(args.freq_bins_per_step),
            "learn_io": bool(args.learn_io),
        },
        "seed": int(args.seed),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out_json) if args.out_json else (out_dir / f"{args.config_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(preset, indent=2) + "\n")

    if args.history_json:
        history_path = Path(args.history_json)
    else:
        history_path = out_path.with_suffix(".history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(result.history, indent=2) + "\n")

    print(f"Wrote preset: {out_path}")
    print(f"Wrote history: {history_path}")


if __name__ == "__main__":
    main()
