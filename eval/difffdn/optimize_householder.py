#!/usr/bin/env python3
"""Run tiny DiffFDN optimization and export a JSON preset.

SOURCE:
- diff-fdn-colorless framing (Dal Santo et al.), pinned commit:
  https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c
- This script is a project-local offline optimizer/preset exporter and does not
  run inside the realtime plugin/audio callback.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from difffdn.difffdn_tiny import (  # type: ignore
        default_io_vectors,
        evaluate_transfer_losses,
        gamma_from_rt60,
        gains_from_gamma,
        optimize_householder,
        normalize_l2,
        rt60_from_gamma,
    )
else:
    from .difffdn_tiny import (
        default_io_vectors,
        evaluate_transfer_losses,
        gamma_from_rt60,
        gains_from_gamma,
        optimize_householder,
        normalize_l2,
        rt60_from_gamma,
    )


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


def _loss_summary(losses: Dict[str, float]) -> str:
    return (
        f"total={losses['total']:.6e} "
        f"spectral={losses['spectral']:.6e} "
        f"sparsity={losses['sparsity']:.6e}"
    )


def _evaluate_householder_loss(
    *,
    fs: float,
    nfft: int,
    delay_samples: List[int],
    gamma: float,
    M: int,
    dtype: torch.dtype,
    alpha_sparsity: float,
    spectral_mode: str,
    train_lossless: bool,
    u: torch.Tensor,
    b: torch.Tensor,
    c_l: torch.Tensor,
    c_r: torch.Tensor,
) -> Dict[str, float]:
    if train_lossless:
        gains = torch.ones((len(delay_samples),), dtype=dtype, device=torch.device("cpu"))
    else:
        gains = gains_from_gamma(delay_samples, gamma, dtype=dtype).to(torch.device("cpu"))
    losses = evaluate_transfer_losses(
        sr=fs,
        nfft=nfft,
        delay_samples=delay_samples,
        gains=gains,
        matrix_type="householder",
        u=u,
        b=b,
        c_l=c_l,
        c_r=c_r,
        alpha_density=alpha_sparsity,
        spectral_mode=spectral_mode,
        freq_grid_size=max(int(M), 2),
    )
    return {
        "total": float(losses["total"].detach().cpu()),
        "spectral": float(losses["spectral"].detach().cpu()),
        "sparsity": float(losses["sparsity"].detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", default="householder_prime_rt60_2p8_paper")
    parser.add_argument("--matrix-type", choices=["householder", "hadamard"], default="householder")
    parser.add_argument("--fs", type=float, default=48000.0, help="Sample rate in Hz (paper default: 48000)")
    parser.add_argument("--sr", type=float, default=None, help="Deprecated alias for --fs")
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--M", type=int, default=480000, help="Frequency-grid points over [0, pi]")
    parser.add_argument("--batch", type=int, default=2000, help="Sampled frequency points per step")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch count; steps/epoch ~= M/batch")
    parser.add_argument("--delay-samples", default="1499,2377,3217,4421")
    parser.add_argument("--rt60", type=float, default=2.8)
    parser.add_argument("--steps", type=int, default=None, help="Optional explicit total steps override")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Optional explicit per-sample decay gamma. If omitted, derived from --rt60.",
    )
    parser.add_argument("--alpha-sparsity", type=float, default=0.05)
    parser.add_argument("--alpha-density", type=float, default=None, help="Deprecated alias for --alpha-sparsity")
    parser.add_argument(
        "--spectral-mode",
        choices=["unity", "mean"],
        default="unity",
        help="Spectral objective mode (recommended default for paper alignment: unity).",
    )
    parser.add_argument(
        "--train-lossless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train with unit loop gain (paper-style colorless core objective).",
    )
    parser.add_argument(
        "--optimize-with-decay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimize using decay gains (experimental path). Overrides --train-lossless.",
    )
    parser.add_argument("--learn-io", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--out-dir", default="eval/out/presets")
    parser.add_argument("--out-json", default=None, help="Optional explicit output path")
    parser.add_argument("--history-json", default=None, help="Optional history output path")
    parser.add_argument(
        "--skip-baseline-compare",
        action="store_true",
        help="Skip fixed/u-only/full baseline comparison printout",
    )
    args = parser.parse_args()

    delay_samples = _parse_delay_csv(args.delay_samples)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    fs = float(args.sr) if args.sr is not None else float(args.fs)
    alpha_sparsity = (
        float(args.alpha_density) if args.alpha_density is not None else float(args.alpha_sparsity)
    )
    batch = max(int(args.batch), 1)
    M = max(int(args.M), 2)
    epochs = max(int(args.epochs), 1)
    steps_per_epoch = max(1, (M + batch - 1) // batch)
    total_steps = int(args.steps) if args.steps is not None else (epochs * steps_per_epoch)
    gamma_used = float(args.gamma) if args.gamma is not None else gamma_from_rt60(fs, args.rt60)
    gamma_source = "explicit_gamma" if args.gamma is not None else "rt60_target"
    train_lossless = bool(args.train_lossless) and (not bool(args.optimize_with_decay))
    training_mode = "lossless-core" if train_lossless else "with-decay"
    gamma_train = 1.0 if train_lossless else gamma_used
    print(
        f"[Config] fs={fs:.1f} rt60_target={float(args.rt60):.4f}s "
        f"gamma_used={gamma_used:.9f} gamma_train={gamma_train:.9f} "
        f"mode={training_mode} spectral_mode={args.spectral_mode} steps={total_steps} "
        f"(epochs={epochs}, steps_per_epoch={steps_per_epoch}, batch={batch}, M={M})"
    )

    result = optimize_householder(
        sr=fs,
        nfft=args.nfft,
        delay_samples=delay_samples,
        rt60=args.rt60,
        matrix_type=args.matrix_type,
        steps=total_steps,
        epochs=epochs,
        M=M,
        batch_size=batch,
        lr=args.lr,
        gamma=gamma_used,
        train_lossless=train_lossless,
        alpha_density=alpha_sparsity,
        learn_io=args.learn_io,
        freq_bins_per_step=batch,
        spectral_mode=args.spectral_mode,
        seed=args.seed,
        dtype=dtype,
        device="cpu",
        log_every=max(total_steps // 24, 1),
    )

    gains_runtime = gains_from_gamma(delay_samples, gamma_used, dtype=dtype).to(torch.device("cpu"))
    preset = {
        "config_id": args.config_id,
        "sr": int(fs),
        "fs": int(fs),
        "nfft": int(args.nfft),
        "M": int(M),
        "batch": int(batch),
        "epochs": int(epochs),
        "delay_samples": [int(v) for v in delay_samples],
        "rt60": float(args.rt60),  # legacy alias
        "rt60_target": float(args.rt60),
        "steps": int(total_steps),
        "steps_per_epoch": int(steps_per_epoch),
        "lr": float(args.lr),
        "gamma": float(gamma_used),  # legacy alias
        "gamma_used": float(gamma_used),
        "gamma_source": gamma_source,
        "train_lossless": bool(train_lossless),
        "optimize_with_decay": bool(not train_lossless),
        "gamma_train": float(gamma_train),
        "alpha_density": float(alpha_sparsity),
        "alpha_sparsity": float(alpha_sparsity),
        "spectral_mode": args.spectral_mode,
        "freq_bins_per_step": int(batch),
        "learn_io": bool(args.learn_io),
        "seed": int(args.seed),
        "gains": _to_float_list(gains_runtime),
        "gains_training": _to_float_list(result.gains),
        "matrix_type": args.matrix_type,
        "fixed_u": [0.5, 0.5, 0.5, 0.5],
        "u": _to_float_list(result.u),
        "b": _to_float_list(result.b),
        "cL": _to_float_list(result.c_l),
        "cR": _to_float_list(result.c_r),
        "losses": {
            **result.losses,
        },
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

    if args.matrix_type == "householder" and not args.skip_baseline_compare:
        device = torch.device("cpu")
        fixed_u = normalize_l2(torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype, device=device))
        b_default, c_l_default, c_r_default = default_io_vectors(4, device=device, dtype=dtype)

        fixed_loss = _evaluate_householder_loss(
            fs=fs,
            nfft=args.nfft,
            delay_samples=delay_samples,
            gamma=gamma_used,
            M=M,
            dtype=dtype,
            alpha_sparsity=alpha_sparsity,
            spectral_mode=args.spectral_mode,
            train_lossless=train_lossless,
            u=fixed_u,
            b=b_default,
            c_l=c_l_default,
            c_r=c_r_default,
        )

        if args.learn_io:
            learned_u_only = optimize_householder(
                sr=fs,
                nfft=args.nfft,
                delay_samples=delay_samples,
                rt60=args.rt60,
                matrix_type=args.matrix_type,
                steps=total_steps,
                epochs=epochs,
                M=M,
                batch_size=batch,
                lr=args.lr,
                gamma=gamma_used,
                train_lossless=train_lossless,
                alpha_density=alpha_sparsity,
                learn_io=False,
                freq_bins_per_step=batch,
                spectral_mode=args.spectral_mode,
                seed=args.seed,
                dtype=dtype,
                device="cpu",
                log_every=0,
            ).losses
            learned_full = result.losses
        else:
            learned_u_only = result.losses
            learned_full = optimize_householder(
                sr=fs,
                nfft=args.nfft,
                delay_samples=delay_samples,
                rt60=args.rt60,
                matrix_type=args.matrix_type,
                steps=total_steps,
                epochs=epochs,
                M=M,
                batch_size=batch,
                lr=args.lr,
                gamma=gamma_used,
                train_lossless=train_lossless,
                alpha_density=alpha_sparsity,
                learn_io=True,
                freq_bins_per_step=batch,
                spectral_mode=args.spectral_mode,
                seed=args.seed,
                dtype=dtype,
                device="cpu",
                log_every=0,
            ).losses

        print("[Baseline] fixed-u/default-io:", _loss_summary(fixed_loss))
        print("[Baseline] learned-u-only:   ", _loss_summary(learned_u_only))
        print("[Baseline] learned-full:     ", _loss_summary(learned_full))

    implied_rt60 = rt60_from_gamma(fs, gamma_used)
    print(
        f"[Sanity] gamma_used={gamma_used:.9f} -> implied_rt60={implied_rt60:.6f}s "
        f"(target={float(args.rt60):.6f}s, source={gamma_source})"
    )

    print(f"Wrote preset: {out_path}")
    print(f"Wrote history: {history_path}")


if __name__ == "__main__":
    main()
