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
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from difffdn.difffdn_tiny import (  # type: ignore
        build_band_k_pool,
        default_io_vectors,
        evaluate_transfer_losses,
        gamma_from_rt60,
        gains_from_gamma,
        hz_to_k,
        k_to_hz,
        optimize_householder,
        normalize_l2,
        rt60_from_gamma,
        sparsity_loss_eq18,
    )
else:
    from .difffdn_tiny import (
        build_band_k_pool,
        default_io_vectors,
        evaluate_transfer_losses,
        gamma_from_rt60,
        gains_from_gamma,
        hz_to_k,
        k_to_hz,
        optimize_householder,
        normalize_l2,
        rt60_from_gamma,
        sparsity_loss_eq18,
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
    stability_gamma: float,
    M: int,
    dtype: torch.dtype,
    alpha_sparsity: float,
    spectral_mode: str,
    train_lossless: bool,
    paper_band_enable: bool,
    paper_band_min_hz: float,
    paper_band_max_hz: float,
    u: torch.Tensor,
    b: torch.Tensor,
    c_l: torch.Tensor,
    c_r: torch.Tensor,
) -> Dict[str, float]:
    if train_lossless:
        gains = gains_from_gamma(delay_samples, stability_gamma, dtype=dtype).to(torch.device("cpu"))
    else:
        gains = gains_from_gamma(delay_samples, gamma, dtype=dtype).to(torch.device("cpu"))
    if paper_band_enable:
        band_pool, _, _, _ = build_band_k_pool(
            sr=fs,
            freq_grid_size=max(int(M), 2),
            fmin_hz=paper_band_min_hz,
            fmax_hz=paper_band_max_hz,
            exclude_dc=True,
            device=torch.device("cpu"),
        )
        eval_k = band_pool
    else:
        eval_k = torch.arange(1, max(int(M), 2), dtype=torch.int64, device=torch.device("cpu"))
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
        k_indices=eval_k,
        freq_grid_size=max(int(M), 2),
        paper_band_enable=paper_band_enable,
        paper_band_min_hz=paper_band_min_hz,
        paper_band_max_hz=paper_band_max_hz,
    )
    return {
        "total": float(losses["total"].detach().cpu()),
        "spectral": float(losses["spectral"].detach().cpu()),
        "sparsity": float(losses["sparsity"].detach().cpu()),
    }


def _print_eq18_sanity(dtype: torch.dtype) -> None:
    n = 4
    eye = torch.eye(n, dtype=dtype)
    hadamard = 0.5 * torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0], [1.0, -1.0, -1.0, 1.0]],
        dtype=dtype,
    )
    u = normalize_l2(torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype))
    house = torch.eye(n, dtype=dtype) - 2.0 * torch.outer(u, u)

    s_eye = float(sparsity_loss_eq18(eye).detach().cpu())
    s_had = float(sparsity_loss_eq18(hadamard).detach().cpu())
    s_house = float(sparsity_loss_eq18(house).detach().cpu())
    print(
        "[Eq18 sanity] I={:.6f}, hadamard={:.6f}, householder_u=[0.5,..]={:.6f}".format(
            s_eye, s_had, s_house
        )
    )
    if not (s_eye > s_had):
        print(
            "[WARN] Eq18 direction unexpected for density claim: expected Eq18(I) > Eq18(Hadamard). "
            "Check alpha_density interpretation."
        )
    else:
        print("[Eq18 sanity] OK: Eq18 penalizes sparse identity more than dense Hadamard.")


def _print_paper_band_debug(
    *,
    fs: float,
    freq_grid_size: int,
    paper_band_enable: bool,
    paper_band_min_hz: float,
    paper_band_max_hz: float,
    batch_size: int,
    dtype: torch.dtype,
) -> Dict[str, int]:
    if paper_band_enable:
        pool, k_min, k_max, pool_count = build_band_k_pool(
            sr=fs,
            freq_grid_size=freq_grid_size,
            fmin_hz=paper_band_min_hz,
            fmax_hz=paper_band_max_hz,
            exclude_dc=True,
            device=torch.device("cpu"),
        )
        fmin_dbg = paper_band_min_hz
        fmax_dbg = paper_band_max_hz
    else:
        fmin_dbg = 0.0
        fmax_dbg = k_to_hz(
            sr=fs,
            freq_grid_size=freq_grid_size,
            k=torch.tensor(freq_grid_size - 1, dtype=torch.int64),
            dtype=dtype,
        ).item()
        pool = torch.arange(1, freq_grid_size, dtype=torch.int64, device=torch.device("cpu"))
        k_min = int(pool[0].item()) if pool.numel() > 0 else 0
        k_max = int(pool[-1].item()) if pool.numel() > 0 else -1
        pool_count = int(pool.numel())

    pool_np = pool.detach().cpu().numpy().astype(int)
    if pool_np.size == 0:
        raise RuntimeError("paper band pool is empty; adjust fs/M/band settings")

    k_from_min = hz_to_k(sr=fs, freq_grid_size=freq_grid_size, hz=fmin_dbg)
    k_from_max = hz_to_k(sr=fs, freq_grid_size=freq_grid_size, hz=fmax_dbg)
    print("[paper-band debug]")
    print(f"  sr={int(fs)} freq_grid_size={int(freq_grid_size)}")
    print(f"  paper_band_min_hz={float(fmin_dbg):.3f} paper_band_max_hz={float(fmax_dbg):.3f}")
    print(
        f"  derived_k_min={k_from_min} derived_k_max={k_from_max} "
        f"pool_k_min={k_min} pool_k_max={k_max} pool_count={pool_count}"
    )

    dbg_rng = np.random.default_rng(1234)
    sample_n = min(10, pool_np.size)
    sample_k = dbg_rng.choice(pool_np, size=sample_n, replace=False)
    sample_k = np.sort(sample_k)
    sample_hz = k_to_hz(
        sr=fs,
        freq_grid_size=freq_grid_size,
        k=torch.as_tensor(sample_k, dtype=torch.int64),
        dtype=dtype,
    ).detach().cpu().tolist()
    print("  sample k->hz:")
    for k_i, hz_i in zip(sample_k.tolist(), sample_hz):
        print(f"    k={int(k_i):6d} -> {float(hz_i):10.6f} Hz")

    replace = int(batch_size) > int(pool_np.size)
    if replace:
        print(
            f"[WARN] batch={int(batch_size)} exceeds pool_count={int(pool_np.size)}; "
            "debug minibatch sampling uses replacement"
        )
    mini = dbg_rng.choice(pool_np, size=max(int(batch_size), 1), replace=replace)
    mini_hz = k_to_hz(
        sr=fs,
        freq_grid_size=freq_grid_size,
        k=torch.as_tensor(mini, dtype=torch.int64),
        dtype=dtype,
    ).detach().cpu()
    print(
        "  sampled minibatch hz stats: min={:.6f} max={:.6f} mean={:.6f}".format(
            float(torch.min(mini_hz).item()),
            float(torch.max(mini_hz).item()),
            float(torch.mean(mini_hz).item()),
        )
    )
    return {
        "band_k_pool_count": int(pool_count),
        "band_k_min": int(k_min),
        "band_k_max": int(k_max),
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
    parser.add_argument(
        "--stability-gamma",
        type=float,
        default=0.9999,
        help="Training-only stability radius (gain_per_sample) used for lossless-core transfer evaluation.",
    )
    parser.add_argument(
        "--alpha-sparsity",
        type=float,
        default=0.05,
        help="Weight for Eq.18 sparseness penalty (higher penalizes sparse matrices, encourages denser mixing).",
    )
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
        help="Default paper mode: train with unit loop gain (lossless core objective).",
    )
    parser.add_argument(
        "--optimize-with-decay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional ablation: optimize using decay gains derived from RT60/gamma.",
    )
    parser.add_argument("--learn-io", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--paper-band-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable paper-band training by sampling minibatches only from [min,max] Hz.",
    )
    parser.add_argument("--paper-band-min-hz", type=float, default=50.0)
    parser.add_argument("--paper-band-max-hz", type=float, default=12000.0)
    parser.add_argument("--paper-band-debug", action="store_true")
    parser.add_argument("--paper-band-selfcheck", action="store_true")
    parser.add_argument("--debug-eq18-sanity", action="store_true")
    parser.add_argument("--debug-k-map", action="store_true")
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
    stability_gamma = min(max(float(args.stability_gamma), 1e-6), 0.999999)
    gamma_source = "explicit_gamma" if args.gamma is not None else "rt60_target"
    train_lossless = bool(args.train_lossless) or (not bool(args.optimize_with_decay))
    optimize_with_decay = not train_lossless
    training_mode = "lossless-core-stabilized" if train_lossless else "with-decay"
    gamma_train = stability_gamma if train_lossless else gamma_used
    paper_band_max = min(max(float(args.paper_band_max_hz), 0.0), 0.49 * fs)
    paper_band_min = max(0.0, min(float(args.paper_band_min_hz), paper_band_max))
    freq_grid_size = max(int(M), 2)

    k_probe = torch.tensor([0, freq_grid_size - 1], dtype=dtype)
    hz_probe = k_to_hz(sr=fs, freq_grid_size=freq_grid_size, k=k_probe, dtype=dtype)
    if abs(float(hz_probe[0].detach().cpu())) > 1e-9:
        raise RuntimeError("k_to_hz mapping check failed: k=0 must map to 0 Hz")
    if args.debug_k_map:
        print(
            f"[k-map] k=0 -> {float(hz_probe[0]):.6f} Hz, "
            f"k={freq_grid_size-1} -> {float(hz_probe[1]):.6f} Hz "
            f"(Nyquist≈{0.5*fs:.6f} Hz)"
        )

    if args.debug_eq18_sanity:
        _print_eq18_sanity(dtype)

    if bool(args.paper_band_enable):
        band_pool, band_k_min, band_k_max, band_pool_count = build_band_k_pool(
            sr=fs,
            freq_grid_size=freq_grid_size,
            fmin_hz=paper_band_min,
            fmax_hz=paper_band_max,
            exclude_dc=True,
            device=torch.device("cpu"),
        )
    else:
        band_pool = torch.arange(1, freq_grid_size, dtype=torch.int64, device=torch.device("cpu"))
        band_pool_count = int(band_pool.numel())
        band_k_min = int(band_pool[0].item()) if band_pool_count > 0 else 0
        band_k_max = int(band_pool[-1].item()) if band_pool_count > 0 else -1
    if int(band_pool.numel()) == 0:
        raise RuntimeError("paper-band pool is empty; adjust fs/M/band args")

    if args.paper_band_selfcheck:
        dbg_meta = _print_paper_band_debug(
            fs=fs,
            freq_grid_size=freq_grid_size,
            paper_band_enable=bool(args.paper_band_enable),
            paper_band_min_hz=paper_band_min,
            paper_band_max_hz=paper_band_max,
            batch_size=batch,
            dtype=dtype,
        )
        band_pool_count = int(dbg_meta["band_k_pool_count"])
        band_k_min = int(dbg_meta["band_k_min"])
        band_k_max = int(dbg_meta["band_k_max"])
    if args.paper_band_selfcheck:
        print("[paper-band selfcheck] OK")
        return

    print("[Repro command]")
    print("  " + " ".join(shlex.quote(x) for x in sys.argv))
    print(
        "[TRAINING CONFIG]\n"
        f"  fs={fs:.1f} rt60_target={float(args.rt60):.4f}s gamma_used={gamma_used:.9f}\n"
        f"  stability_gamma={stability_gamma:.9f} gamma_train={gamma_train:.9f} "
        f"training_mode={training_mode} train_lossless={str(train_lossless).lower()}\n"
        f"  M={M} freq_grid_size={freq_grid_size} batch={batch} epochs={epochs} total_steps={total_steps} lr={args.lr}\n"
        f"  seed={args.seed} spectral_mode={args.spectral_mode} alpha_density={alpha_sparsity:.6f} learn_io={str(bool(args.learn_io)).lower()}\n"
        f"  paper_band_enable={str(bool(args.paper_band_enable)).lower()} paper_band_min_hz={paper_band_min:.2f} paper_band_max_hz={paper_band_max:.2f} "
        f"band_k_pool_count={band_pool_count}"
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
        stability_gamma=stability_gamma,
        train_lossless=train_lossless,
        alpha_density=alpha_sparsity,
        learn_io=args.learn_io,
        freq_bins_per_step=batch,
        paper_band_enable=bool(args.paper_band_enable),
        paper_band_min_hz=paper_band_min,
        paper_band_max_hz=paper_band_max,
        paper_band_debug=bool(args.paper_band_debug),
        val_split=0.2,
        spectral_mode=args.spectral_mode,
        seed=args.seed,
        dtype=dtype,
        device="cpu",
        log_every=max(total_steps // 24, 1),
    )
    print(
        "[Band pool] training_band_k_count={} train_band_k_count={} val_band_k_count={}".format(
            int(round(float(result.losses.get("training_band_k_count", 0.0)))),
            int(round(float(result.losses.get("train_band_k_count", 0.0)))),
            int(round(float(result.losses.get("val_band_k_count", 0.0)))),
        )
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
        "optimize_with_decay": bool(optimize_with_decay),
        "stability_gamma": float(stability_gamma),
        "gamma_train": float(result.gamma_train),
        "training_mode": str(result.training_mode),
        "alpha_density": float(alpha_sparsity),
        "alpha_sparsity": float(alpha_sparsity),
        "spectral_mode": args.spectral_mode,
        "freq_grid_size": int(freq_grid_size),
        "freq_bins_per_step": int(batch),
        "learn_io": bool(args.learn_io),
        "paper_band_enable": bool(args.paper_band_enable),
        "paper_band_min_hz": float(paper_band_min),
        "paper_band_max_hz": float(paper_band_max),
        "band_k_pool_count": int(band_pool_count),
        "band_k_min": int(band_k_min),
        "band_k_max": int(band_k_max),
        "k_to_hz_formula": "hz = k * sr / (2 * freq_grid_size)",
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
        "best_val_step": int(round(float(result.losses.get("best_val_step", -1.0)))),
        "best_val_spectral_loss_like_50_12k": float(
            result.losses.get("best_val_spectral_loss_like_50_12k", float("nan"))
        ),
        "best_val_spectral_dev_db_50_12k": float(
            result.losses.get("best_val_spectral_dev_db_50_12k", float("nan"))
        ),
        "training_band_k_count": int(round(float(result.losses.get("training_band_k_count", 0.0)))),
        "band_k_min_loss": int(round(float(result.losses.get("band_k_min", float(band_k_min))))),
        "band_k_max_loss": int(round(float(result.losses.get("band_k_max", float(band_k_max))))),
        "train_band_k_count": int(round(float(result.losses.get("train_band_k_count", 0.0)))),
        "val_band_k_count": int(round(float(result.losses.get("val_band_k_count", 0.0)))),
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
            stability_gamma=stability_gamma,
            M=M,
            dtype=dtype,
            alpha_sparsity=alpha_sparsity,
            spectral_mode=args.spectral_mode,
            train_lossless=train_lossless,
            paper_band_enable=bool(args.paper_band_enable),
            paper_band_min_hz=paper_band_min,
            paper_band_max_hz=paper_band_max,
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
                stability_gamma=stability_gamma,
                train_lossless=train_lossless,
                alpha_density=alpha_sparsity,
                learn_io=False,
                freq_bins_per_step=batch,
                paper_band_enable=bool(args.paper_band_enable),
                paper_band_min_hz=paper_band_min,
                paper_band_max_hz=paper_band_max,
                paper_band_debug=False,
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
                stability_gamma=stability_gamma,
                train_lossless=train_lossless,
                alpha_density=alpha_sparsity,
                learn_io=True,
                freq_bins_per_step=batch,
                paper_band_enable=bool(args.paper_band_enable),
                paper_band_min_hz=paper_band_min,
                paper_band_max_hz=paper_band_max,
                paper_band_debug=False,
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
