#!/usr/bin/env python3
"""Run reproducible multi-seed Fixed-vs-Diff evaluation and aggregate results."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

MODE_ORDER = ["fixed", "u_only", "full"]
MODE_LABELS = {
    "fixed": "Fixed Householder",
    "u_only": "Diff Householder (u-only)",
    "full": "Diff Householder (u+b+c)",
}
MODE_COLORS = {
    "fixed": "tab:orange",
    "u_only": "tab:blue",
    "full": "tab:green",
}

DEFAULT_B = [0.25, 0.25, 0.25, 0.25]
DEFAULT_CL = [0.5, -0.5, 0.5, -0.5]
DEFAULT_CR = [0.5, 0.5, -0.5, -0.5]

METRICS = [
    "spectral_loss_like",
    "spectral_loss_like_50_12k",
    "spectral_dev_db",
    "spectral_dev_db_50_12k",
    "rt60_s",
    "edt_s",
    "ringiness",
    "kurtosis_mean_50_300ms",
    "echo_density_events_per_s_50_300ms",
]

METRIC_LABELS = {
    "spectral_loss_like": "spectral loss-like",
    "spectral_loss_like_50_12k": "spectral loss-like (50-12k)",
    "spectral_dev_db": "spectral deviation (dB)",
    "spectral_dev_db_50_12k": "spectral deviation 50-12k (dB)",
    "rt60_s": "RT60 (s)",
    "edt_s": "EDT (s)",
    "ringiness": "ringiness (peak/mean)",
    "kurtosis_mean_50_300ms": "kurtosis mean 50-300 ms",
    "echo_density_events_per_s_50_300ms": "echo density (events/s)",
}

WIN_DIRECTION = {
    "spectral_loss_like": "lower",
    "spectral_loss_like_50_12k": "lower",
    "spectral_dev_db": "lower",
    "spectral_dev_db_50_12k": "lower",
    "rt60_s": "target",
    "edt_s": "target",
    "ringiness": "lower",
    "kurtosis_mean_50_300ms": "lower",
    "echo_density_events_per_s_50_300ms": "higher",
}

DELTA_METRICS = [
    "spectral_dev_db_50_12k",
    "spectral_loss_like_50_12k",
    "ringiness",
    "echo_density_events_per_s_50_300ms",
]

DELTA_DIRECTION = {
    "spectral_dev_db_50_12k": "lower is better",
    "spectral_loss_like_50_12k": "lower is better",
    "ringiness": "lower is better",
    "echo_density_events_per_s_50_300ms": "higher is denser / better",
}


def _parse_seed_csv(text: str) -> List[int]:
    out = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not out:
        raise ValueError("No seeds provided")
    return out


def _run(cmd: Sequence[str], *, dry_run: bool) -> None:
    print("+", shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(list(cmd), check=True)


def _bool_flag(name: str, value: bool) -> str:
    return f"--{name}" if value else f"--no-{name}"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(v: object) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x


def _aggregate_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0.0,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": float(arr.size),
    }


def _extract_metric_from_preset(payload: Dict[str, object], key: str) -> float:
    if key in payload:
        return _safe_float(payload.get(key))
    losses = payload.get("losses")
    if isinstance(losses, dict) and key in losses:
        return _safe_float(losses.get(key))
    return float("nan")


def _max_abs_diff(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray([float(v) for v in a], dtype=np.float64)
    bb = np.asarray([float(v) for v in b], dtype=np.float64)
    return float(np.max(np.abs(aa - bb)))


def _preset_has_learned_io(payload: Dict[str, object], eps: float = 1e-6) -> bool:
    b = payload.get("b")
    c_l = payload.get("cL")
    c_r = payload.get("cR")
    if not (isinstance(b, list) and isinstance(c_l, list) and isinstance(c_r, list)):
        return False
    if not (len(b) == 4 and len(c_l) == 4 and len(c_r) == 4):
        return False
    return bool(
        max(
            _max_abs_diff(b, DEFAULT_B),
            _max_abs_diff(c_l, DEFAULT_CL),
            _max_abs_diff(c_r, DEFAULT_CR),
        )
        > float(eps)
    )


def _aggregate_curves(
    series: Sequence[Tuple[np.ndarray, np.ndarray]],
    *,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid_series: List[Tuple[np.ndarray, np.ndarray]] = []
    for t, y in series:
        if t.size < 2 or y.size < 2:
            continue
        mask = np.isfinite(t) & np.isfinite(y)
        if np.count_nonzero(mask) < 2:
            continue
        t_v = t[mask]
        y_v = y[mask]
        order = np.argsort(t_v)
        valid_series.append((t_v[order], y_v[order]))

    if not valid_series:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    t_max = max(float(np.max(t)) for t, _ in valid_series)
    t_grid = np.arange(0.0, t_max + dt, dt, dtype=np.float64)
    ys: List[np.ndarray] = []
    for t, y in valid_series:
        yi = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
        ys.append(yi)
    y_mat = np.stack(ys, axis=0)
    valid = np.isfinite(y_mat)
    count = np.sum(valid, axis=0)
    sum_y = np.nansum(y_mat, axis=0)
    mean_y = np.full_like(sum_y, np.nan, dtype=np.float64)
    nz = count > 0
    mean_y[nz] = sum_y[nz] / count[nz]

    diff = np.where(valid, y_mat - mean_y[None, :], 0.0)
    var = np.full_like(mean_y, np.nan, dtype=np.float64)
    var[nz] = np.sum(diff[:, nz] * diff[:, nz], axis=0) / count[nz]
    std_y = np.sqrt(var)
    return t_grid, mean_y, std_y


def _plot_metrics_errorbars(
    out_path: Path,
    stats_by_mode: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_list = list(METRICS)
    ncols = 4
    nrows = int(np.ceil(float(len(metric_list)) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows))
    fig.suptitle("multiseed fixed vs diff metrics (mean ± std)", fontsize=15)
    x = np.arange(len(MODE_ORDER))
    labels = [MODE_LABELS[m] for m in MODE_ORDER]
    colors = [MODE_COLORS[m] for m in MODE_ORDER]

    for i, metric in enumerate(metric_list):
        ax = axes.flat[i]
        means = [stats_by_mode.get(m, {}).get(metric, {}).get("mean", np.nan) for m in MODE_ORDER]
        stds = [stats_by_mode.get(m, {}).get(metric, {}).get("std", np.nan) for m in MODE_ORDER]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.9)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.set_xticks(x, labels, rotation=12, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=9)

    flat_axes = np.atleast_1d(axes).reshape(-1)
    for j in range(len(metric_list), len(flat_axes)):
        flat_axes[j].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_deltas_errorbars(
    out_path: Path,
    *,
    delta_rows: Sequence[Dict[str, object]],
    delta_stats: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))
    fig.suptitle("multiseed improvement vs fixed (per-seed points + mean ± std)", fontsize=15)
    scen_order = ["u_only", "full"]
    scen_labels = [MODE_LABELS[m] for m in scen_order]
    scen_colors = [MODE_COLORS[m] for m in scen_order]
    x = np.arange(len(scen_order))

    for i, metric in enumerate(DELTA_METRICS):
        ax = axes.flat[i]
        ax.axhline(0.0, color="0.45", lw=1.0, ls="--", zorder=1)
        means = [delta_stats.get(m, {}).get(metric, {}).get("mean", np.nan) for m in scen_order]
        stds = [delta_stats.get(m, {}).get(metric, {}).get("std", np.nan) for m in scen_order]
        for j, scenario in enumerate(scen_order):
            seed_rows = [
                row
                for row in delta_rows
                if str(row.get("scenario", "")).strip() == scenario and str(row.get("metric", "")).strip() == metric
            ]
            seed_rows = sorted(seed_rows, key=lambda row: int(_safe_float(row.get("seed", 0))))
            vals = np.asarray([_safe_float(row.get("delta_vs_fixed", float("nan"))) for row in seed_rows], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size > 0:
                jitter = (
                    np.linspace(-0.12, 0.12, vals.size, dtype=np.float64)
                    if vals.size > 1
                    else np.zeros(1, dtype=np.float64)
                )
                ax.scatter(
                    np.full(vals.size, x[j], dtype=np.float64) + jitter,
                    vals,
                    s=36,
                    color=scen_colors[j],
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.7,
                    zorder=3,
                )
            mean_v = means[j]
            std_v = stds[j]
            if np.isfinite(mean_v):
                ax.errorbar(
                    x[j],
                    mean_v,
                    yerr=std_v if np.isfinite(std_v) else None,
                    fmt="D",
                    ms=7,
                    mfc="white",
                    mec=scen_colors[j],
                    mew=1.5,
                    ecolor=scen_colors[j],
                    elinewidth=1.5,
                    capsize=5,
                    capthick=1.5,
                    zorder=4,
                )
        subtitle = DELTA_DIRECTION.get(metric, "")
        title = METRIC_LABELS.get(metric, metric)
        if subtitle:
            title = f"{title}\n({subtitle})"
        ax.set_title(title, fontsize=10)
        ax.set_xticks(x, scen_labels, rotation=10, ha="right")
        ax.set_ylabel("improvement vs fixed\n(+ better)", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=9)

    fig.text(
        0.5,
        0.01,
        "points = individual seeds, diamonds = mean ± std",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_curve_mean_std(
    out_path: Path,
    *,
    title: str,
    ylabel: str,
    curves_by_mode: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    summary_tmin: float,
    summary_tmax: float,
    x_max: float | None = None,
    y_min: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title, fontsize=14)
    ax.axvspan(summary_tmin, summary_tmax, color="0.92", alpha=0.8, label="summary window")

    for mode in MODE_ORDER:
        if mode not in curves_by_mode or not curves_by_mode[mode]:
            continue
        t, mean_y, std_y = _aggregate_curves(curves_by_mode[mode], dt=0.005)
        ax.plot(t, mean_y, lw=1.8, color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        ax.fill_between(t, mean_y - std_y, mean_y + std_y, color=MODE_COLORS[mode], alpha=0.18)

    ax.set_xlabel("time (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10)
    if x_max is not None and x_max > 0.0:
        ax.set_xlim(0.0, float(x_max))
    if y_min is not None:
        ax.set_ylim(bottom=float(y_min))
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _append_curves_from_run_payload(
    payload: Dict[str, object],
    *,
    curves_kurt: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    curves_echo: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
) -> None:
    modes = payload.get("modes", {})
    for mode in MODE_ORDER:
        mode_payload = modes.get(mode)
        if not isinstance(mode_payload, dict):
            continue
        curves = mode_payload.get("curves", {})
        if not isinstance(curves, dict):
            continue
        kurt = curves.get("kurtosis", {})
        echo = curves.get("echo_density", {})
        t_k = np.asarray(kurt.get("t", []), dtype=np.float64)
        y_k = np.asarray(kurt.get("y", []), dtype=np.float64)
        t_e = np.asarray(echo.get("t", []), dtype=np.float64)
        y_e = np.asarray(echo.get("y", []), dtype=np.float64)
        if t_k.size > 1 and y_k.size > 1:
            curves_kurt[mode].append((t_k, y_k))
        if t_e.size > 1 and y_e.size > 1:
            curves_echo[mode].append((t_e, y_e))


def _write_multiseed_paper_figures(
    *,
    paper_dir: Path,
    config_id: str,
    stats_by_mode: Dict[str, Dict[str, Dict[str, float]]],
    delta_rows: Sequence[Dict[str, object]],
    delta_stats: Dict[str, Dict[str, Dict[str, float]]],
    curves_kurt: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    curves_echo: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    summary_tmin: float,
    summary_tmax: float,
) -> None:
    paper_dir.mkdir(parents=True, exist_ok=True)
    metrics_fig = paper_dir / "multiseed_metrics_errorbars.png"
    deltas_fig = paper_dir / "multiseed_deltas_errorbars.png"
    diffusion_fig = paper_dir / "multiseed_diffusion_meanstd.png"
    echo_fig = paper_dir / "multiseed_echo_density_meanstd.png"

    _plot_metrics_errorbars(metrics_fig, stats_by_mode)
    _plot_deltas_errorbars(deltas_fig, delta_rows=delta_rows, delta_stats=delta_stats)
    _plot_curve_mean_std(
        diffusion_fig,
        title=f"multiseed impulsiveness proxy (kurtosis, exploratory) ({config_id})",
        ylabel="excess kurtosis (lower = less impulsive)",
        curves_by_mode=curves_kurt,
        summary_tmin=summary_tmin,
        summary_tmax=summary_tmax,
    )
    _plot_curve_mean_std(
        echo_fig,
        title=f"multiseed echo-density proxy ({config_id})",
        ylabel="events/s",
        curves_by_mode=curves_echo,
        summary_tmin=summary_tmin,
        summary_tmax=summary_tmax,
        x_max=0.5,
        y_min=0.0,
    )

    print("Wrote multiseed paper figures:")
    print(f"  - {metrics_fig}")
    print(f"  - {deltas_fig}")
    print(f"  - {diffusion_fig}")
    print(f"  - {echo_fig}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--restart-seed-mult", type=int, default=1000)
    parser.add_argument(
        "--select-metric",
        default="best_val_spectral_dev_db_50_12k",
        help="Preset metric key used to select best restart.",
    )
    parser.add_argument("--select-direction", choices=["lower", "higher"], default="lower")
    parser.add_argument("--dry-run", action="store_true")

    # optimizer pass-through
    parser.add_argument("--matrix-type", default="householder", choices=["householder", "hadamard"])
    parser.add_argument("--fs", type=float, default=48000.0)
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--M", type=int, default=480000)
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--delay-samples", default="1499,2377,3217,4421")
    parser.add_argument("--rt60", type=float, default=2.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha-sparsity", type=float, default=0.05)
    parser.add_argument("--spectral-mode", choices=["unity", "mean"], default="unity")
    parser.add_argument("--train-lossless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimize-with-decay", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--paper-band-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--paper-band-min-hz", type=float, default=50.0)
    parser.add_argument("--paper-band-max-hz", type=float, default=12000.0)
    parser.add_argument(
        "--learn-io",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable learned b/c optimization (paper experiment default).",
    )
    parser.add_argument(
        "--force-learn-io",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="If set, overrides --learn-io. Default behavior: enabled when scope includes full/all.",
    )

    # compare pass-through
    parser.add_argument("--scope", choices=["fixed", "u_only", "full", "all"], default="all")
    parser.add_argument("--channel", choices=["L", "R"], default="L")
    parser.add_argument("--smooth-bins", type=int, default=25)
    parser.add_argument("--max-freq-hz", type=float, default=12000.0)
    parser.add_argument("--kurtosis-rms-threshold-db", type=float, default=-60.0)
    parser.add_argument("--echo-density-threshold-db", type=float, default=-30.0)
    parser.add_argument("--echo-density-mad-k", type=float, default=3.0)
    parser.add_argument("--echo-density-min-spacing-ms", type=float, default=1.0)
    parser.add_argument("--echo-density-window-ms", type=float, default=10.0)
    parser.add_argument("--echo-density-hop-ms", type=float, default=5.0)
    parser.add_argument("--echo-density-tmin", type=float, default=0.05)
    parser.add_argument("--echo-density-tmax", type=float, default=0.30)
    parser.add_argument("--trim-leading-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trim-threshold-db", type=float, default=-60.0)
    parser.add_argument("--trim-max-seconds", type=float, default=0.5)
    parser.add_argument("--sanity-check", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--preset-out-root",
        default="eval/out/presets",
        help="Directory for seed-specific preset JSON outputs",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Root output dir (default: eval/figs/multiseed/<config-id>)",
    )
    parser.add_argument(
        "--reuse-existing-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip optimizer/compare and redraw paper figures from existing multiseed outputs under out-root.",
    )
    # Backwards-compat aliases.
    parser.add_argument("--preset-dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fig-root", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_seed_csv(args.seeds)

    repo_root = Path(__file__).resolve().parents[2]
    preset_root = args.preset_out_root if args.preset_out_root is not None else args.preset_dir
    if preset_root is None:
        preset_root = "eval/out/presets"
    preset_dir = (repo_root / preset_root).resolve()
    if args.out_root:
        run_root = (repo_root / args.out_root).resolve()
    else:
        fig_base = args.fig_root if args.fig_root is not None else "eval/figs/multiseed"
        run_root = (repo_root / fig_base / args.config_id).resolve()
    paper_dir = run_root / "paper"
    run_root.mkdir(parents=True, exist_ok=True)
    preset_dir.mkdir(parents=True, exist_ok=True)

    per_seed_rows: List[Dict[str, str]] = []
    curves_kurt: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {m: [] for m in MODE_ORDER}
    curves_echo: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {m: [] for m in MODE_ORDER}

    python = "python3"
    print("[Repro command]")
    print("  " + " ".join(shlex.quote(x) for x in sys.argv))
    force_learn_io = args.force_learn_io
    if force_learn_io is None:
        force_learn_io = args.scope in {"all", "full"}
    effective_learn_io = bool(args.learn_io or force_learn_io)
    if force_learn_io and not bool(args.learn_io):
        print("[Info] force-learn-io enabled by scope; overriding --no-learn-io.")

    if bool(args.reuse_existing_data):
        aggregate_stats = run_root / "aggregate_stats.json"
        aggregate_deltas = run_root / "aggregate_deltas.csv"
        if not aggregate_stats.is_file():
            raise RuntimeError(f"Missing existing aggregate stats: {aggregate_stats}")
        if not aggregate_deltas.is_file():
            raise RuntimeError(f"Missing existing aggregate deltas: {aggregate_deltas}")

        aggregate_payload = json.loads(aggregate_stats.read_text())
        stats_by_mode = aggregate_payload.get("metrics", {})
        delta_stats = aggregate_payload.get("delta_vs_fixed", {})
        delta_rows = _read_csv_rows(aggregate_deltas)
        for seed in seeds:
            run_json = run_root / f"seed{seed}" / "fixed_vs_diff_run.json"
            if not run_json.is_file():
                raise RuntimeError(f"Missing per-seed run payload for reuse mode: {run_json}")
            _append_curves_from_run_payload(
                json.loads(run_json.read_text()),
                curves_kurt=curves_kurt,
                curves_echo=curves_echo,
            )

        _write_multiseed_paper_figures(
            paper_dir=paper_dir,
            config_id=args.config_id,
            stats_by_mode=stats_by_mode,
            delta_rows=delta_rows,
            delta_stats=delta_stats,
            curves_kurt=curves_kurt,
            curves_echo=curves_echo,
            summary_tmin=float(args.echo_density_tmin),
            summary_tmax=float(args.echo_density_tmax),
        )
        print(f"Reused aggregate stats:  {aggregate_stats}")
        print(f"Reused aggregate deltas: {aggregate_deltas}")
        return

    selected_preset_by_seed: Dict[int, Path] = {}

    for seed in seeds:
        seed_dir = run_root / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        restart_rows: List[Dict[str, object]] = []
        restart_candidates: List[Dict[str, object]] = []
        restart_count = max(int(args.restarts), 1)
        for restart_idx in range(restart_count):
            restart_seed = int(seed) * int(args.restart_seed_mult) + int(restart_idx)
            preset_path = preset_dir / f"{args.config_id}_seed{seed}_r{restart_idx}.json"
            history_path = seed_dir / f"optimizer_history_r{restart_idx}.json"
            optimize_cmd = [
                python,
                str((repo_root / "eval/difffdn/optimize_householder.py").resolve()),
                "--config-id",
                f"{args.config_id}_seed{seed}_r{restart_idx}",
                "--matrix-type",
                args.matrix_type,
                "--fs",
                str(args.fs),
                "--nfft",
                str(args.nfft),
                "--M",
                str(args.M),
                "--batch",
                str(args.batch),
                "--epochs",
                str(args.epochs),
                "--delay-samples",
                args.delay_samples,
                "--rt60",
                str(args.rt60),
                "--lr",
                str(args.lr),
                "--alpha-sparsity",
                str(args.alpha_sparsity),
                "--spectral-mode",
                args.spectral_mode,
                _bool_flag("train-lossless", bool(args.train_lossless)),
                _bool_flag("optimize-with-decay", bool(args.optimize_with_decay)),
                _bool_flag("paper-band-enable", bool(args.paper_band_enable)),
                "--paper-band-min-hz",
                str(args.paper_band_min_hz),
                "--paper-band-max-hz",
                str(args.paper_band_max_hz),
                _bool_flag("learn-io", bool(effective_learn_io)),
                "--seed",
                str(restart_seed),
                "--out-json",
                str(preset_path),
                "--history-json",
                str(history_path),
                "--out-dir",
                str(preset_dir),
            ]
            _run(optimize_cmd, dry_run=args.dry_run)

            if args.dry_run:
                continue

            preset_payload = json.loads(preset_path.read_text())
            metric_value = _extract_metric_from_preset(preset_payload, str(args.select_metric))
            has_learned_io = _preset_has_learned_io(preset_payload)
            restart_row = {
                "restart_idx": restart_idx,
                "restart_seed": restart_seed,
                "selection_metric": str(args.select_metric),
                "selection_value": metric_value,
                "best_val_spectral_loss_like_50_12k": _extract_metric_from_preset(
                    preset_payload, "best_val_spectral_loss_like_50_12k"
                ),
                "best_val_spectral_dev_db_50_12k": _extract_metric_from_preset(
                    preset_payload, "best_val_spectral_dev_db_50_12k"
                ),
                "has_learned_io": bool(has_learned_io),
                "preset_path": str(preset_path),
            }
            restart_rows.append(restart_row)
            restart_candidates.append(
                {
                    "path": preset_path,
                    "metric": metric_value,
                    "has_learned_io": bool(has_learned_io),
                    "restart_seed": restart_seed,
                    "restart_idx": restart_idx,
                }
            )

        if args.dry_run:
            continue

        restarts_csv = seed_dir / "restarts_summary.csv"
        with restarts_csv.open("w", newline="") as handle:
            fields = [
                "restart_idx",
                "restart_seed",
                "selection_metric",
                "selection_value",
                "best_val_spectral_loss_like_50_12k",
                "best_val_spectral_dev_db_50_12k",
                "has_learned_io",
                "preset_path",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in restart_rows:
                writer.writerow(row)

        finite_candidates = [c for c in restart_candidates if np.isfinite(_safe_float(c.get("metric", float("nan"))))]
        if not finite_candidates:
            finite_candidates = restart_candidates
        if not finite_candidates:
            raise RuntimeError(f"No restart presets generated for seed {seed}")

        reverse = str(args.select_direction).lower() == "higher"
        best_candidate = sorted(
            finite_candidates,
            key=lambda c: _safe_float(c.get("metric", float("nan"))),
            reverse=reverse,
        )[0]

        if args.scope in {"full", "all"} and not bool(best_candidate.get("has_learned_io", False)):
            learned_candidates = [c for c in finite_candidates if bool(c.get("has_learned_io", False))]
            if learned_candidates:
                best_candidate = sorted(
                    learned_candidates,
                    key=lambda c: _safe_float(c.get("metric", float("nan"))),
                    reverse=reverse,
                )[0]
                print(
                    f"[WARN] seed {seed}: selected restart lacked learned IO; "
                    "falling back to best restart with learned IO."
                )
            else:
                print(
                    f"[WARN] seed {seed}: scope includes full but no restart has learned IO. "
                    "Full vs u_only may become redundant."
                )

        best_preset_path = seed_dir / "best_preset.json"
        best_payload = json.loads(Path(best_candidate["path"]).read_text())
        best_preset_path.write_text(json.dumps(best_payload, indent=2) + "\n")
        selected_preset_by_seed[int(seed)] = best_preset_path

        compare_cmd = [
            python,
            str((repo_root / "eval/scripts/compare_fixed_vs_diff.py").resolve()),
            "--preset",
            str(best_preset_path),
            "--scope",
            args.scope,
            "--seed",
            str(seed),
            "--channel",
            args.channel,
            "--smooth-bins",
            str(args.smooth_bins),
            "--max-freq-hz",
            str(args.max_freq_hz),
            "--kurtosis-rms-threshold-db",
            str(args.kurtosis_rms_threshold_db),
            "--echo-density-threshold-db",
            str(args.echo_density_threshold_db),
            "--echo-density-mad-k",
            str(args.echo_density_mad_k),
            "--echo-density-min-spacing-ms",
            str(args.echo_density_min_spacing_ms),
            "--echo-density-window-ms",
            str(args.echo_density_window_ms),
            "--echo-density-hop-ms",
            str(args.echo_density_hop_ms),
            "--echo-density-tmin",
            str(args.echo_density_tmin),
            "--echo-density-tmax",
            str(args.echo_density_tmax),
            _bool_flag("trim-leading-silence", bool(args.trim_leading_silence)),
            "--trim-threshold-db",
            str(args.trim_threshold_db),
            "--trim-max-seconds",
            str(args.trim_max_seconds),
            _bool_flag("sanity-check", bool(args.sanity_check)),
            "--fig-dir",
            str(seed_dir),
            "--summary-csv",
            str(seed_dir / "summary_fixed_vs_diff.csv"),
            "--out-ir-dir",
            str(seed_dir / "ir"),
            "--render-preset-dir",
            str(seed_dir / "render_presets"),
        ]
        _run(compare_cmd, dry_run=args.dry_run)

        if args.dry_run:
            continue

        summary_table = seed_dir / "fixed_vs_diff_summary_table.csv"
        if not summary_table.is_file():
            raise RuntimeError(
                f"Missing per-seed summary table: {summary_table}. "
                "compare_fixed_vs_diff.py may have failed before writing outputs."
            )
        rows = _read_csv_rows(summary_table)
        if not rows:
            raise RuntimeError(
                f"Per-seed summary table has no data rows: {summary_table}. "
                "Refusing to write an empty aggregate_summary.csv."
            )
        for row in rows:
            row["seed"] = str(seed)
            per_seed_rows.append(row)

        run_json = seed_dir / "fixed_vs_diff_run.json"
        if not run_json.is_file():
            raise RuntimeError(
                f"Missing per-seed run payload: {run_json}. "
                "compare_fixed_vs_diff.py did not produce fixed_vs_diff_run.json."
            )
        payload = json.loads(run_json.read_text())
        _append_curves_from_run_payload(payload, curves_kurt=curves_kurt, curves_echo=curves_echo)

    if args.dry_run:
        print("Dry run complete.")
        return

    by_mode_metric: Dict[str, Dict[str, List[float]]] = {m: {k: [] for k in METRICS} for m in MODE_ORDER}
    for row in per_seed_rows:
        mode_key = str(row.get("scope", "")).strip()
        if mode_key not in by_mode_metric:
            continue
        for metric in METRICS:
            by_mode_metric[mode_key][metric].append(_safe_float(row.get(metric, "")))

    stats_by_mode: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in MODE_ORDER}
    for mode in MODE_ORDER:
        for metric in METRICS:
            stats_by_mode[mode][metric] = _aggregate_metric(by_mode_metric[mode][metric])

    if not per_seed_rows:
        raise RuntimeError(
            "No per-seed rows were collected. "
            "aggregate_summary.csv was not written to avoid a header-only empty report."
        )

    aggregate_summary = run_root / "aggregate_summary.csv"
    aggregate_stats = run_root / "aggregate_stats.json"
    aggregate_deltas = run_root / "aggregate_deltas.csv"
    wins_table_csv = run_root / "wins_table.csv"
    ordered_fields = ["seed"] + [k for k in per_seed_rows[0].keys() if k != "seed"]
    with aggregate_summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for row in per_seed_rows:
            writer.writerow({k: row.get(k, "") for k in ordered_fields})

    # Paired deltas versus fixed baseline per seed.
    by_seed_scope: Dict[int, Dict[str, Dict[str, str]]] = {}
    for row in per_seed_rows:
        try:
            seed_i = int(row.get("seed", "0"))
        except Exception:
            continue
        scope = str(row.get("scope", "")).strip()
        if scope not in MODE_ORDER:
            continue
        by_seed_scope.setdefault(seed_i, {})[scope] = row

    delta_rows: List[Dict[str, object]] = []
    delta_values: Dict[str, Dict[str, List[float]]] = {"u_only": {}, "full": {}}
    for scenario in ("u_only", "full"):
        delta_values[scenario] = {m: [] for m in DELTA_METRICS}
    wins_counts: Dict[str, Dict[str, int]] = {"u_only": {}, "full": {}}
    for scenario in ("u_only", "full"):
        wins_counts[scenario] = {m: 0 for m in METRICS}

    for seed_i in sorted(by_seed_scope.keys()):
        per_scope = by_seed_scope[seed_i]
        fixed_row = per_scope.get("fixed")
        if fixed_row is None:
            continue
        for scenario in ("u_only", "full"):
            scenario_row = per_scope.get(scenario)
            if scenario_row is None:
                continue
            for metric in METRICS:
                fixed_v = _safe_float(fixed_row.get(metric, ""))
                scen_v = _safe_float(scenario_row.get(metric, ""))
                if not (np.isfinite(fixed_v) and np.isfinite(scen_v)):
                    continue
                direction = WIN_DIRECTION.get(metric, "target")
                if direction.startswith("lower"):
                    delta_v = float(fixed_v - scen_v)
                    if scen_v < fixed_v:
                        wins_counts[scenario][metric] += 1
                elif direction.startswith("higher"):
                    delta_v = float(scen_v - fixed_v)
                    if scen_v > fixed_v:
                        wins_counts[scenario][metric] += 1
                else:
                    delta_v = float(scen_v - fixed_v)
                if metric in delta_values[scenario]:
                    delta_values[scenario][metric].append(delta_v)
                delta_rows.append(
                    {
                        "seed": seed_i,
                        "scenario": scenario,
                        "metric": metric,
                        "delta_vs_fixed": delta_v,
                        "fixed_value": fixed_v,
                        "scenario_value": scen_v,
                        "direction_note": direction,
                    }
                )

    with aggregate_deltas.open("w", newline="") as handle:
        fields = [
            "seed",
            "scenario",
            "metric",
            "delta_vs_fixed",
            "fixed_value",
            "scenario_value",
            "direction_note",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in delta_rows:
            writer.writerow(row)

    delta_stats: Dict[str, Dict[str, Dict[str, float]]] = {"u_only": {}, "full": {}}
    for scenario in ("u_only", "full"):
        for metric in DELTA_METRICS:
            delta_stats[scenario][metric] = _aggregate_metric(delta_values[scenario][metric])

    # Add wins_count_vs_fixed to per-mode aggregate stats for paper-friendly reporting.
    for scenario in ("u_only", "full"):
        for metric in METRICS:
            stats_by_mode[scenario][metric]["wins_count_vs_fixed"] = float(wins_counts[scenario][metric])
    for metric in METRICS:
        stats_by_mode["fixed"][metric]["wins_count_vs_fixed"] = float("nan")

    with wins_table_csv.open("w", newline="") as handle:
        fields = ["metric", "direction", "wins_u_only", "wins_full", "n_seeds"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric in METRICS:
            writer.writerow(
                {
                    "metric": metric,
                    "direction": WIN_DIRECTION.get(metric, "target"),
                    "wins_u_only": wins_counts["u_only"].get(metric, 0),
                    "wins_full": wins_counts["full"].get(metric, 0),
                    "n_seeds": len(by_seed_scope),
                }
            )

    aggregate_payload = {
        "config_id": args.config_id,
        "seeds": seeds,
        "n_rows_seed_mode": len(per_seed_rows),
        "metrics": stats_by_mode,
        "delta_vs_fixed": delta_stats,
        "paths": {
            "aggregate_summary_csv": str(aggregate_summary),
            "aggregate_deltas_csv": str(aggregate_deltas),
            "aggregate_stats_json": str(aggregate_stats),
            "wins_table_csv": str(wins_table_csv),
        },
    }
    aggregate_stats.write_text(json.dumps(aggregate_payload, indent=2) + "\n")

    _write_multiseed_paper_figures(
        paper_dir=paper_dir,
        config_id=args.config_id,
        stats_by_mode=stats_by_mode,
        delta_rows=delta_rows,
        delta_stats=delta_stats,
        curves_kurt=curves_kurt,
        curves_echo=curves_echo,
        summary_tmin=float(args.echo_density_tmin),
        summary_tmax=float(args.echo_density_tmax),
    )

    print("Wrote selected best presets:")
    for seed in seeds:
        print(f"  - seed{seed}: {selected_preset_by_seed.get(seed, Path('unknown'))}")
    print(f"Wrote aggregate summary: {aggregate_summary}")
    print(f"Wrote aggregate deltas:  {aggregate_deltas}")
    print(f"Wrote wins table:        {wins_table_csv}")
    print(f"Wrote aggregate stats:   {aggregate_stats}")

if __name__ == "__main__":
    main()
