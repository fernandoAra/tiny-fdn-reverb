#!/usr/bin/env python3
"""Batch IR analysis for tiny DiffFDN experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def _infer_matrix_type(name: str) -> str:
    lower = name.lower()
    if "house" in lower:
        return "householder"
    if "had" in lower:
        return "hadamard"
    return "unknown"


def load_ir_mono(path: Path) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim == 2:
        x = data.astype(np.float64).mean(axis=1)
    else:
        x = data.astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        full_scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        x /= float(full_scale)

    # Trim leading silence (up to first 0.5 s) using -60 dB threshold.
    thr = 10.0 ** (-60.0 / 20.0)
    scan_n = min(len(x), int(0.5 * sr))
    start = 0
    for i in range(scan_n):
        if abs(x[i]) > thr:
            start = i
            break
    if start > 0:
        x = x[start:]

    peak = max(float(np.max(np.abs(x))), 1e-12)
    x = x / peak
    return sr, x


def schroeder_edc_db(x: np.ndarray) -> np.ndarray:
    energy = np.cumsum((x[::-1] ** 2))[::-1]
    energy = np.maximum(energy, 1e-20)
    edc_db = 10.0 * np.log10(energy / energy[0])
    return edc_db


def fit_decay(
    t: np.ndarray, edc_db: np.ndarray, upper_db: float, lower_db: float
) -> Tuple[float, Optional[Tuple[float, float, np.ndarray]]]:
    mask = (edc_db <= upper_db) & (edc_db >= lower_db)
    if int(np.sum(mask)) < 8:
        return np.nan, None

    slope, intercept = np.polyfit(t[mask], edc_db[mask], 1)
    if slope >= 0.0:
        return np.nan, (slope, intercept, mask)

    rt60_equiv = -60.0 / slope
    return float(rt60_equiv), (float(slope), float(intercept), mask)


def echo_density_proxy(x: np.ndarray, sr: int) -> float:
    win = max(8, int(0.010 * sr))
    step = max(1, int(0.005 * sr))
    peaks = np.r_[
        False, (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]), False
    ] & (np.abs(x) > 0.02)

    densities = []
    for i in range(0, max(0, len(x) - win), step):
        density_ps = float(np.sum(peaks[i : i + win])) * (float(sr) / float(win))
        t_center = (i + 0.5 * win) / float(sr)
        if 0.05 <= t_center <= 0.30:
            densities.append(density_ps)
    if not densities:
        return 0.0
    return float(np.mean(densities))


def ringiness_proxy(x: np.ndarray) -> float:
    if len(x) < 512:
        return 0.0
    start = int(0.20 * len(x))
    end = int(0.80 * len(x))
    if end - start < 512:
        start = max(0, len(x) - 4096)
        end = len(x)
    tail = x[start:end]
    if len(tail) < 512:
        return 0.0

    win = np.hanning(len(tail))
    spec = np.abs(np.fft.rfft(tail * win))
    mean_mag = float(np.mean(spec)) + 1e-12
    peak_mag = float(np.max(spec))
    return float(peak_mag / mean_mag)


def load_sidecar(path: Path) -> Dict[str, object]:
    meta_path = path.with_suffix(path.suffix + ".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return {}


def analyze_one(path: Path) -> Dict[str, object]:
    sr, x = load_ir_mono(path)
    t = np.arange(len(x), dtype=np.float64) / float(sr)
    edc_db = schroeder_edc_db(x)

    rt60_t30, _ = fit_decay(t, edc_db, upper_db=-5.0, lower_db=-35.0)
    rt60_t20, _ = fit_decay(t, edc_db, upper_db=-5.0, lower_db=-25.0)
    edt, _ = fit_decay(t, edc_db, upper_db=0.0, lower_db=-10.0)

    rt60 = rt60_t30 if np.isfinite(rt60_t30) else rt60_t20
    rt60_method = "T30" if np.isfinite(rt60_t30) else ("T20" if np.isfinite(rt60_t20) else "none")

    metadata = load_sidecar(path)
    config_id = str(metadata.get("config_id", path.stem))
    matrix_type = str(metadata.get("matrix_type", _infer_matrix_type(path.stem)))

    return {
        "path": str(path),
        "name": path.stem,
        "sr": int(sr),
        "config_id": config_id,
        "matrix_type": matrix_type,
        "t": t,
        "edc_db": edc_db,
        "rt60_s": float(rt60) if np.isfinite(rt60) else np.nan,
        "rt60_method": rt60_method,
        "edt_s": float(edt) if np.isfinite(edt) else np.nan,
        "ringiness": ringiness_proxy(x),
        "echo_density_ps": echo_density_proxy(x, sr),
    }


def write_summary_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "file",
        "config_id",
        "matrix_type",
        "sr",
        "rt60_s",
        "rt60_method",
        "edt_s",
        "ringiness",
        "echo_density_ps",
    ]
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "file": r["path"],
                    "config_id": r["config_id"],
                    "matrix_type": r["matrix_type"],
                    "sr": r["sr"],
                    "rt60_s": f"{r['rt60_s']:.6f}" if np.isfinite(r["rt60_s"]) else "",
                    "rt60_method": r["rt60_method"],
                    "edt_s": f"{r['edt_s']:.6f}" if np.isfinite(r["edt_s"]) else "",
                    "ringiness": f"{r['ringiness']:.6f}",
                    "echo_density_ps": f"{r['echo_density_ps']:.6f}",
                }
            )


def make_plots(rows: List[Dict[str, object]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    color_by_matrix = {
        "householder": "tab:blue",
        "hadamard": "tab:orange",
        "unknown": "tab:gray",
    }

    # 1) EDC overlay
    plt.figure(figsize=(9, 4))
    for r in rows:
        color = color_by_matrix.get(str(r["matrix_type"]).lower(), "tab:gray")
        plt.plot(r["t"], r["edc_db"], lw=1.3, color=color, alpha=0.9, label=f"{r['config_id']} ({r['matrix_type']})")
    plt.axhline(-5.0, color="0.8", lw=0.8)
    plt.axhline(-25.0, color="0.9", lw=0.8)
    plt.axhline(-35.0, color="0.9", lw=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("EDC (dB)")
    plt.title("EDC Overlay by Matrix Type")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "edc_overlay.png", dpi=160)
    plt.close()

    labels = [str(r["config_id"]) for r in rows]
    x = np.arange(len(rows))
    rt60_vals = np.array([float(r["rt60_s"]) if np.isfinite(r["rt60_s"]) else 0.0 for r in rows], dtype=np.float64)
    edt_vals = np.array([float(r["edt_s"]) if np.isfinite(r["edt_s"]) else 0.0 for r in rows], dtype=np.float64)

    # 2) RT60/EDT bars
    plt.figure(figsize=(9, 4))
    width = 0.38
    plt.bar(x - width / 2.0, rt60_vals, width=width, label="RT60 (s)")
    plt.bar(x + width / 2.0, edt_vals, width=width, label="EDT (s)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Seconds")
    plt.title("RT60 and EDT Across Configs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "rt60_edt_bars.png", dpi=160)
    plt.close()

    # 3) Ringiness bars
    ring_vals = np.array([float(r["ringiness"]) for r in rows], dtype=np.float64)
    bar_colors = [color_by_matrix.get(str(r["matrix_type"]).lower(), "tab:gray") for r in rows]
    plt.figure(figsize=(9, 4))
    plt.bar(x, ring_vals, color=bar_colors)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Peak/Mean Spectrum Ratio")
    plt.title("Ringiness Proxy Across Configs")
    plt.tight_layout()
    plt.savefig(fig_dir / "ringiness_bars.png", dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", default="eval/out/ir", help="Folder containing WAV IR files")
    parser.add_argument("--pattern", default="*.wav", help="Glob pattern inside ir-dir")
    parser.add_argument("--fig-dir", default="eval/figs", help="Output figure folder")
    args = parser.parse_args()

    ir_dir = Path(args.ir_dir)
    wavs = sorted(ir_dir.glob(args.pattern))
    if not wavs:
        raise SystemExit(f"No WAV files found in {ir_dir} with pattern {args.pattern}")

    rows = [analyze_one(wav) for wav in wavs]

    fig_dir = Path(args.fig_dir)
    out_csv = fig_dir / "summary.csv"
    write_summary_csv(rows, out_csv)
    make_plots(rows, fig_dir)

    print(f"Wrote summary: {out_csv}")
    print(f"Wrote figure: {fig_dir / 'edc_overlay.png'}")
    print(f"Wrote figure: {fig_dir / 'rt60_edt_bars.png'}")
    print(f"Wrote figure: {fig_dir / 'ringiness_bars.png'}")


if __name__ == "__main__":
    main()
