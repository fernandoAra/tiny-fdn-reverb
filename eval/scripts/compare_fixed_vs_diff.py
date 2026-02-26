#!/usr/bin/env python3
"""Compare Fixed vs Diff Householder conditions with paper-oriented offline metrics.

SOURCE:
- DiffFDN colorless framing (Dal Santo et al.), pinned commit:
  https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c
- Householder parameterization reference (Dal Santo/flamo), pinned commit:
  https://github.com/gdalsanto/flamo/blob/4c8097d4feda76132691bb2a3e465ebcba11dcea/flamo/processor/dsp.py#L621-L725

This script is strictly offline and does not touch realtime plugin execution.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

EPS = 1e-12
DEFAULT_FIXED_U = [0.5, 0.5, 0.5, 0.5]
DEFAULT_B = [0.25, 0.25, 0.25, 0.25]
DEFAULT_CL = [0.5, -0.5, 0.5, -0.5]
DEFAULT_CR = [0.5, 0.5, -0.5, -0.5]
LTI_COMPARE_MOD_DEPTH = 0.0
LTI_COMPARE_DETUNE = 0.0
LTI_COMPARE_DAMP_HZ = 1.0e9
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

PLOT_TITLE_FONTSIZE = 15
PLOT_LABEL_FONTSIZE = 12
PLOT_LEGEND_FONTSIZE = 10
PLOT_LINEWIDTH_MAIN = 1.8
PLOT_LINEWIDTH_AUX = 1.1

SUMMARY_FIELDS = [
    "timestamp",
    "config_id",
    "scope",
    "mode",
    "sr",
    "delay_set",
    "rt60_target",
    "gamma_used",
    "u0",
    "u1",
    "u2",
    "u3",
    "spectral_loss_like",
    "spectral_dev_db",
    "rt60_s",
    "rt60_method",
    "edt_s",
    "ringiness",
    "kurtosis_mean_50_300ms",
    "echo_density_events_per_s_50_300ms",
    "sanity_max_db_error",
]


@dataclass
class ScenarioResult:
    key: str
    label: str
    preset: Dict[str, object]
    preset_path: Path
    wav_path: Path
    sidecar_path: Path
    sr: int
    signal: np.ndarray
    t: np.ndarray
    edc_db: np.ndarray
    edt_s: float
    rt60_s: float
    rt60_method: str
    ringiness: float
    spectral_loss_like: float
    spectral_dev_db: float
    ir_mag_db: np.ndarray
    ir_mag_db_smooth: np.ndarray
    analytic_mag_db: np.ndarray
    analytic_mag_db_smooth: np.ndarray
    kurt_t: np.ndarray
    kurtosis_curve: np.ndarray
    kurtosis_mean_50_300ms: float
    echo_density_t: np.ndarray
    echo_density_curve: np.ndarray
    echo_density_events_per_s_50_300ms: float
    sanity_max_db_error: float


def _parse_delay_csv(text: str) -> List[int]:
    values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if len(values) != 4:
        raise ValueError(f"Expected exactly 4 delay samples, got: {values}")
    return values


def _normalize_vector(values: Sequence[float], fallback: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        arr = np.asarray(fallback, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
    return [float(v / max(norm, 1e-12)) for v in arr]


def _normalize_u(values: Sequence[float]) -> List[float]:
    return _normalize_vector(values, DEFAULT_FIXED_U)


def _safe_vec4(payload: Dict[str, object], key: str, fallback: Sequence[float]) -> List[float]:
    raw = payload.get(key)
    if isinstance(raw, list) and len(raw) == 4:
        try:
            return [float(v) for v in raw]
        except Exception:
            return [float(v) for v in fallback]
    return [float(v) for v in fallback]


def _resolve_fixed_u(base: Dict[str, object]) -> Tuple[List[float], str]:
    raw = base.get("fixed_u")
    if isinstance(raw, list) and len(raw) == 4:
        return _normalize_u([float(v) for v in raw]), "preset"
    return _normalize_u(DEFAULT_FIXED_U), "constant"


def _infer_delay_set(config_id: str, delay_samples: Sequence[int]) -> str:
    cid = config_id.lower()
    if "prime" in cid:
        return "prime"
    if "spread" in cid:
        return "spread"
    if list(delay_samples) == [1499, 2377, 3217, 4421]:
        return "prime"
    if list(delay_samples) == [1200, 1800, 2400, 3000]:
        return "spread"
    return "custom"


def _slug(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in text)


def _preset_rt60_target(payload: Dict[str, object], fallback: float) -> float:
    raw = payload.get("rt60_target", payload.get("rt60", fallback))
    try:
        return float(raw)
    except Exception:
        return float(fallback)


def _preset_gamma_used(payload: Dict[str, object], fs: float, rt60_target: float) -> float:
    raw = payload.get("gamma_used", payload.get("gamma"))
    if raw is not None:
        try:
            return float(raw)
        except Exception:
            pass
    fs_safe = max(float(fs), 1.0)
    rt60_safe = max(float(rt60_target), 1e-3)
    return float(10.0 ** (-3.0 / (fs_safe * rt60_safe)))


def _decay_label(payload: Dict[str, object], fs: float, fallback_rt60: float) -> str:
    rt60_target = _preset_rt60_target(payload, fallback_rt60)
    gamma_used = _preset_gamma_used(payload, fs, rt60_target)
    source = str(payload.get("gamma_source", "rt60_target")).lower()
    if source == "explicit_gamma":
        return f"gamma={gamma_used:.7f}"
    return f"rt60={rt60_target:.2f}s"


def _git_commit_or_unknown(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def _build_gen_ir_if_needed(gen_ir_bin: Path, gen_ir_src: Path) -> None:
    if gen_ir_bin.exists():
        return
    gen_ir_bin.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "c++",
        "-std=c++17",
        "-O2",
        "-Wall",
        "-Wextra",
        "-pedantic",
        str(gen_ir_src),
        "-o",
        str(gen_ir_bin),
    ]
    subprocess.run(cmd, check=True)


def _render_ir(gen_ir_bin: Path, preset_path: Path, wav_path: Path, seconds: float) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(gen_ir_bin), str(preset_path), str(wav_path), f"{seconds:.6f}"]
    subprocess.run(cmd, check=True)


def _load_preset(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    if "u" not in payload:
        raise ValueError(f"Preset missing required field 'u': {path}")
    return payload


def _find_presets_from_args(args: argparse.Namespace) -> List[Path]:
    if args.preset:
        return [Path(p).resolve() for p in args.preset]

    preset_dir = Path(args.preset_dir)
    if args.config_id:
        out: List[Path] = []
        for cfg in args.config_id:
            p = (preset_dir / f"{cfg}.json").resolve()
            if not p.exists():
                raise FileNotFoundError(f"Preset not found for --config-id {cfg}: {p}")
            out.append(p)
        return out

    if args.sr is not None and args.delay_samples is not None and args.rt60 is not None:
        target_delay = _parse_delay_csv(args.delay_samples)
        target_sr = int(args.sr)
        target_rt60 = float(args.rt60)
        matches: List[Path] = []
        for p in sorted(preset_dir.glob("*.json")):
            try:
                payload = _load_preset(p)
                if str(payload.get("matrix_type", "")).lower() != "householder":
                    continue
                if int(payload.get("sr", -1)) != target_sr:
                    continue
                if [int(v) for v in payload.get("delay_samples", [])] != target_delay:
                    continue
                if abs(float(payload.get("rt60", -1.0)) - target_rt60) > 1e-6:
                    continue
                matches.append(p.resolve())
            except Exception:
                continue
        if not matches:
            raise FileNotFoundError(
                "No householder preset matched tuple "
                f"(sr={target_sr}, delay_samples={target_delay}, rt60={target_rt60}) in {preset_dir}"
            )
        return matches

    raise SystemExit(
        "Provide either --preset, or --config-id, or full tuple (--sr --delay-samples --rt60)."
    )


def _scope_modes(scope: str) -> List[str]:
    if scope == "fixed":
        return ["fixed"]
    if scope == "u_only":
        return ["fixed", "u_only"]
    if scope == "full":
        return ["fixed", "full"]
    if scope == "all":
        return ["fixed", "u_only", "full"]
    raise ValueError(f"Unsupported scope: {scope}")


def _build_mode_preset(
    base: Dict[str, object],
    *,
    mode_key: str,
    base_config_id: str,
    fixed_u: Sequence[float],
) -> Dict[str, object]:
    p = dict(base)
    p["matrix_type"] = "householder"

    if mode_key == "fixed":
        p["u"] = _normalize_u(fixed_u)
        p["b"] = list(DEFAULT_B)
        p["cL"] = list(DEFAULT_CL)
        p["cR"] = list(DEFAULT_CR)
    elif mode_key == "u_only":
        p["u"] = _normalize_u([float(v) for v in base["u"]])  # type: ignore[index]
        p["b"] = list(DEFAULT_B)
        p["cL"] = list(DEFAULT_CL)
        p["cR"] = list(DEFAULT_CR)
    elif mode_key == "full":
        p["u"] = _normalize_u([float(v) for v in base["u"]])  # type: ignore[index]
        p["b"] = _safe_vec4(base, "b", DEFAULT_B)
        p["cL"] = _safe_vec4(base, "cL", DEFAULT_CL)
        p["cR"] = _safe_vec4(base, "cR", DEFAULT_CR)
    else:
        raise ValueError(f"Unknown mode_key: {mode_key}")

    p["config_id"] = f"{base_config_id}_{mode_key}"
    p["comparison_mode"] = mode_key
    # Enforce LTI comparison conditions so analytic H(w) and IR render are aligned.
    p["mod_depth"] = float(LTI_COMPARE_MOD_DEPTH)
    p["detune"] = float(LTI_COMPARE_DETUNE)
    p["damp_hz"] = float(LTI_COMPARE_DAMP_HZ)
    p["compare_lti_enforced"] = True
    return p


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _load_wav_channel(
    path: Path,
    channel: str,
    *,
    trim_leading_silence: bool,
    trim_threshold_db: float,
    trim_max_seconds: float,
) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(path))
    if data.ndim == 1:
        x = data.astype(np.float64)
    else:
        idx = 0 if channel.upper() == "L" else 1
        if idx >= data.shape[1]:
            raise RuntimeError(f"Requested channel {channel}, but WAV shape is {data.shape}")
        x = data[:, idx].astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        full_scale = float(max(abs(info.min), info.max))
        x /= full_scale

    if trim_leading_silence:
        thr = 10.0 ** (float(trim_threshold_db) / 20.0)
        scan_n = min(x.size, int(max(0.0, float(trim_max_seconds)) * float(sr)))
        start = 0
        for i in range(scan_n):
            if abs(float(x[i])) > thr:
                start = i
                break
        if start > 0:
            x = x[start:]

    return int(sr), x


def _schroeder_edc_db(x: np.ndarray) -> np.ndarray:
    energy = np.cumsum((x[::-1] ** 2))[::-1]
    energy = np.maximum(energy, 1e-20)
    return 10.0 * np.log10(energy / energy[0])


def _fit_decay(
    t: np.ndarray, edc_db: np.ndarray, upper_db: float, lower_db: float
) -> Tuple[float, str]:
    mask = (edc_db <= upper_db) & (edc_db >= lower_db)
    if int(np.sum(mask)) < 8:
        return math.nan, "none"
    slope, _intercept = np.polyfit(t[mask], edc_db[mask], 1)
    if slope >= 0.0:
        return math.nan, "none"
    return float(-60.0 / slope), "ok"


def _rt_metrics(t: np.ndarray, edc_db: np.ndarray) -> Tuple[float, float, str]:
    edt_s, edt_status = _fit_decay(t, edc_db, upper_db=0.0, lower_db=-10.0)
    t20_s, t20_status = _fit_decay(t, edc_db, upper_db=-5.0, lower_db=-25.0)
    t30_s, t30_status = _fit_decay(t, edc_db, upper_db=-5.0, lower_db=-35.0)

    if t30_status == "ok":
        rt60_s = float(t30_s)
        method = "T30"
    elif t20_status == "ok":
        rt60_s = float(t20_s)
        method = "T20"
    else:
        rt60_s = math.nan
        method = "none"

    return float(edt_s), float(rt60_s), method


def _ringiness_proxy(x: np.ndarray) -> float:
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
    return float(np.max(spec) / (np.mean(spec) + EPS))


def _moving_average(x: np.ndarray, win_bins: int) -> np.ndarray:
    w = max(int(win_bins), 1)
    if w <= 1:
        return x.copy()
    ker = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x, ker, mode="same")


def _spectral_loss_like_from_mag(mag_lin: np.ndarray) -> float:
    rel = mag_lin / (np.mean(mag_lin[1:]) + EPS)
    return float(np.mean((rel[1:] - 1.0) ** 2))


def _spectral_dev_db(smoothed_mag_db: np.ndarray, freqs: np.ndarray) -> float:
    mask = freqs >= 50.0
    if not np.any(mask):
        return 0.0
    vals = smoothed_mag_db[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    return float(np.std(vals))


def _short_time_excess_kurtosis(
    x: np.ndarray,
    sr: int,
    *,
    win_ms: float = 20.0,
    hop_ms: float = 5.0,
    rms_threshold_db: float = -60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    win = max(16, int(float(sr) * win_ms * 0.001))
    hop = max(1, int(float(sr) * hop_ms * 0.001))
    if x.size < win:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    rms_threshold = 10.0 ** (float(rms_threshold_db) / 20.0)
    times: List[float] = []
    kurtosis: List[float] = []
    for start in range(0, x.size - win + 1, hop):
        frame = x[start : start + win]
        frame_rms = float(np.sqrt(np.mean(frame * frame) + EPS))
        if frame_rms < rms_threshold:
            times.append((start + 0.5 * win) / float(sr))
            kurtosis.append(float("nan"))
            continue
        # Diffusion proxy uses RMS-normalized frames to avoid amplitude-driven blowups.
        frame_n = frame / (frame_rms + EPS)
        mu = float(np.mean(frame_n))
        centered = frame_n - mu
        m2 = float(np.mean(centered * centered))
        if m2 < 1e-10:
            k = float("nan")
        else:
            m4 = float(np.mean(centered * centered * centered * centered))
            k = float(m4 / (m2 * m2 + EPS) - 3.0)
        times.append((start + 0.5 * win) / float(sr))
        kurtosis.append(k)

    return np.asarray(times, dtype=np.float64), np.asarray(kurtosis, dtype=np.float64)


def _short_time_rms_envelope(
    x: np.ndarray,
    sr: int,
    *,
    win_ms: float,
    hop_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    win = max(8, int(float(sr) * win_ms * 0.001))
    hop = max(1, int(float(sr) * hop_ms * 0.001))
    if x.size < win:
        return np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float64)

    times: List[float] = []
    env: List[float] = []
    for start in range(0, x.size - win + 1, hop):
        frame = x[start : start + win]
        frame_rms = float(np.sqrt(np.mean(frame * frame) + EPS))
        times.append((start + 0.5 * win) / float(sr))
        env.append(frame_rms)
    return np.asarray(times, dtype=np.float64), np.asarray(env, dtype=np.float64)


def _echo_density_curve(
    x: np.ndarray,
    sr: int,
    *,
    threshold_db: float,
    min_spacing_ms: float,
    window_ms: float,
    hop_ms: float,
    tmin: float,
    tmax: float,
) -> Tuple[np.ndarray, np.ndarray, float]:
    t_env, env = _short_time_rms_envelope(x, sr, win_ms=window_ms, hop_ms=hop_ms)
    if env.size == 0:
        return t_env, np.zeros_like(t_env), math.nan

    thr = float(np.max(env) * (10.0 ** (float(threshold_db) / 20.0)))
    min_spacing_s = max(float(min_spacing_ms), 0.0) * 1e-3
    half_win_s = 0.5 * max(float(window_ms), 1e-3) * 1e-3
    win_s = max(float(window_ms), 1e-3) * 1e-3

    peak_times: List[float] = []
    last_peak_t = -1e12
    if env.size >= 3:
        for i in range(1, env.size - 1):
            if env[i] < thr:
                continue
            if env[i] < env[i - 1] or env[i] < env[i + 1]:
                continue
            ti = float(t_env[i])
            if (ti - last_peak_t) < min_spacing_s:
                continue
            peak_times.append(ti)
            last_peak_t = ti
    elif env.size >= 1 and env[0] >= thr:
        peak_times.append(float(t_env[0]))

    curve = np.zeros_like(t_env)
    if peak_times:
        peaks = np.asarray(peak_times, dtype=np.float64)
        for i, ti in enumerate(t_env):
            lo = float(ti) - half_win_s
            hi = float(ti) + half_win_s
            count = np.count_nonzero((peaks >= lo) & (peaks <= hi))
            curve[i] = float(count) / win_s

    summary = _mean_in_window(t_env, curve, tmin, tmax)
    return t_env, curve, float(summary)


def _mean_in_window(t: np.ndarray, y: np.ndarray, t0: float, t1: float) -> float:
    mask = (t >= t0) & (t <= t1)
    if not np.any(mask):
        return math.nan
    vals = y[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan
    return float(np.mean(vals))


def _hadamard4() -> np.ndarray:
    return 0.5 * np.array(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _householder_matrix(u: Sequence[float]) -> np.ndarray:
    uu = np.asarray(_normalize_u(u), dtype=np.float64)
    eye = np.eye(uu.size, dtype=np.float64)
    return eye - 2.0 * np.outer(uu, uu)


def _analytic_transfer_mag(
    preset: Dict[str, object],
    *,
    channel: str,
    nfft: int,
) -> Tuple[np.ndarray, np.ndarray]:
    delay = np.asarray([int(v) for v in preset["delay_samples"]], dtype=np.float64)  # type: ignore[index]
    gains = np.asarray([float(v) for v in preset["gains"]], dtype=np.float64)  # type: ignore[index]
    b = np.asarray([float(v) for v in preset["b"]], dtype=np.float64)  # type: ignore[index]
    c_key = "cL" if channel.upper() == "L" else "cR"
    c = np.asarray([float(v) for v in preset[c_key]], dtype=np.float64)  # type: ignore[index]

    matrix_type = str(preset.get("matrix_type", "householder")).lower()
    if matrix_type == "hadamard":
        U = _hadamard4()
    else:
        U = _householder_matrix([float(v) for v in preset["u"]])  # type: ignore[index]

    UG = U @ np.diag(gains)
    bins = np.arange((nfft // 2) + 1, dtype=np.float64)
    omegas = (2.0 * math.pi / float(nfft)) * bins

    n = delay.size
    eye = np.eye(n, dtype=np.complex128)
    H = np.zeros(bins.size, dtype=np.complex128)

    for i, w in enumerate(omegas):
        d = np.exp(-1j * w * delay)
        F = d[:, None] * UG
        rhs = d * b
        x = np.linalg.solve(eye - F + 1e-12 * eye, rhs)
        H[i] = np.sum(c * x)

    mag = np.abs(H)
    rel = mag / (np.mean(mag[1:]) + EPS)
    db = 20.0 * np.log10(np.maximum(rel, EPS))
    return mag, db


def _plot_overlay_with_delta(
    out_path: Path,
    *,
    title: str,
    freqs: np.ndarray,
    db_by_mode: Dict[str, np.ndarray],
    smooth_bins: int,
    y_axis_label: str,
    max_freq_hz: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    smoothed: Dict[str, np.ndarray] = {
        k: _moving_average(v, smooth_bins) for k, v in db_by_mode.items()
    }
    freq_mask = freqs <= float(max_freq_hz)
    if not np.any(freq_mask):
        freq_mask = np.ones_like(freqs, dtype=bool)
    freqs_plot = freqs[freq_mask]

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 10.0), sharex=True)
    fig.suptitle(title, fontsize=PLOT_TITLE_FONTSIZE)

    for mode_key in MODE_ORDER:
        if mode_key not in db_by_mode:
            continue
        axes[0].plot(
            freqs_plot,
            db_by_mode[mode_key][freq_mask],
            label=MODE_LABELS[mode_key],
            lw=PLOT_LINEWIDTH_AUX,
            color=MODE_COLORS[mode_key],
        )
    axes[0].axhline(0.0, color="0.5", lw=0.9, ls="--")
    axes[0].set_ylabel(y_axis_label, fontsize=PLOT_LABEL_FONTSIZE)
    axes[0].set_title("Raw overlay", fontsize=PLOT_LABEL_FONTSIZE)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=PLOT_LEGEND_FONTSIZE)
    axes[0].tick_params(labelsize=PLOT_LEGEND_FONTSIZE)

    for mode_key in MODE_ORDER:
        if mode_key not in smoothed:
            continue
        axes[1].plot(
            freqs_plot,
            smoothed[mode_key][freq_mask],
            label=MODE_LABELS[mode_key],
            lw=PLOT_LINEWIDTH_MAIN,
            color=MODE_COLORS[mode_key],
        )
    axes[1].axhline(0.0, color="0.5", lw=0.9, ls="--", label="0 dB = mean magnitude")
    axes[1].set_ylabel(y_axis_label, fontsize=PLOT_LABEL_FONTSIZE)
    axes[1].set_title("Smoothed overlay", fontsize=PLOT_LABEL_FONTSIZE)
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=PLOT_LEGEND_FONTSIZE)
    axes[1].tick_params(labelsize=PLOT_LEGEND_FONTSIZE)

    fixed_key = "fixed"
    delta_summary: List[str] = []
    for mode_key in MODE_ORDER:
        if mode_key == fixed_key or mode_key not in smoothed:
            continue
        if fixed_key not in smoothed:
            continue
        delta = smoothed[mode_key] - smoothed[fixed_key]
        delta_band = delta[freq_mask]
        mean_abs = float(np.mean(np.abs(delta_band)))
        max_abs = float(np.max(np.abs(delta_band)))
        delta_summary.append(f"{mode_key}: mean|Δ|={mean_abs:.2f} dB, max|Δ|={max_abs:.2f} dB")
        axes[2].plot(
            freqs_plot,
            delta_band,
            lw=PLOT_LINEWIDTH_MAIN,
            color=MODE_COLORS[mode_key],
            label=f"{MODE_LABELS[mode_key]} - {MODE_LABELS[fixed_key]} (dB)",
        )
    axes[2].axhline(0.0, color="0.5", lw=0.9, ls="--")
    axes[2].set_xlabel("Frequency (Hz)", fontsize=PLOT_LABEL_FONTSIZE)
    axes[2].set_ylabel("Delta (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    axes[2].set_title("Delta vs fixed baseline", fontsize=PLOT_LABEL_FONTSIZE)
    axes[2].grid(alpha=0.25)
    if delta_summary:
        axes[2].text(
            0.01,
            0.98,
            "\n".join(delta_summary),
            transform=axes[2].transAxes,
            va="top",
            ha="left",
            fontsize=PLOT_LEGEND_FONTSIZE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )
    handles, labels = axes[2].get_legend_handles_labels()
    if handles:
        axes[2].legend(fontsize=PLOT_LEGEND_FONTSIZE)
    axes[2].tick_params(labelsize=PLOT_LEGEND_FONTSIZE)
    axes[2].set_xlim(float(freqs_plot[0]), float(freqs_plot[-1]))

    # Keep frequency-axis y-range consistent across modes for readability.
    y_candidates = []
    for mode_key in db_by_mode:
        y_candidates.append(db_by_mode[mode_key][freq_mask])
        y_candidates.append(smoothed[mode_key][freq_mask])
    if y_candidates:
        y_all = np.concatenate(y_candidates)
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size > 0:
            y_min = float(np.percentile(y_all, 2))
            y_max = float(np.percentile(y_all, 98))
            pad = 0.1 * max(1e-6, y_max - y_min)
            axes[0].set_ylim(y_min - pad, y_max + pad)
            axes[1].set_ylim(y_min - pad, y_max + pad)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_diffusion_curve(
    out_path: Path,
    *,
    config_id: str,
    results: Dict[str, ScenarioResult],
    rms_threshold_db: float,
    echo_tmin: float,
    echo_tmax: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)
    fig.suptitle(
        f"Diffusion / echo-density proxies ({config_id}, RMS>{rms_threshold_db:.0f} dBFS)",
        fontsize=PLOT_TITLE_FONTSIZE,
    )
    ax_k = axes[0]
    ax_e = axes[1]

    ax_k.axvspan(
        echo_tmin,
        echo_tmax,
        color="0.92",
        alpha=0.8,
        label=f"summary window ({int(echo_tmin*1000)}-{int(echo_tmax*1000)} ms)",
    )
    ax_e.axvspan(
        echo_tmin,
        echo_tmax,
        color="0.92",
        alpha=0.8,
        label=f"summary window ({int(echo_tmin*1000)}-{int(echo_tmax*1000)} ms)",
    )
    for mode_key in MODE_ORDER:
        if mode_key not in results:
            continue
        r = results[mode_key]
        mean_txt = f"mean50-300={r.kurtosis_mean_50_300ms:.3f}" if np.isfinite(r.kurtosis_mean_50_300ms) else "mean50-300=N/A"
        ax_k.plot(
            r.kurt_t,
            r.kurtosis_curve,
            lw=PLOT_LINEWIDTH_MAIN,
            color=MODE_COLORS[mode_key],
            label=f"{MODE_LABELS[mode_key]} ({mean_txt})",
        )
        echo_txt = (
            f"events/s50-300={r.echo_density_events_per_s_50_300ms:.2f}"
            if np.isfinite(r.echo_density_events_per_s_50_300ms)
            else "events/s50-300=N/A"
        )
        ax_e.plot(
            r.echo_density_t,
            r.echo_density_curve,
            lw=PLOT_LINEWIDTH_MAIN,
            color=MODE_COLORS[mode_key],
            label=f"{MODE_LABELS[mode_key]} ({echo_txt})",
        )

    ax_k.set_ylabel("Excess kurtosis\n(lower = more diffuse)", fontsize=PLOT_LABEL_FONTSIZE)
    ax_k.grid(alpha=0.25)
    ax_k.legend(fontsize=PLOT_LEGEND_FONTSIZE)
    ax_k.set_title("Kurtosis proxy", fontsize=PLOT_LABEL_FONTSIZE)
    ax_k.tick_params(labelsize=PLOT_LEGEND_FONTSIZE)

    ax_e.set_xlabel("Time (s)", fontsize=PLOT_LABEL_FONTSIZE)
    ax_e.set_ylabel("Echo density\n(events/s)", fontsize=PLOT_LABEL_FONTSIZE)
    ax_e.grid(alpha=0.25)
    ax_e.legend(fontsize=PLOT_LEGEND_FONTSIZE)
    ax_e.set_title("Echo-density proxy", fontsize=PLOT_LABEL_FONTSIZE)
    ax_e.tick_params(labelsize=PLOT_LEGEND_FONTSIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_edc_overlay(
    out_path: Path,
    *,
    config_id: str,
    results: Dict[str, ScenarioResult],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_title(f"EDC overlay ({config_id})", fontsize=PLOT_TITLE_FONTSIZE)
    ax.axhspan(-10.0, 0.0, color="0.95", alpha=0.8, label="EDT fit band (0 to -10 dB)")
    ax.axhspan(-25.0, -5.0, color="0.92", alpha=0.7, label="T20 fit band (-5 to -25 dB)")
    ax.axhspan(-35.0, -5.0, color="0.88", alpha=0.45, label="T30 fit band (-5 to -35 dB)")

    for mode_key in MODE_ORDER:
        if mode_key not in results:
            continue
        r = results[mode_key]
        ax.plot(
            r.t,
            r.edc_db,
            lw=PLOT_LINEWIDTH_MAIN,
            color=MODE_COLORS[mode_key],
            label=f"{MODE_LABELS[mode_key]} (RT60={r.rt60_s:.2f}s {r.rt60_method})",
        )

    ax.set_xlabel("Time (s)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel("EDC (dB)", fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylim(-80.0, 2.0)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=PLOT_LEGEND_FONTSIZE, loc="upper right")
    ax.tick_params(labelsize=PLOT_LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _max_db_error(
    a_db: np.ndarray,
    b_db: np.ndarray,
    freqs: np.ndarray,
    floor_db: float = -60.0,
    max_freq_hz: float = 12000.0,
) -> float:
    if a_db.shape != b_db.shape:
        raise ValueError(f"Shape mismatch for sanity check: {a_db.shape} vs {b_db.shape}")
    if a_db.size <= 1:
        return 0.0
    if freqs.shape != a_db.shape:
        raise ValueError(f"Frequency shape mismatch for sanity check: {freqs.shape} vs {a_db.shape}")
    band = freqs[1:] <= float(max_freq_hz)
    valid = band & (a_db[1:] > floor_db) & (b_db[1:] > floor_db)
    if not np.any(valid):
        return 0.0
    diff = np.abs(a_db[1:][valid] - b_db[1:][valid])
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return 0.0
    return float(np.max(diff))


def _bar_values(vals: Sequence[float]) -> np.ndarray:
    out = np.asarray(vals, dtype=np.float64)
    out[~np.isfinite(out)] = 0.0
    return out


def _plot_metrics_bars(
    out_path: Path,
    *,
    config_id: str,
    mode_keys: List[str],
    results: Dict[str, ScenarioResult],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [MODE_LABELS[k] for k in mode_keys]
    x = np.arange(len(mode_keys))
    colors = [MODE_COLORS[k] for k in mode_keys]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    fig.suptitle(f"Fixed vs Diff metrics ({config_id})", fontsize=PLOT_TITLE_FONTSIZE)

    edt_vals = _bar_values([results[k].edt_s for k in mode_keys])
    rt_vals = _bar_values([results[k].rt60_s for k in mode_keys])
    spec_vals = _bar_values([results[k].spectral_dev_db for k in mode_keys])
    ring_vals = _bar_values([results[k].ringiness for k in mode_keys])
    kurt_vals = _bar_values([results[k].kurtosis_mean_50_300ms for k in mode_keys])
    echo_vals = _bar_values([results[k].echo_density_events_per_s_50_300ms for k in mode_keys])

    axes[0, 0].bar(x, edt_vals, color=colors)
    axes[0, 0].set_title("EDT", fontsize=PLOT_LABEL_FONTSIZE)
    axes[0, 0].set_ylabel("seconds", fontsize=PLOT_LABEL_FONTSIZE)

    axes[0, 1].bar(x, rt_vals, color=colors)
    axes[0, 1].set_title("RT60", fontsize=PLOT_LABEL_FONTSIZE)
    axes[0, 1].set_ylabel("seconds", fontsize=PLOT_LABEL_FONTSIZE)

    axes[0, 2].bar(x, spec_vals, color=colors)
    axes[0, 2].set_title("Spectral deviation", fontsize=PLOT_LABEL_FONTSIZE)
    axes[0, 2].set_ylabel("dB std (50 Hz - Nyquist)", fontsize=PLOT_LABEL_FONTSIZE)

    axes[1, 0].bar(x, ring_vals, color=colors)
    axes[1, 0].set_title("Ringiness", fontsize=PLOT_LABEL_FONTSIZE)
    axes[1, 0].set_ylabel("peak/mean", fontsize=PLOT_LABEL_FONTSIZE)

    axes[1, 1].bar(x, kurt_vals, color=colors)
    axes[1, 1].set_title("Kurtosis diffusion proxy", fontsize=PLOT_LABEL_FONTSIZE)
    axes[1, 1].set_ylabel("mean excess kurtosis (50-300 ms)", fontsize=PLOT_LABEL_FONTSIZE)

    axes[1, 2].bar(x, echo_vals, color=colors)
    axes[1, 2].set_title("Echo-density proxy", fontsize=PLOT_LABEL_FONTSIZE)
    axes[1, 2].set_ylabel("events/s (50-300 ms)", fontsize=PLOT_LABEL_FONTSIZE)

    for ax in (axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]):
        ax.set_xticks(x, labels, rotation=10, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=PLOT_LEGEND_FONTSIZE)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _write_sidecar(
    *,
    path: Path,
    base_config_id: str,
    scope: str,
    mode_key: str,
    source_preset_path: Path,
    render_preset_path: Path,
    wav_path: Path,
    preset: Dict[str, object],
    git_commit: str,
    normalization_peak: float,
    ir_seconds: float,
    channel: str,
    fixed_u_source: str,
    fixed_u: Sequence[float],
    nfft_used: int,
    aligned_length_samples: int,
    echo_density_threshold_db: float,
    echo_density_min_spacing_ms: float,
    echo_density_window_ms: float,
    echo_density_hop_ms: float,
    echo_density_tmin: float,
    echo_density_tmax: float,
) -> None:
    rt60_target = _preset_rt60_target(preset, float(preset.get("rt60", 2.8)))
    gamma_used = _preset_gamma_used(preset, float(preset["sr"]), rt60_target)
    payload = {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": git_commit,
        "compare_script": "eval/scripts/compare_fixed_vs_diff.py",
        "renderer": "eval/tools/gen_ir.cpp",
        "base_config_id": base_config_id,
        "scope": scope,
        "mode_key": mode_key,
        "mode_label": MODE_LABELS[mode_key],
        "fixed_u_source": fixed_u_source,
        "fixed_u": [float(v) for v in fixed_u],
        "source_preset_path": str(source_preset_path),
        "render_preset_path": str(render_preset_path),
        "wav_path": str(wav_path),
        "config_id": str(preset.get("config_id", base_config_id)),
        "sr": int(preset["sr"]),
        "nfft": int(preset.get("nfft", 2048)),
        "nfft_used": int(nfft_used),
        "aligned_length_samples": int(aligned_length_samples),
        "delay_set": _infer_delay_set(str(base_config_id), [int(v) for v in preset["delay_samples"]]),  # type: ignore[index]
        "delay_samples": [int(v) for v in preset["delay_samples"]],  # type: ignore[index]
        "rt60": float(preset.get("rt60", rt60_target)),
        "rt60_target": float(rt60_target),
        "gamma_used": float(gamma_used),
        "gamma_source": str(preset.get("gamma_source", "unknown")),
        "gains": [float(v) for v in preset["gains"]],  # type: ignore[index]
        "matrix_type": str(preset.get("matrix_type", "householder")),
        "u": [float(v) for v in preset["u"]],  # type: ignore[index]
        "b": [float(v) for v in preset["b"]],  # type: ignore[index]
        "cL": [float(v) for v in preset["cL"]],  # type: ignore[index]
        "cR": [float(v) for v in preset["cR"]],  # type: ignore[index]
        "ir_seconds": float(ir_seconds),
        "channel_used_for_analysis": channel,
        "normalization_peak_common": float(normalization_peak),
        "diffusion_proxy_name": "short_time_excess_kurtosis_rmsnorm",
        "diffusion_proxy_definition": (
            "Per-frame excess kurtosis on RMS-normalized frames; windows below "
            "RMS threshold are ignored. Lower values indicate more Gaussian-like "
            "amplitude statistics and higher diffusion (proxy)."
        ),
        "diffusion_proxy_window_ms": 20.0,
        "diffusion_proxy_hop_ms": 5.0,
        "diffusion_proxy_summary_window_s": [0.05, 0.30],
        "echo_density_proxy_name": "envelope_peak_event_rate",
        "echo_density_proxy_definition": (
            "Short-time RMS envelope peak/event rate with dynamic threshold and "
            "minimum peak spacing. Summary is mean events/s in the configured window."
        ),
        "echo_density_threshold_db": float(echo_density_threshold_db),
        "echo_density_min_spacing_ms": float(echo_density_min_spacing_ms),
        "echo_density_window_ms": float(echo_density_window_ms),
        "echo_density_hop_ms": float(echo_density_hop_ms),
        "echo_density_summary_window_s": [float(echo_density_tmin), float(echo_density_tmax)],
        "compare_lti_enforced": bool(preset.get("compare_lti_enforced", False)),
        "compare_mod_depth": float(preset.get("mod_depth", 0.0)),
        "compare_detune": float(preset.get("detune", 0.0)),
        "compare_damp_hz": float(preset.get("damp_hz", 0.0)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_run_payload(
    *,
    path: Path,
    base_config_id: str,
    mode_keys: Sequence[str],
    results: Dict[str, ScenarioResult],
) -> None:
    payload = {
        "config_id": base_config_id,
        "modes": {},
    }
    for mode_key in mode_keys:
        if mode_key not in results:
            continue
        r = results[mode_key]
        payload["modes"][mode_key] = {
            "label": r.label,
            "metrics": {
                "spectral_loss_like": float(r.spectral_loss_like),
                "spectral_dev_db": float(r.spectral_dev_db),
                "rt60_s": float(r.rt60_s),
                "edt_s": float(r.edt_s),
                "ringiness": float(r.ringiness),
                "kurtosis_mean_50_300ms": float(r.kurtosis_mean_50_300ms),
                "echo_density_events_per_s_50_300ms": float(r.echo_density_events_per_s_50_300ms),
                "sanity_max_db_error": float(r.sanity_max_db_error),
            },
            "curves": {
                "kurtosis": {
                    "t": [float(v) for v in r.kurt_t.tolist()],
                    "y": [float(v) for v in r.kurtosis_curve.tolist()],
                },
                "echo_density": {
                    "t": [float(v) for v in r.echo_density_t.tolist()],
                    "y": [float(v) for v in r.echo_density_curve.tolist()],
                },
            },
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", action="append", help="Preset JSON path (repeat for batch)")
    parser.add_argument("--config-id", action="append", help="Config id in --preset-dir (repeat for batch)")
    parser.add_argument("--preset-dir", default="eval/out/presets", help="Preset folder for config lookup")
    parser.add_argument("--sr", type=int, default=None, help="Tuple lookup only: sample rate")
    parser.add_argument("--delay-samples", default=None, help="Tuple lookup only: e.g. 1499,2377,3217,4421")
    parser.add_argument("--rt60", type=float, default=None, help="Tuple lookup only: target RT60")
    parser.add_argument("--scope", choices=["fixed", "u_only", "full", "all"], default="all")
    parser.add_argument("--channel", choices=["L", "R"], default="L", help="Channel for analysis/FFT")
    parser.add_argument("--seconds", type=float, default=None, help="IR seconds (default: max(8, 4*rt60))")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic auxiliary ops")
    parser.add_argument(
        "--nfft",
        type=int,
        default=None,
        help=(
            "Minimum FFT size. Actual FFT uses max(requested, aligned IR length) "
            "to avoid truncation (default requested value: preset nfft)."
        ),
    )
    parser.add_argument("--smooth-bins", type=int, default=25, help="Smoothing window for dB overlays")
    parser.add_argument("--max-freq-hz", type=float, default=12000.0, help="Max frequency shown in overlays")
    parser.add_argument(
        "--sanity-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify analytic-vs-IR FFT max dB error per scope (default: enabled)",
    )
    parser.add_argument(
        "--kurtosis-rms-threshold-db",
        type=float,
        default=-60.0,
        help="Ignore diffusion windows below this RMS (dBFS)",
    )
    parser.add_argument(
        "--echo-density-threshold-db",
        type=float,
        default=-30.0,
        help="Dynamic threshold for echo-density event detection (dB relative to max envelope)",
    )
    parser.add_argument(
        "--echo-density-min-spacing-ms",
        type=float,
        default=1.0,
        help="Minimum spacing between counted envelope peaks (ms)",
    )
    parser.add_argument(
        "--echo-density-window-ms",
        type=float,
        default=10.0,
        help="RMS envelope window and event-rate integration window (ms)",
    )
    parser.add_argument(
        "--echo-density-hop-ms",
        type=float,
        default=5.0,
        help="Hop size for echo-density envelope/event curve (ms)",
    )
    parser.add_argument(
        "--echo-density-tmin",
        type=float,
        default=0.05,
        help="Summary window start for echo-density metric (s)",
    )
    parser.add_argument(
        "--echo-density-tmax",
        type=float,
        default=0.30,
        help="Summary window end for echo-density metric (s)",
    )
    parser.add_argument(
        "--trim-leading-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading silence before analysis (default: enabled)",
    )
    parser.add_argument(
        "--trim-threshold-db",
        type=float,
        default=-60.0,
        help="Leading-silence trim threshold in dBFS (default: -60 dB)",
    )
    parser.add_argument(
        "--trim-max-seconds",
        type=float,
        default=0.5,
        help="Max leading duration to scan for trim (default: 0.5 s)",
    )
    parser.add_argument("--gen-ir-bin", default="eval/out/bin/gen_ir", help="Path to gen_ir binary")
    parser.add_argument("--gen-ir-src", default="eval/tools/gen_ir.cpp", help="Path to gen_ir source")
    parser.add_argument("--out-ir-dir", default="eval/out/ir", help="Output folder for rendered IR WAVs")
    parser.add_argument(
        "--render-preset-dir",
        default="eval/out/presets",
        help="Output folder for generated comparison render presets",
    )
    parser.add_argument("--fig-dir", default="eval/figs", help="Figure output directory")
    parser.add_argument(
        "--summary-csv",
        default="eval/figs/summary_fixed_vs_diff.csv",
        help="Main summary CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))
    repo_root = Path(__file__).resolve().parents[2]
    preset_paths = _find_presets_from_args(args)

    gen_ir_bin = (repo_root / args.gen_ir_bin).resolve()
    gen_ir_src = (repo_root / args.gen_ir_src).resolve()
    out_ir_dir = (repo_root / args.out_ir_dir).resolve()
    out_render_preset_dir = (repo_root / args.render_preset_dir).resolve()
    fig_dir = (repo_root / args.fig_dir).resolve()
    summary_csv = (repo_root / args.summary_csv).resolve()
    summary_table_csv = fig_dir / "fixed_vs_diff_summary_table.csv"

    _build_gen_ir_if_needed(gen_ir_bin, gen_ir_src)
    git_commit = _git_commit_or_unknown(repo_root)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()

    summary_rows: List[Dict[str, object]] = []
    generated_analytic: List[Path] = []
    generated_irfft: List[Path] = []
    generated_edc: List[Path] = []
    generated_diffusion: List[Path] = []
    generated_metrics: List[Path] = []
    generated_run_json: List[Path] = []
    rendered_wavs: List[Path] = []

    for preset_path in preset_paths:
        source = _load_preset(preset_path)
        base_config_id = str(source.get("config_id", preset_path.stem))
        slug = _slug(base_config_id)
        fixed_u, fixed_u_source = _resolve_fixed_u(source)

        mode_keys = _scope_modes(args.scope)
        rt60 = _preset_rt60_target(source, float(source.get("rt60", 2.8)))
        gamma_used_source = _preset_gamma_used(source, float(source.get("sr", 48000)), rt60)
        decay_tag = _decay_label(source, float(source.get("sr", 48000)), rt60)
        seconds = float(args.seconds) if args.seconds is not None else max(8.0, 4.0 * rt60)
        nfft_req = int(args.nfft) if args.nfft is not None else int(source.get("nfft", 2048))
        if nfft_req <= 0:
            raise ValueError(f"nfft must be positive, got {nfft_req}")

        signals_raw: Dict[str, np.ndarray] = {}
        srs: Dict[str, int] = {}
        presets_rendered: Dict[str, Dict[str, object]] = {}
        preset_paths_rendered: Dict[str, Path] = {}
        wav_paths: Dict[str, Path] = {}
        sidecar_paths: Dict[str, Path] = {}

        for mode_key in mode_keys:
            render_preset = _build_mode_preset(
                source,
                mode_key=mode_key,
                base_config_id=base_config_id,
                fixed_u=fixed_u,
            )
            render_preset_path = out_render_preset_dir / f"{slug}_{mode_key}.json"
            _write_json(render_preset_path, render_preset)

            wav_path = out_ir_dir / f"IR_{slug}_{mode_key}.wav"
            _render_ir(gen_ir_bin, render_preset_path, wav_path, seconds)
            rendered_wavs.append(wav_path)

            sr, sig = _load_wav_channel(
                wav_path,
                args.channel,
                trim_leading_silence=bool(args.trim_leading_silence),
                trim_threshold_db=float(args.trim_threshold_db),
                trim_max_seconds=float(args.trim_max_seconds),
            )

            signals_raw[mode_key] = sig
            srs[mode_key] = sr
            presets_rendered[mode_key] = render_preset
            preset_paths_rendered[mode_key] = render_preset_path
            wav_paths[mode_key] = wav_path
            sidecar_paths[mode_key] = wav_path.with_suffix(wav_path.suffix + ".json")

        sr_unique = set(srs.values())
        if len(sr_unique) != 1:
            raise RuntimeError(f"Sample-rate mismatch for {base_config_id}: {srs}")
        sr = next(iter(sr_unique))

        min_len = min(sig.size for sig in signals_raw.values())
        if min_len < 2:
            raise RuntimeError(f"Signals too short after trim/alignment for {base_config_id}: {min_len}")
        nfft = max(nfft_req, min_len)
        if nfft > nfft_req:
            print(
                f"[Info] {base_config_id}: expanded nfft {nfft_req} -> {nfft} "
                f"to avoid IR truncation"
            )

        common_peak = max(float(np.max(np.abs(sig[:min_len]))) for sig in signals_raw.values())
        common_peak = max(common_peak, EPS)

        normalized_signals: Dict[str, np.ndarray] = {
            k: (signals_raw[k][:min_len] / common_peak) for k in mode_keys
        }

        freqs = np.fft.rfftfreq(nfft, d=1.0 / float(sr))
        scenario_results: Dict[str, ScenarioResult] = {}
        analytic_db_by_mode: Dict[str, np.ndarray] = {}
        ir_db_by_mode: Dict[str, np.ndarray] = {}
        sanity_errors_db: Dict[str, float] = {}

        for mode_key in mode_keys:
            sig = normalized_signals[mode_key]
            win = np.hanning(sig.size)
            ir_mag = np.abs(np.fft.rfft(sig * win, n=nfft))
            ir_rel = ir_mag / (np.mean(ir_mag[1:]) + EPS)
            ir_mag_db = 20.0 * np.log10(np.maximum(ir_rel, EPS))
            ir_mag_db_s = _moving_average(ir_mag_db, args.smooth_bins)
            # For analytic-vs-IR sanity, use unwindowed FFT to avoid extra mismatch from tapering.
            ir_mag_unwindowed = np.abs(np.fft.rfft(sig, n=nfft))
            ir_rel_unwindowed = ir_mag_unwindowed / (np.mean(ir_mag_unwindowed[1:]) + EPS)
            ir_mag_db_unwindowed = 20.0 * np.log10(np.maximum(ir_rel_unwindowed, EPS))
            ir_mag_db_unwindowed_s = _moving_average(ir_mag_db_unwindowed, args.smooth_bins)

            analytic_mag_lin, analytic_mag_db = _analytic_transfer_mag(
                presets_rendered[mode_key],
                channel=args.channel,
                nfft=nfft,
            )
            analytic_mag_db_s = _moving_average(analytic_mag_db, args.smooth_bins)

            edc_db = _schroeder_edc_db(sig)
            t = np.arange(sig.size, dtype=np.float64) / float(sr)
            edt_s, rt60_s, rt60_method = _rt_metrics(t, edc_db)

            kurt_t, kurt_curve = _short_time_excess_kurtosis(
                sig,
                sr,
                rms_threshold_db=float(args.kurtosis_rms_threshold_db),
            )
            kurt_mean = _mean_in_window(kurt_t, kurt_curve, 0.05, 0.30)
            echo_t, echo_curve, echo_mean = _echo_density_curve(
                sig,
                sr,
                threshold_db=float(args.echo_density_threshold_db),
                min_spacing_ms=float(args.echo_density_min_spacing_ms),
                window_ms=float(args.echo_density_window_ms),
                hop_ms=float(args.echo_density_hop_ms),
                tmin=float(args.echo_density_tmin),
                tmax=float(args.echo_density_tmax),
            )
            sanity_max_db_error = _max_db_error(
                analytic_mag_db_s,
                ir_mag_db_unwindowed_s,
                freqs,
                max_freq_hz=float(args.max_freq_hz),
            )
            sanity_errors_db[mode_key] = sanity_max_db_error

            result = ScenarioResult(
                key=mode_key,
                label=MODE_LABELS[mode_key],
                preset=presets_rendered[mode_key],
                preset_path=preset_paths_rendered[mode_key],
                wav_path=wav_paths[mode_key],
                sidecar_path=sidecar_paths[mode_key],
                sr=sr,
                signal=sig,
                t=t,
                edc_db=edc_db,
                edt_s=edt_s,
                rt60_s=rt60_s,
                rt60_method=rt60_method,
                ringiness=_ringiness_proxy(sig),
                spectral_loss_like=_spectral_loss_like_from_mag(analytic_mag_lin),
                spectral_dev_db=_spectral_dev_db(analytic_mag_db_s, freqs),
                ir_mag_db=ir_mag_db,
                ir_mag_db_smooth=ir_mag_db_s,
                analytic_mag_db=analytic_mag_db,
                analytic_mag_db_smooth=analytic_mag_db_s,
                kurt_t=kurt_t,
                kurtosis_curve=kurt_curve,
                kurtosis_mean_50_300ms=kurt_mean,
                echo_density_t=echo_t,
                echo_density_curve=echo_curve,
                echo_density_events_per_s_50_300ms=echo_mean,
                sanity_max_db_error=sanity_max_db_error,
            )
            scenario_results[mode_key] = result
            analytic_db_by_mode[mode_key] = result.analytic_mag_db
            ir_db_by_mode[mode_key] = result.ir_mag_db

            _write_sidecar(
                path=result.sidecar_path,
                base_config_id=base_config_id,
                scope=args.scope,
                mode_key=mode_key,
                source_preset_path=preset_path,
                render_preset_path=result.preset_path,
                wav_path=result.wav_path,
                preset=result.preset,
                git_commit=git_commit,
                normalization_peak=common_peak,
                ir_seconds=seconds,
                channel=args.channel,
                fixed_u_source=fixed_u_source,
                fixed_u=fixed_u,
                nfft_used=nfft,
                aligned_length_samples=min_len,
                echo_density_threshold_db=float(args.echo_density_threshold_db),
                echo_density_min_spacing_ms=float(args.echo_density_min_spacing_ms),
                echo_density_window_ms=float(args.echo_density_window_ms),
                echo_density_hop_ms=float(args.echo_density_hop_ms),
                echo_density_tmin=float(args.echo_density_tmin),
                echo_density_tmax=float(args.echo_density_tmax),
            )

            u_vals = [float(v) for v in result.preset["u"]]  # type: ignore[index]
            summary_rows.append(
                {
                    "timestamp": now_iso,
                    "config_id": base_config_id,
                    "scope": mode_key,
                    "mode": result.label,
                    "sr": sr,
                    "delay_set": _infer_delay_set(base_config_id, [int(v) for v in source["delay_samples"]]),  # type: ignore[index]
                    "rt60_target": float(rt60),
                    "gamma_used": float(gamma_used_source),
                    "u0": u_vals[0],
                    "u1": u_vals[1],
                    "u2": u_vals[2],
                    "u3": u_vals[3],
                    "spectral_loss_like": result.spectral_loss_like,
                    "spectral_dev_db": result.spectral_dev_db,
                    "rt60_s": result.rt60_s,
                    "rt60_method": result.rt60_method,
                    "edt_s": result.edt_s,
                    "ringiness": result.ringiness,
                    "kurtosis_mean_50_300ms": result.kurtosis_mean_50_300ms,
                    "echo_density_events_per_s_50_300ms": result.echo_density_events_per_s_50_300ms,
                    "sanity_max_db_error": result.sanity_max_db_error,
                }
            )

        analytic_fig = fig_dir / f"fixed_vs_diff_mag_overlay_{slug}.png"
        irfft_fig = fig_dir / f"fixed_vs_diff_irfft_overlay_{slug}.png"
        edc_fig = fig_dir / f"fixed_vs_diff_edc_overlay_{slug}.png"
        diffusion_fig = fig_dir / f"fixed_vs_diff_diffusion_{slug}.png"
        metrics_fig = fig_dir / f"fixed_vs_diff_metrics_bars_{slug}.png"

        _plot_overlay_with_delta(
            analytic_fig,
            title=f"Analytic transfer magnitude ({base_config_id}, {decay_tag})",
            freqs=freqs,
            db_by_mode=analytic_db_by_mode,
            smooth_bins=args.smooth_bins,
            y_axis_label="Magnitude (dB, mean-normalized)",
            max_freq_hz=float(args.max_freq_hz),
        )
        _plot_overlay_with_delta(
            irfft_fig,
            title=f"IR FFT (windowed) magnitude ({base_config_id}, {decay_tag})",
            freqs=freqs,
            db_by_mode=ir_db_by_mode,
            smooth_bins=args.smooth_bins,
            y_axis_label="Magnitude (dB, mean-normalized)",
            max_freq_hz=float(args.max_freq_hz),
        )
        _plot_edc_overlay(
            edc_fig,
            config_id=f"{base_config_id}, {decay_tag}",
            results=scenario_results,
        )
        _plot_diffusion_curve(
            diffusion_fig,
            config_id=f"{base_config_id}, {decay_tag}",
            results=scenario_results,
            rms_threshold_db=float(args.kurtosis_rms_threshold_db),
            echo_tmin=float(args.echo_density_tmin),
            echo_tmax=float(args.echo_density_tmax),
        )
        _plot_metrics_bars(
            metrics_fig,
            config_id=f"{base_config_id}, {decay_tag}",
            mode_keys=mode_keys,
            results=scenario_results,
        )

        generated_analytic.append(analytic_fig)
        generated_irfft.append(irfft_fig)
        generated_edc.append(edc_fig)
        generated_diffusion.append(diffusion_fig)
        generated_metrics.append(metrics_fig)

        run_payload = fig_dir / f"fixed_vs_diff_run_{slug}.json"
        _write_run_payload(
            path=run_payload,
            base_config_id=base_config_id,
            mode_keys=mode_keys,
            results=scenario_results,
        )
        generated_run_json.append(run_payload)

        if args.sanity_check:
            print(f"Sanity analytic-vs-IR max dB error ({base_config_id}):")
            for mode_key in mode_keys:
                print(f"  - {mode_key}: {sanity_errors_db[mode_key]:.3f} dB")

    if not summary_rows:
        raise RuntimeError("No rows produced. Check inputs.")

    generic_analytic = fig_dir / "fixed_vs_diff_mag_overlay.png"
    generic_irfft = fig_dir / "fixed_vs_diff_irfft_overlay.png"
    generic_edc = fig_dir / "fixed_vs_diff_edc_overlay.png"
    generic_diffusion = fig_dir / "fixed_vs_diff_diffusion.png"
    generic_metrics = fig_dir / "fixed_vs_diff_metrics_bars.png"
    shutil.copyfile(generated_analytic[0], generic_analytic)
    shutil.copyfile(generated_irfft[0], generic_irfft)
    shutil.copyfile(generated_edc[0], generic_edc)
    shutil.copyfile(generated_diffusion[0], generic_diffusion)
    shutil.copyfile(generated_metrics[0], generic_metrics)
    if generated_run_json:
        shutil.copyfile(generated_run_json[0], fig_dir / "fixed_vs_diff_run.json")

    _write_summary_csv(summary_table_csv, summary_rows)
    _write_summary_csv(summary_csv, summary_rows)

    print("Rendered IR WAVs:")
    for wav in rendered_wavs:
        print(f"  - {wav}")
        print(f"    sidecar: {wav}.json")

    print(f"Wrote figure: {generic_analytic}")
    print(f"Wrote figure: {generic_irfft}")
    print(f"Wrote figure: {generic_edc}")
    print(f"Wrote figure: {generic_diffusion}")
    print(f"Wrote figure: {generic_metrics}")
    run_json_generic = fig_dir / "fixed_vs_diff_run.json"
    if run_json_generic.exists():
        print(f"Wrote run JSON: {run_json_generic}")

    if len(generated_analytic) > 1:
        print("Additional per-config figures:")
        for p in generated_analytic:
            print(f"  - {p}")
        for p in generated_irfft:
            print(f"  - {p}")
        for p in generated_edc:
            print(f"  - {p}")
        for p in generated_diffusion:
            print(f"  - {p}")
        for p in generated_metrics:
            print(f"  - {p}")

    print(f"Wrote summary table: {summary_table_csv}")
    print(f"Wrote summary CSV:   {summary_csv}")


if __name__ == "__main__":
    main()
