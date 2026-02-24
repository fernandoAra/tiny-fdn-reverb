#!/usr/bin/env python3
"""Verify transfer_function() against FFT of an IR rendered by gen_ir."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from difffdn.difffdn_tiny import hadamard4_matrix, transfer_function  # type: ignore
    from difffdn.householder import householder_matrix, unit_vector_from_raw  # type: ignore
else:
    from ..difffdn.difffdn_tiny import hadamard4_matrix, transfer_function
    from ..difffdn.householder import householder_matrix, unit_vector_from_raw


def _next_pow2_floor(v: int) -> int:
    if v <= 1:
        return 1
    return 1 << (int(v).bit_length() - 1)


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


def _load_channel(wav_path: Path, channel: str) -> Tuple[int, np.ndarray]:
    sr, data = wavfile.read(str(wav_path))
    if data.ndim == 1:
        x = data.astype(np.float64)
    else:
        idx = 0 if channel == "L" else 1
        if idx >= data.shape[1]:
            raise ValueError(f"Requested channel {channel}, but WAV has shape {data.shape}")
        x = data[:, idx].astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        full_scale = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
        x /= float(full_scale)
    return sr, x


def _predict_magnitude(preset: Dict[str, object], channel: str, nfft: int) -> np.ndarray:
    sr = float(preset["sr"])
    delay_samples = [int(v) for v in preset["delay_samples"]]  # type: ignore[index]

    dtype = torch.float64
    gains = torch.tensor(preset["gains"], dtype=dtype)  # type: ignore[arg-type]
    b = torch.tensor(preset["b"], dtype=dtype)  # type: ignore[arg-type]
    c_key = "cL" if channel == "L" else "cR"
    c = torch.tensor(preset[c_key], dtype=dtype)  # type: ignore[arg-type]

    matrix_type = str(preset["matrix_type"]).lower()
    if matrix_type == "householder":
        raw_u = torch.tensor(preset["u"], dtype=dtype)  # type: ignore[arg-type]
        u = unit_vector_from_raw(raw_u)
        U = householder_matrix(u)
    elif matrix_type == "hadamard":
        U = hadamard4_matrix(dtype=dtype)
    else:
        raise ValueError(f"Unsupported matrix_type: {preset['matrix_type']}")

    H = transfer_function(
        sr=sr,
        nfft=nfft,
        delay_samples=delay_samples,
        gains=gains,
        U=U,
        b=b,
        c=c,
        eps=0.0,
    )
    return np.abs(H.detach().cpu().numpy())


def _compute_errors(measured_mag: np.ndarray, predicted_mag: np.ndarray) -> Tuple[float, float]:
    abs_err = np.abs(measured_mag - predicted_mag)
    max_err = float(np.max(abs_err))
    db_err = np.abs(20.0 * np.log10(measured_mag + 1e-12) - 20.0 * np.log10(predicted_mag + 1e-12))
    max_db_err = float(np.max(db_err))
    return max_err, max_db_err


def _plot_compare(
    sr: int,
    measured_mag: np.ndarray,
    predicted_mag: np.ndarray,
    out_plot: Path,
    title: str,
) -> None:
    out_plot.parent.mkdir(parents=True, exist_ok=True)
    freqs = np.fft.rfftfreq((len(measured_mag) - 1) * 2, d=1.0 / float(sr))
    eps = 1e-12
    plt.figure(figsize=(9, 4))
    plt.plot(freqs, 20.0 * np.log10(predicted_mag + eps), label="transfer_function()")
    plt.plot(freqs, 20.0 * np.log10(measured_mag + eps), "--", label="FFT(gen_ir IR)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True, help="Path to preset JSON")
    parser.add_argument("--channel", choices=["L", "R"], default="L")
    parser.add_argument("--seconds", type=float, default=None, help="IR render seconds (default: max(8, 4*rt60))")
    parser.add_argument("--fft-nfft", type=int, default=None, help="FFT size for comparison (default: auto)")
    parser.add_argument("--max-db-target", type=float, default=1.0, help="Target for auto-refinement")
    parser.add_argument("--auto-refine", action="store_true", help="Increase FFT size until max dB error <= target")
    parser.add_argument("--gen-ir-bin", default="eval/out/bin/gen_ir")
    parser.add_argument("--gen-ir-src", default="eval/tools/gen_ir.cpp")
    parser.add_argument("--out-wav", default=None)
    parser.add_argument("--out-plot", default=None)
    args = parser.parse_args()

    preset_path = Path(args.preset)
    preset = json.loads(preset_path.read_text())
    config_id = str(preset.get("config_id", preset_path.stem))

    seconds = float(args.seconds) if args.seconds is not None else max(8.0, 4.0 * float(preset.get("rt60", 2.0)))
    out_wav = Path(args.out_wav) if args.out_wav else Path(f"eval/out/verify/verify_{config_id}_{args.channel}.wav")
    out_plot = Path(args.out_plot) if args.out_plot else Path(f"eval/figs/verify_transfer_{config_id}_{args.channel}.png")

    gen_ir_bin = Path(args.gen_ir_bin)
    gen_ir_src = Path(args.gen_ir_src)
    _build_gen_ir_if_needed(gen_ir_bin, gen_ir_src)
    _render_ir(gen_ir_bin, preset_path, out_wav, seconds)

    sr, x = _load_channel(out_wav, args.channel)
    preset_nfft = int(preset["nfft"])

    if args.fft_nfft is not None:
        nfft = int(args.fft_nfft)
    else:
        rt60 = float(preset.get("rt60", 2.0))
        target_len = max(preset_nfft, int(min(len(x), max(4096.0, sr * rt60 * 2.0))))
        nfft = max(preset_nfft, _next_pow2_floor(target_len))

    if nfft < preset_nfft:
        nfft = preset_nfft

    measured_mag = np.abs(np.fft.rfft(x, n=nfft))
    predicted_mag = _predict_magnitude(preset, args.channel, nfft=nfft)
    if measured_mag.shape != predicted_mag.shape:
        raise RuntimeError(f"Shape mismatch: measured {measured_mag.shape}, predicted {predicted_mag.shape}")

    max_err, max_db_err = _compute_errors(measured_mag, predicted_mag)

    if args.auto_refine:
        while max_db_err > float(args.max_db_target) and (nfft * 2) <= len(x):
            nfft *= 2
            measured_mag = np.abs(np.fft.rfft(x, n=nfft))
            predicted_mag = _predict_magnitude(preset, args.channel, nfft=nfft)
            max_err, max_db_err = _compute_errors(measured_mag, predicted_mag)

    _plot_compare(
        sr=sr,
        measured_mag=measured_mag,
        predicted_mag=predicted_mag,
        out_plot=out_plot,
        title=f"Transfer Verification ({config_id}, channel {args.channel}, nfft={nfft})",
    )

    print(f"Preset: {preset_path}")
    print(f"IR WAV: {out_wav}")
    print(f"Comparison FFT nfft: {nfft}")
    print(f"Max magnitude abs error: {max_err:.6e}")
    print(f"Max magnitude dB error:  {max_db_err:.6f} dB")
    print(f"Comparison plot: {out_plot}")


if __name__ == "__main__":
    main()
