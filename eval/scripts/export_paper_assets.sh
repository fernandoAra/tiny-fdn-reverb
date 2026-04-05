#!/usr/bin/env bash
set -euo pipefail

# Expected upstream regeneration flow (repo-relative):
#   python3 eval/difffdn/optimize_householder.py --fs 48000 --rt60 2.8 ...
#   python3 eval/scripts/compare_fixed_vs_diff.py --preset eval/out/presets/<preset>.json --scope all --sanity-check
#   bash eval/scripts/export_paper_assets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
FIG_DIR="${ROOT_DIR}/eval/figs"
PAPER_DIR="${ROOT_DIR}/paper_assets"
PAPER_FIG_DIR="${PAPER_DIR}/figures"
PAPER_TAB_DIR="${PAPER_DIR}/tables"
PAPER_FIG_MS_DIR="${PAPER_DIR}/figures_multiseed"
PAPER_TAB_MS_DIR="${PAPER_DIR}/tables_multiseed"
SELECTED_SLUG="${1:-}"
MULTISEED_CONFIG="${2:-}"

mkdir -p "${PAPER_FIG_DIR}" "${PAPER_TAB_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/theoretical_plots/run_notebook_figures.py"

copy_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Missing required asset: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dst}"
}

copy_optional() {
  local src="$1"
  local dst="$2"
  if [[ -f "${src}" ]]; then
    cp "${src}" "${dst}"
  fi
}

select_representative_seed() {
  local multi_dir="$1"
  "${PYTHON_BIN}" - "${multi_dir}" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
summary_path = run_root / "aggregate_summary.csv"
stats_path = run_root / "aggregate_stats.json"
rows = list(csv.DictReader(summary_path.open("r", newline="")))
stats = json.loads(stats_path.read_text())

metrics = (
    "spectral_dev_db_50_12k",
    "rt60_s",
    "edt_s",
    "ringiness",
    "kurtosis_mean_50_300ms",
    "echo_density_events_per_s_50_300ms",
)
seed_scores = {}
seed_ids = sorted({int(row["seed"]) for row in rows if row.get("seed", "").strip()})
for seed in seed_ids:
    total = 0.0
    count = 0
    for scope in ("u_only", "full"):
        row = next((r for r in rows if int(r.get("seed", "-1")) == seed and r.get("scope", "").strip() == scope), None)
        if row is None:
            continue
        for metric in metrics:
            value = float(row[metric])
            mean = float(stats["metrics"][scope][metric]["mean"])
            std = float(stats["metrics"][scope][metric]["std"])
            if not (math.isfinite(value) and math.isfinite(mean)):
                continue
            scale = std if math.isfinite(std) and std > 1e-9 else 1.0
            total += ((value - mean) / scale) ** 2
            count += 1
    if count > 0:
        seed_scores[seed] = total / count

if not seed_scores:
    raise SystemExit("Unable to choose representative seed from existing multiseed outputs.")
print(min(seed_scores, key=seed_scores.get))
PY
}

copy_required "${FIG_DIR}/fixed_vs_diff_mag_overlay.png" "${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff.png"
copy_required "${FIG_DIR}/fixed_vs_diff_irfft_overlay.png" "${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff.png"
copy_required "${FIG_DIR}/fixed_vs_diff_diffusion.png" "${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff.png"
copy_required "${FIG_DIR}/fixed_vs_diff_metrics_bars.png" "${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff.png"
copy_required "${FIG_DIR}/fixed_vs_diff_summary_table.csv" "${PAPER_TAB_DIR}/table_fixed_vs_diff.csv"

if [[ -z "${SELECTED_SLUG}" ]]; then
  FIRST_SLUGGED_MAG="$(find "${FIG_DIR}" -maxdepth 1 -type f -name 'fixed_vs_diff_mag_overlay_*.png' | sort | head -n 1 || true)"
  if [[ -n "${FIRST_SLUGGED_MAG}" ]]; then
    BASENAME="$(basename "${FIRST_SLUGGED_MAG}")"
    SELECTED_SLUG="${BASENAME#fixed_vs_diff_mag_overlay_}"
    SELECTED_SLUG="${SELECTED_SLUG%.png}"
  fi
fi

if [[ -n "${SELECTED_SLUG}" ]]; then
  copy_required "${FIG_DIR}/fixed_vs_diff_mag_overlay_${SELECTED_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff_${SELECTED_SLUG}.png"
  copy_required "${FIG_DIR}/fixed_vs_diff_irfft_overlay_${SELECTED_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff_${SELECTED_SLUG}.png"
  copy_required "${FIG_DIR}/fixed_vs_diff_diffusion_${SELECTED_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff_${SELECTED_SLUG}.png"
  copy_required "${FIG_DIR}/fixed_vs_diff_metrics_bars_${SELECTED_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff_${SELECTED_SLUG}.png"
fi

if [[ -n "${MULTISEED_CONFIG}" ]]; then
  MULTI_DIR="${FIG_DIR}/multiseed/${MULTISEED_CONFIG}"
  REPRESENTATIVE_SEED="$(select_representative_seed "${MULTI_DIR}")"
  REPRESENTATIVE_SLUG="${MULTISEED_CONFIG}_seed${REPRESENTATIVE_SEED}"
  copy_required "${MULTI_DIR}/seed${REPRESENTATIVE_SEED}/fixed_vs_diff_edc_overlay_${REPRESENTATIVE_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff.png"
  copy_required "${MULTI_DIR}/seed${REPRESENTATIVE_SEED}/fixed_vs_diff_edc_overlay_${REPRESENTATIVE_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff_${REPRESENTATIVE_SLUG}.png"
  copy_required "${MULTI_DIR}/seed${REPRESENTATIVE_SEED}/fixed_vs_diff_edc_overlay_${REPRESENTATIVE_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_edc_representative_${MULTISEED_CONFIG}.png"
  copy_required "${MULTI_DIR}/seed${REPRESENTATIVE_SEED}/fixed_vs_diff_irfft_overlay_${REPRESENTATIVE_SLUG}.png" \
    "${PAPER_FIG_DIR}/fig_irfft_representative_${MULTISEED_CONFIG}.png"
else
  copy_required "${FIG_DIR}/fixed_vs_diff_edc_overlay.png" "${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff.png"
  if [[ -n "${SELECTED_SLUG}" ]]; then
    copy_required "${FIG_DIR}/fixed_vs_diff_edc_overlay_${SELECTED_SLUG}.png" \
      "${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff_${SELECTED_SLUG}.png"
  fi
fi

if [[ -n "${MULTISEED_CONFIG}" ]]; then
  MULTI_DIR="${FIG_DIR}/multiseed/${MULTISEED_CONFIG}"
  MULTI_PAPER_DIR="${MULTI_DIR}/paper"
  mkdir -p "${PAPER_FIG_MS_DIR}" "${PAPER_TAB_MS_DIR}"
  copy_required "${MULTI_PAPER_DIR}/multiseed_metrics_errorbars.png" \
    "${PAPER_FIG_MS_DIR}/fig_multiseed_metrics_${MULTISEED_CONFIG}.png"
  copy_required "${MULTI_PAPER_DIR}/multiseed_deltas_errorbars.png" \
    "${PAPER_FIG_MS_DIR}/fig_multiseed_deltas_${MULTISEED_CONFIG}.png"
  copy_required "${MULTI_PAPER_DIR}/multiseed_diffusion_meanstd.png" \
    "${PAPER_FIG_MS_DIR}/fig_multiseed_diffusion_${MULTISEED_CONFIG}.png"
  copy_required "${MULTI_PAPER_DIR}/multiseed_echo_density_meanstd.png" \
    "${PAPER_FIG_MS_DIR}/fig_multiseed_echo_density_${MULTISEED_CONFIG}.png"
  copy_required "${MULTI_DIR}/aggregate_summary.csv" \
    "${PAPER_TAB_MS_DIR}/table_multiseed_${MULTISEED_CONFIG}.csv"
  copy_required "${MULTI_DIR}/aggregate_deltas.csv" \
    "${PAPER_TAB_MS_DIR}/deltas_multiseed_${MULTISEED_CONFIG}.csv"
  copy_optional "${MULTI_DIR}/wins_table.csv" \
    "${PAPER_TAB_MS_DIR}/wins_multiseed_${MULTISEED_CONFIG}.csv"
  copy_required "${MULTI_DIR}/aggregate_stats.json" \
    "${PAPER_TAB_MS_DIR}/stats_multiseed_${MULTISEED_CONFIG}.json"
fi

echo "Exported paper assets:"
echo "  ${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig1_feedback_comb_mag.png"
echo "  ${PAPER_FIG_DIR}/fig2_allpass_impulse_comparison.png"
echo "  ${PAPER_FIG_DIR}/waveform_to_delayed_samples.png"
echo "  ${PAPER_TAB_DIR}/table_fixed_vs_diff.csv"
if [[ -n "${SELECTED_SLUG}" ]]; then
  echo "  ${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff_${SELECTED_SLUG}.png"
fi
if [[ -n "${MULTISEED_CONFIG}" ]]; then
  echo "  ${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff_${REPRESENTATIVE_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_edc_representative_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_irfft_representative_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_FIG_MS_DIR}/fig_multiseed_metrics_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_FIG_MS_DIR}/fig_multiseed_deltas_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_FIG_MS_DIR}/fig_multiseed_diffusion_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_FIG_MS_DIR}/fig_multiseed_echo_density_${MULTISEED_CONFIG}.png"
  echo "  ${PAPER_TAB_MS_DIR}/table_multiseed_${MULTISEED_CONFIG}.csv"
  echo "  ${PAPER_TAB_MS_DIR}/deltas_multiseed_${MULTISEED_CONFIG}.csv"
  echo "  ${PAPER_TAB_MS_DIR}/stats_multiseed_${MULTISEED_CONFIG}.json"
  if [[ -f "${PAPER_TAB_MS_DIR}/wins_multiseed_${MULTISEED_CONFIG}.csv" ]]; then
    echo "  ${PAPER_TAB_MS_DIR}/wins_multiseed_${MULTISEED_CONFIG}.csv"
  fi
else
  if [[ -n "${SELECTED_SLUG}" ]]; then
    echo "  ${PAPER_FIG_DIR}/fig_edc_fixed_vs_diff_${SELECTED_SLUG}.png"
  fi
fi
