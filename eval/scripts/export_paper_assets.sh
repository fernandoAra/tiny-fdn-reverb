#!/usr/bin/env bash
set -euo pipefail

# Expected upstream regeneration flow (repo-relative):
#   python3 eval/difffdn/optimize_householder.py --fs 48000 --rt60 2.8 ...
#   python3 eval/scripts/compare_fixed_vs_diff.py --preset eval/out/presets/<preset>.json --scope all --sanity-check
#   bash eval/scripts/export_paper_assets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FIG_DIR="${ROOT_DIR}/eval/figs"
PAPER_DIR="${ROOT_DIR}/paper_assets"
PAPER_FIG_DIR="${PAPER_DIR}/figures"
PAPER_TAB_DIR="${PAPER_DIR}/tables"
SELECTED_SLUG="${1:-}"

mkdir -p "${PAPER_FIG_DIR}" "${PAPER_TAB_DIR}"

copy_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Missing required asset: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dst}"
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

echo "Exported paper assets:"
echo "  ${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff.png"
echo "  ${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff.png"
echo "  ${PAPER_TAB_DIR}/table_fixed_vs_diff.csv"
if [[ -n "${SELECTED_SLUG}" ]]; then
  echo "  ${PAPER_FIG_DIR}/fig_mag_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_irfft_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_diffusion_fixed_vs_diff_${SELECTED_SLUG}.png"
  echo "  ${PAPER_FIG_DIR}/fig_metrics_fixed_vs_diff_${SELECTED_SLUG}.png"
fi
