#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_ID="${1:-householder_prime_rt60_2p8_paper}"
SEEDS="${2:-0,1,2,3,4}"

python3 eval/scripts/run_multiseed_fixed_vs_diff.py \
  --config-id "${CONFIG_ID}" \
  --seeds "${SEEDS}" \
  --scope all

echo "multiseed artifacts:"
echo "  ${ROOT_DIR}/eval/figs/multiseed/${CONFIG_ID}/aggregate_summary.csv"
echo "  ${ROOT_DIR}/eval/figs/multiseed/${CONFIG_ID}/aggregate_stats.json"
echo "  ${ROOT_DIR}/eval/figs/multiseed/${CONFIG_ID}/paper/multiseed_metrics_errorbars.png"
echo "  ${ROOT_DIR}/eval/figs/multiseed/${CONFIG_ID}/paper/multiseed_diffusion_meanstd.png"
echo "  ${ROOT_DIR}/eval/figs/multiseed/${CONFIG_ID}/paper/multiseed_echo_density_meanstd.png"
