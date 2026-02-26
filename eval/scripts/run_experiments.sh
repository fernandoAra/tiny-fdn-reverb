#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${ROOT_DIR}/eval/out"
PRESET_DIR="${OUT_DIR}/presets"
IR_DIR="${OUT_DIR}/ir"
BIN_DIR="${OUT_DIR}/bin"
FIG_DIR="${ROOT_DIR}/eval/figs"
VERIFY_DIR="${OUT_DIR}/verify"

ALPHA_DENSITY="0.05"
RT60="2.8"
NFFT="2048"
SR="48000"
SEED="1234"
DELAY_PRIME="1499,2377,3217,4421"
DELAY_SPREAD="1200,1800,2400,3000"
PRESET_IDS=(
  "householder_prime_rt60_2p8_ad005"
  "hadamard_prime_rt60_2p8_ad005"
  "householder_spread_rt60_2p8_ad005"
  "hadamard_spread_rt60_2p8_ad005"
)

check_python_module() {
  local module="$1"
  if ! python3 -c "import ${module}" >/dev/null 2>&1; then
    echo "Missing python module: ${module}"
    echo "Install dependencies first, e.g.: python3 -m pip install torch numpy scipy matplotlib"
    exit 1
  fi
}

check_python_module torch
check_python_module numpy
check_python_module scipy
check_python_module matplotlib

mkdir -p "${PRESET_DIR}" "${IR_DIR}" "${BIN_DIR}" "${FIG_DIR}" "${VERIFY_DIR}"

echo "[1/6] Optimize presets (Prime + Spread, alpha_density=${ALPHA_DENSITY})"
optimize_preset() {
  local config_id="$1"
  local matrix_type="$2"
  local delay_samples="$3"
  local steps="$4"
  local learn_io="$5"
  local extra_args=()
  if [[ "${learn_io}" == "1" ]]; then
    extra_args+=(--learn-io)
  fi
  python3 "${ROOT_DIR}/eval/difffdn/optimize_householder.py" \
    --config-id "${config_id}" \
    --matrix-type "${matrix_type}" \
    --sr "${SR}" --nfft "${NFFT}" \
    --delay-samples "${delay_samples}" \
    --rt60 "${RT60}" \
    --steps "${steps}" \
    --lr 0.03 \
    --alpha-density "${ALPHA_DENSITY}" \
    --seed "${SEED}" \
    "${extra_args[@]}" \
    --out-dir "${PRESET_DIR}"
}

optimize_preset "${PRESET_IDS[0]}" "householder" "${DELAY_PRIME}" "800" "1"
optimize_preset "${PRESET_IDS[1]}" "hadamard" "${DELAY_PRIME}" "1" "0"
optimize_preset "${PRESET_IDS[2]}" "householder" "${DELAY_SPREAD}" "800" "1"
optimize_preset "${PRESET_IDS[3]}" "hadamard" "${DELAY_SPREAD}" "1" "0"

echo "[2/6] Build offline IR renderer"
c++ -std=c++17 -O2 -Wall -Wextra -pedantic \
  "${ROOT_DIR}/eval/tools/gen_ir.cpp" \
  -o "${BIN_DIR}/gen_ir"

echo "[3/6] Render IR WAVs from presets"
for preset_name in "${PRESET_IDS[@]}"; do
  preset_path="${PRESET_DIR}/${preset_name}.json"
  "${BIN_DIR}/gen_ir" "${preset_path}" "${IR_DIR}/IR_${preset_name}.wav" 4.0
done

echo "[4/6] Analyze IR batch and generate figures"
python3 "${ROOT_DIR}/eval/scripts/analyze_ir_batch.py" \
  --ir-dir "${IR_DIR}" \
  --fig-dir "${FIG_DIR}"

echo "[5/6] Verify transfer_function() vs FFT(gen_ir IR)"
python3 "${ROOT_DIR}/eval/scripts/verify_transfer_match.py" \
  --preset "${PRESET_DIR}/${PRESET_IDS[0]}.json" \
  --channel L \
  --gen-ir-bin "${BIN_DIR}/gen_ir" \
  --out-wav "${VERIFY_DIR}/verify_householder_prime_L.wav" \
  --out-plot "${FIG_DIR}/verify_transfer_householder_prime_L.png"

echo "[6/6] Compare Fixed vs Diff (u-only/full) for paper figures"
python3 "${ROOT_DIR}/eval/scripts/compare_fixed_vs_diff.py" \
  --preset "${PRESET_DIR}/${PRESET_IDS[0]}.json" \
  --scope all \
  --channel L \
  --nfft "${NFFT}" \
  --fig-dir "${FIG_DIR}"

echo "Done."
echo "Summary CSV: ${FIG_DIR}/summary.csv"
echo "Figures: ${FIG_DIR}/edc_overlay.png, ${FIG_DIR}/rt60_edt_bars.png, ${FIG_DIR}/ringiness_bars.png"
echo "Verification plot: ${FIG_DIR}/verify_transfer_householder_prime_L.png"
echo "Fixed-vs-Diff figures: ${FIG_DIR}/fixed_vs_diff_mag_overlay.png, ${FIG_DIR}/fixed_vs_diff_irfft_overlay.png, ${FIG_DIR}/fixed_vs_diff_diffusion.png, ${FIG_DIR}/fixed_vs_diff_metrics_bars.png"
