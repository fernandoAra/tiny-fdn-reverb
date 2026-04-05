#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  bash eval/scripts/build_demo_bundle.sh --preset <preset.json> [options]

Required:
  --preset <path>             Canonical learned preset JSON (repo-relative or absolute)

Options:
  --sync-plugin-header        Regenerate plugins/tiny-fdn-reverb/DiffFdnPresets.hpp from --preset
  --skip-optimize             Skip optional optimize step if preset is missing
  --config-id <string>        Config id used only when preset is missing and optimize is allowed
  --demo-stimulus <kind>      pink_noise | click_train | input_wav (default: pink_noise)
  --demo-input-wav <path>     Input wav path when --demo-stimulus=input_wav
  --scope <mode>              fixed_full | all (default: all)

Canonical example:
  bash eval/scripts/build_demo_bundle.sh \
    --preset eval/out/presets/householder_prime_rt60_2p8_paper_seed0.json \
    --sync-plugin-header
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PRESET_PATH=""
SYNC_PLUGIN_HEADER=0
SKIP_OPTIMIZE=0
CONFIG_ID=""
DEMO_STIMULUS="pink_noise"
DEMO_INPUT_WAV=""
SCOPE="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset)
      PRESET_PATH="${2:-}"
      shift 2
      ;;
    --sync-plugin-header)
      SYNC_PLUGIN_HEADER=1
      shift
      ;;
    --skip-optimize)
      SKIP_OPTIMIZE=1
      shift
      ;;
    --config-id)
      CONFIG_ID="${2:-}"
      shift 2
      ;;
    --demo-stimulus)
      DEMO_STIMULUS="${2:-}"
      shift 2
      ;;
    --demo-input-wav)
      DEMO_INPUT_WAV="${2:-}"
      shift 2
      ;;
    --scope)
      SCOPE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PRESET_PATH}" ]]; then
  echo "Missing required argument: --preset" >&2
  usage
  exit 1
fi

if [[ "${SCOPE}" != "all" && "${SCOPE}" != "fixed_full" ]]; then
  echo "Invalid --scope '${SCOPE}', expected one of: fixed_full, all" >&2
  exit 1
fi

if [[ "${DEMO_STIMULUS}" != "pink_noise" && "${DEMO_STIMULUS}" != "click_train" && "${DEMO_STIMULUS}" != "input_wav" ]]; then
  echo "Invalid --demo-stimulus '${DEMO_STIMULUS}'" >&2
  exit 1
fi
if [[ "${DEMO_STIMULUS}" == "input_wav" && -z "${DEMO_INPUT_WAV}" ]]; then
  echo "--demo-input-wav is required when --demo-stimulus=input_wav" >&2
  exit 1
fi

if [[ "${PRESET_PATH}" = /* ]]; then
  PRESET_ABS="${PRESET_PATH}"
else
  PRESET_ABS="${ROOT_DIR}/${PRESET_PATH}"
fi

if [[ ! -f "${PRESET_ABS}" ]]; then
  if [[ "${SKIP_OPTIMIZE}" -eq 1 ]]; then
    echo "Preset not found and --skip-optimize set: ${PRESET_ABS}" >&2
    exit 1
  fi
  if [[ -z "${CONFIG_ID}" ]]; then
    echo "Preset not found: ${PRESET_ABS}" >&2
    echo "Provide --config-id to allow auto-generate, or pass --skip-optimize and an existing preset." >&2
    exit 1
  fi
  echo "[Info] Preset missing; generating with optimize_householder.py for config_id=${CONFIG_ID}"
  HISTORY_PATH="${PRESET_ABS%.json}.history.json"
  python eval/difffdn/optimize_householder.py \
    --config-id "${CONFIG_ID}" \
    --out-json "${PRESET_ABS}" \
    --history-json "${HISTORY_PATH}"
fi

if [[ ! -f "${PRESET_ABS}" ]]; then
  echo "Preset still not found after optional optimize step: ${PRESET_ABS}" >&2
  exit 1
fi

CANONICAL_CONFIG_ID="$(python -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("config_id","")).strip())' "${PRESET_ABS}")"
if [[ -z "${CANONICAL_CONFIG_ID}" ]]; then
  CANONICAL_CONFIG_ID="$(basename "${PRESET_ABS}" .json)"
fi

PLUGIN_HEADER_PATH="${ROOT_DIR}/plugins/tiny-fdn-reverb/DiffFdnPresets.hpp"
PLUGIN_HEADER_SYNCED=false

if [[ "${SYNC_PLUGIN_HEADER}" -eq 1 ]]; then
  HEADER_PRESETS=("${PRESET_ABS}")
  # Keep the plugin bank aligned with the currently selected tuned variants:
  # 44.1 kHz Prime -> full learned IO
  # 48 kHz Prime  -> u-only (learned u, fixed IO)
  # 48 kHz Spread -> full learned IO
  PRIME_44100_PRESET="${ROOT_DIR}/eval/figs/multiseed/householder_prime_sr44100_bank/seed0/render_presets/householder_prime_sr44100_bank_seed0_r2_full.json"
  PRIME_48000_PRESET="${ROOT_DIR}/eval/figs/multiseed/householder_prime_rt60_2p8_paper/seed2/render_presets/householder_prime_rt60_2p8_paper_seed2_u_only.json"
  SPREAD_48000_PRESET="${ROOT_DIR}/eval/figs/multiseed/householder_spread_rt60_2p8_bank/seed2/render_presets/householder_spread_rt60_2p8_bank_seed2_r2_full.json"

  if [[ -f "${PRIME_44100_PRESET}" && "${PRIME_44100_PRESET}" != "${PRESET_ABS}" ]]; then
    HEADER_PRESETS+=("${PRIME_44100_PRESET}")
  fi
  if [[ -f "${PRIME_48000_PRESET}" && "${PRIME_48000_PRESET}" != "${PRESET_ABS}" ]]; then
    HEADER_PRESETS+=("${PRIME_48000_PRESET}")
  fi
  if [[ -f "${SPREAD_48000_PRESET}" && "${SPREAD_48000_PRESET}" != "${PRESET_ABS}" ]]; then
    HEADER_PRESETS+=("${SPREAD_48000_PRESET}")
  fi

  EXPORT_CMD=(python eval/difffdn/export_cpp_header.py --out-header "${PLUGIN_HEADER_PATH}")
  for preset_path in "${HEADER_PRESETS[@]}"; do
    EXPORT_CMD+=(--preset "${preset_path}")
  done

  "${EXPORT_CMD[@]}"
  if ! grep -Fq "${CANONICAL_CONFIG_ID}" "${PLUGIN_HEADER_PATH}"; then
    echo "Synced plugin header does not contain canonical config_id '${CANONICAL_CONFIG_ID}'" >&2
    exit 1
  fi
  PLUGIN_HEADER_SYNCED=true
  echo "[SYNC] Embedded canonical preset in plugin header: ${CANONICAL_CONFIG_ID}"
  echo "[SYNC] Header path: ${PLUGIN_HEADER_PATH}"
fi

COMPARE_SCOPE="all"
if [[ "${SCOPE}" == "fixed_full" ]]; then
  COMPARE_SCOPE="full"
fi

COMPARE_CMD=(
  python eval/scripts/compare_fixed_vs_diff.py
  --preset "${PRESET_ABS}"
  --scope "${COMPARE_SCOPE}"
  --sanity-check
  --export-demo-wavs
  --level-match-method lufs
  --level-match-target fixed
  --require-lufs
  --demo-stimulus "${DEMO_STIMULUS}"
)
if [[ "${DEMO_STIMULUS}" == "input_wav" ]]; then
  COMPARE_CMD+=(--demo-input-wav "${DEMO_INPUT_WAV}")
fi

echo "[RUN] ${COMPARE_CMD[*]}"
"${COMPARE_CMD[@]}"

RUN_JSON="${ROOT_DIR}/eval/figs/fixed_vs_diff_run.json"
if [[ ! -f "${RUN_JSON}" ]]; then
  echo "Missing required compare output: ${RUN_JSON}" >&2
  exit 1
fi

SLUG="$(python -c 'import json,re,sys; p=json.load(open(sys.argv[1])); s=str(p.get("base_config_id") or p.get("config_id") or ""); t=s.lower(); t=re.sub(r"[^a-z0-9]+","_",t); t=t.strip("_"); print(t)' "${RUN_JSON}")"
if [[ -z "${SLUG}" ]]; then
  echo "Could not derive preset slug from ${RUN_JSON}" >&2
  exit 1
fi

DEMO_ROOT="${ROOT_DIR}/eval/out/demo/${SLUG}"
DEMO_MANIFEST="${DEMO_ROOT}/demo_manifest.json"
if [[ ! -f "${DEMO_MANIFEST}" ]]; then
  echo "Missing required demo manifest: ${DEMO_MANIFEST}" >&2
  exit 1
fi

BUNDLE_DIR="${ROOT_DIR}/eval/out/demo_bundle/${SLUG}"
rm -rf "${BUNDLE_DIR}"
mkdir -p "${BUNDLE_DIR}"

copy_required() {
  local src="$1"
  local dst="$2"
  if [[ ! -f "${src}" ]]; then
    echo "Missing required artifact: ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dst}"
}

copy_required "${ROOT_DIR}/eval/figs/fixed_vs_diff_run.json" "${BUNDLE_DIR}/fixed_vs_diff_run.json"
copy_required "${ROOT_DIR}/eval/figs/fixed_vs_diff_mag_overlay.png" "${BUNDLE_DIR}/fixed_vs_diff_mag_overlay.png"
copy_required "${ROOT_DIR}/eval/figs/fixed_vs_diff_irfft_overlay.png" "${BUNDLE_DIR}/fixed_vs_diff_irfft_overlay.png"
copy_required "${ROOT_DIR}/eval/figs/fixed_vs_diff_diffusion.png" "${BUNDLE_DIR}/fixed_vs_diff_diffusion.png"
copy_required "${ROOT_DIR}/eval/figs/fixed_vs_diff_metrics_bars.png" "${BUNDLE_DIR}/fixed_vs_diff_metrics_bars.png"
copy_required "${ROOT_DIR}/eval/figs/summary_fixed_vs_diff.csv" "${BUNDLE_DIR}/summary_fixed_vs_diff.csv"
copy_required "${DEMO_MANIFEST}" "${BUNDLE_DIR}/demo_manifest.json"

DEMO_WAV_COUNT=0
while IFS= read -r wav_path; do
  cp "${wav_path}" "${BUNDLE_DIR}/"
  DEMO_WAV_COUNT=$((DEMO_WAV_COUNT + 1))
done < <(find "${DEMO_ROOT}" -maxdepth 1 -type f -name "demo_${SLUG}_*.wav" | sort)

if [[ "${DEMO_WAV_COUNT}" -eq 0 ]]; then
  echo "No demo WAVs found in ${DEMO_ROOT}" >&2
  exit 1
fi

python - <<'PY' "${RUN_JSON}" "${DEMO_MANIFEST}" "${BUNDLE_DIR}" "${PRESET_ABS}" "${CANONICAL_CONFIG_ID}" "${PLUGIN_HEADER_SYNCED}" "${PLUGIN_HEADER_PATH}" "${SCOPE}" "${COMPARE_CMD[*]}"
import datetime as dt
import json
import pathlib
import subprocess
import sys

run_json = pathlib.Path(sys.argv[1])
demo_manifest = pathlib.Path(sys.argv[2])
bundle_dir = pathlib.Path(sys.argv[3])
preset_abs = pathlib.Path(sys.argv[4])
config_id = sys.argv[5]
plugin_synced = sys.argv[6].lower() == "true"
plugin_header_path = sys.argv[7]
scope = sys.argv[8]
compare_cmd = sys.argv[9]

run_payload = json.loads(run_json.read_text())
demo_payload = json.loads(demo_manifest.read_text())

git_commit = "unknown"
try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(bundle_dir.parents[3]),
        text=True,
    ).strip()
except Exception:
    pass

bundle_manifest = {
    "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "git_commit": git_commit,
    "canonical_preset_path": str(preset_abs),
    "canonical_config_id": config_id,
    "compare_repro_command": compare_cmd,
    "plugin_header_synced": plugin_synced,
    "plugin_header_path": plugin_header_path,
    "bundle_files": [],
    "level_match_method_requested": demo_payload.get("level_match", {}).get("method_requested", "unknown"),
    "level_match_method_used": demo_payload.get("level_match", {}).get("method_used", "unknown"),
    "level_match_target_policy": demo_payload.get("level_match", {}).get("target_policy", "unknown"),
    "scope": scope,
    "sample_rate": demo_payload.get("sample_rate", 0),
    "note": (
        "Defense bundle: matched demo WAVs are the primary fairness-controlled listening evidence; "
        "live plugin switching is secondary realtime proof."
    ),
}

manifest_path = bundle_dir / "bundle_manifest.json"
manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n")

bundle_manifest["bundle_files"] = sorted(
    [p.relative_to(bundle_dir).as_posix() for p in bundle_dir.rglob("*") if p.is_file()]
)
manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n")
PY

REQUIRED_BUNDLE=(
  "fixed_vs_diff_run.json"
  "demo_manifest.json"
  "fixed_vs_diff_mag_overlay.png"
  "fixed_vs_diff_irfft_overlay.png"
  "fixed_vs_diff_diffusion.png"
  "fixed_vs_diff_metrics_bars.png"
  "summary_fixed_vs_diff.csv"
  "bundle_manifest.json"
)
for file_name in "${REQUIRED_BUNDLE[@]}"; do
  if [[ ! -f "${BUNDLE_DIR}/${file_name}" ]]; then
    echo "Bundle missing required file: ${BUNDLE_DIR}/${file_name}" >&2
    exit 1
  fi
done

echo "[BUNDLE] Created: ${BUNDLE_DIR}"
echo "[BUNDLE] Canonical config_id: ${CANONICAL_CONFIG_ID}"
echo "[BUNDLE] Scope: ${SCOPE} (compare scope=${COMPARE_SCOPE})"
echo "[BUNDLE] Level match requested=LUFS, require_lufs=true"
echo "[BUNDLE] Artifacts:"
echo "  - ${BUNDLE_DIR}/bundle_manifest.json"
echo "  - ${BUNDLE_DIR}/fixed_vs_diff_run.json"
echo "  - ${BUNDLE_DIR}/demo_manifest.json"
echo "  - ${BUNDLE_DIR}/summary_fixed_vs_diff.csv"
echo "  - demo wav count: ${DEMO_WAV_COUNT}"
