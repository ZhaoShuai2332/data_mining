#!/bin/bash
set -euo pipefail

PROGRAM="fc2_mnist_infer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MP_SPDZ_DIR="${SCRIPT_DIR}/../MP-SPDZ"
ALT_MP_SPDZ_DIR="${SCRIPT_DIR}/../mp-spdz-0.4.2"
FC2_CONFIG_PATH="${FC2_CONFIG_PATH:-${SCRIPT_DIR}/config/fc2_config.json}"
PARSER_SCRIPT="${SCRIPT_DIR}/scripts/parse_fc2_run.py"

if [[ -z "${MODEL_DIR:-}" ]]; then
  echo "MODEL_DIR is not set"
  exit 1
fi
if [[ -z "${INPUT_FILE:-}" ]]; then
  echo "INPUT_FILE is not set"
  exit 1
fi

if [[ -z "${MP_SPDZ_DIR:-}" ]]; then
  if [[ -d "${DEFAULT_MP_SPDZ_DIR}" ]]; then
    MP_SPDZ_DIR="${DEFAULT_MP_SPDZ_DIR}"
  elif [[ -d "${ALT_MP_SPDZ_DIR}" ]]; then
    MP_SPDZ_DIR="${ALT_MP_SPDZ_DIR}"
  else
    echo "MP_SPDZ_DIR is not set and default paths do not exist."
    echo "Tried: ${DEFAULT_MP_SPDZ_DIR} and ${ALT_MP_SPDZ_DIR}"
    exit 1
  fi
fi

MODEL_DIR="$(cd "${MODEL_DIR}" && pwd)"
INPUT_FILE="$(cd "$(dirname "${INPUT_FILE}")" && pwd)/$(basename "${INPUT_FILE}")"
PORT="${PORT:-5000}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run_logs}"
if [[ -z "${RUN_DIR:-}" ]]; then
  TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
  RUN_DIR="${RUN_ROOT}/${TIMESTAMP}"
fi
mkdir -p "${RUN_DIR}"

COMPILE_LOG="${RUN_DIR}/compile.log"
PARTY0_LOG="${RUN_DIR}/party0.log"
PARTY1_LOG="${RUN_DIR}/party1.log"
SUMMARY_JSON="${RUN_DIR}/summary.json"
SUMMARY_CSV="${RUN_DIR}/summary.csv"

if [[ ! -f "${MODEL_DIR}/fixed_params.txt" ]]; then
  echo "Missing model file: ${MODEL_DIR}/fixed_params.txt"
  exit 1
fi
if [[ ! -f "${MODEL_DIR}/meta.json" ]]; then
  echo "Missing model metadata file: ${MODEL_DIR}/meta.json"
  exit 1
fi
if [[ ! -f "${INPUT_FILE}" ]]; then
  echo "Missing input file: ${INPUT_FILE}"
  exit 1
fi
if [[ ! -f "${FC2_CONFIG_PATH}" ]]; then
  echo "Missing FC2 config file: ${FC2_CONFIG_PATH}"
  exit 1
fi
if [[ ! -x "${PARSER_SCRIPT}" ]]; then
  echo "Missing parser script: ${PARSER_SCRIPT}"
  exit 1
fi

EXPECTED_FRACTIONAL_BITS="$(python3 - "${FC2_CONFIG_PATH}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(int(data['fractional_bits']))
PY
)"
ACTUAL_FRACTIONAL_BITS="$(python3 - "${MODEL_DIR}/meta.json" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
value = data.get('fractional_bits')
if value is None:
    raise SystemExit('meta.json missing fractional_bits')
print(int(value))
PY
)"
if [[ "${EXPECTED_FRACTIONAL_BITS}" != "${ACTUAL_FRACTIONAL_BITS}" ]]; then
  echo "Fractional bits mismatch:"
  echo "  expected from ${FC2_CONFIG_PATH}: ${EXPECTED_FRACTIONAL_BITS}"
  echo "  actual in ${MODEL_DIR}/meta.json: ${ACTUAL_FRACTIONAL_BITS}"
  echo "Please re-export parameters with matching fractional_bits."
  exit 1
fi

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  PORT=15000
  while lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; do
    PORT=$((PORT + 1))
  done
  echo "Default port 5000 is busy. Switched to PORT=${PORT}"
fi

prepare_inputs_dir_layout() {
  rm -rf Player-Data/Input-P0-0 Player-Data/Input-P1-0
  mkdir -p Player-Data/Input-P0-0 Player-Data/Input-P1-0
  cp "${MODEL_DIR}/fixed_params.txt" "Player-Data/Input-P0-0/${PROGRAM}-P0-0"
  cp "${INPUT_FILE}" "Player-Data/Input-P1-0/${PROGRAM}-P1-0"
}

prepare_inputs_flat_layout() {
  rm -rf Player-Data/Input-P0-0 Player-Data/Input-P1-0
  cp "${MODEL_DIR}/fixed_params.txt" "Player-Data/Input-P0-0"
  cp "${INPUT_FILE}" "Player-Data/Input-P1-0"
}

run_protocol_once() {
  local log0="$1"
  local log1="$2"
  local rc0 rc1
  ./semi2k-party.x -pn "${PORT}" 0 "${PROGRAM}" >"${log0}" 2>&1 &
  P0=$!
  ./semi2k-party.x -pn "${PORT}" 1 "${PROGRAM}" >"${log1}" 2>&1 &
  P1=$!
  set +e
  wait "${P0}"; rc0=$?
  wait "${P1}"; rc1=$?
  set -e
  if [[ ${rc0} -ne 0 || ${rc1} -ne 0 ]]; then
    return 1
  fi
}

cd "${MP_SPDZ_DIR}"
mkdir -p Programs/Source
cp -f "${SCRIPT_DIR}/Programs/Source/${PROGRAM}.mpc" "Programs/Source/${PROGRAM}.mpc"

if [[ ! -x "./semi2k-party.x" ]]; then
  echo "semi2k-party.x not found in ${MP_SPDZ_DIR}"
  echo "Build it first, e.g. make semi2k-party.x USE_KOS=1 MY_CFLAGS='-Wno-error=unused-parameter'"
  exit 1
fi

echo "=== FC2 MPC Run ===" | tee "${RUN_DIR}/run.info"
echo "run_dir=${RUN_DIR}" | tee -a "${RUN_DIR}/run.info"
echo "model_dir=${MODEL_DIR}" | tee -a "${RUN_DIR}/run.info"
echo "input_file=${INPUT_FILE}" | tee -a "${RUN_DIR}/run.info"
echo "port=${PORT}" | tee -a "${RUN_DIR}/run.info"
echo "fractional_bits=${EXPECTED_FRACTIONAL_BITS}" | tee -a "${RUN_DIR}/run.info"

./compile.py --ring 64 "${PROGRAM}" 2>&1 | tee "${COMPILE_LOG}"

prepare_inputs_dir_layout
TMP_P0_FIRST="$(mktemp)"
TMP_P1_FIRST="$(mktemp)"
if ! run_protocol_once "${TMP_P0_FIRST}" "${TMP_P1_FIRST}"; then
  cat "${TMP_P0_FIRST}" > "${PARTY0_LOG}"
  cat "${TMP_P1_FIRST}" > "${PARTY1_LOG}"
  if grep -Eq "not enough inputs in Player-Data/Input-P0-0|not enough inputs in Player-Data/Input-P1-0" "${PARTY0_LOG}" "${PARTY1_LOG}"; then
    echo "Detected flat input layout in this MP-SPDZ build, retrying with flat files."
    prepare_inputs_flat_layout
    TMP_P0_SECOND="$(mktemp)"
    TMP_P1_SECOND="$(mktemp)"
    run_protocol_once "${TMP_P0_SECOND}" "${TMP_P1_SECOND}"
    {
      echo "=== First attempt (directory layout) ==="
      cat "${TMP_P0_FIRST}"
      echo
      echo "=== Second attempt (flat-file layout) ==="
      cat "${TMP_P0_SECOND}"
    } > "${PARTY0_LOG}"
    {
      echo "=== First attempt (directory layout) ==="
      cat "${TMP_P1_FIRST}"
      echo
      echo "=== Second attempt (flat-file layout) ==="
      cat "${TMP_P1_SECOND}"
    } > "${PARTY1_LOG}"
  else
    exit 1
  fi
else
  cat "${TMP_P0_FIRST}" > "${PARTY0_LOG}"
  cat "${TMP_P1_FIRST}" > "${PARTY1_LOG}"
fi

PARSE_ARGS=(
  --compile-log "${COMPILE_LOG}"
  --party0-log "${PARTY0_LOG}"
  --party1-log "${PARTY1_LOG}"
  --summary-json "${SUMMARY_JSON}"
  --summary-csv "${SUMMARY_CSV}"
  --program "${PROGRAM}"
  --run-dir "${RUN_DIR}"
  --fractional-bits "${EXPECTED_FRACTIONAL_BITS}"
)
if [[ -n "${SAMPLE_INDEX:-}" ]]; then
  PARSE_ARGS+=(--sample-index "${SAMPLE_INDEX}")
fi
if [[ -n "${TRUE_LABEL:-}" ]]; then
  PARSE_ARGS+=(--true-label "${TRUE_LABEL}")
fi

python3 "${PARSER_SCRIPT}" "${PARSE_ARGS[@]}"

python3 - "${SUMMARY_JSON}" <<'PY'
import json
import sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print("=== Run Summary ===")
print(f"summary_json: {path}")
print(f"summary_csv: {path.rsplit('/', 1)[0] + '/summary.csv'}")
print(f"predicted_label: {data.get('predicted_label', 'N/A')}")
print(f"elapsed_time_seconds: {data.get('elapsed_time_seconds', 'N/A')}")
print(f"party0_sent_mb: {data.get('party0_sent_mb', 'N/A')}")
print(f"party1_sent_mb: {data.get('party1_sent_mb', 'N/A')}")
print(f"total_sent_mb: {data.get('total_sent_mb', 'N/A')}")
print(f"rounds: {data.get('rounds', 'N/A')}")
print(f"triples: {data.get('triples', 'N/A')}")
PY
