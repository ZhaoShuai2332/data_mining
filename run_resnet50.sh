#!/bin/bash
set -euo pipefail

PROGRAM="resnet50_mnist_infer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MP_SPDZ_DIR="${SCRIPT_DIR}/../MP-SPDZ"
ALT_MP_SPDZ_DIR="${SCRIPT_DIR}/../mp-spdz-0.4.2"
PARSER_SCRIPT="${SCRIPT_DIR}/scripts/parse_resnet50_run.py"

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
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/run_logs}"
if [[ -z "${RUN_DIR:-}" ]]; then
  TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
  RUN_DIR="${RUN_ROOT}/${TIMESTAMP}"
fi
RUN_DIR="$(python3 - "${RUN_DIR}" <<'PY'
import os
import sys
print(os.path.abspath(sys.argv[1]))
PY
)"
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
if [[ ! -x "${PARSER_SCRIPT}" ]]; then
  echo "Missing parser script: ${PARSER_SCRIPT}"
  exit 1
fi

FRACTIONAL_BITS="$(python3 - "${MODEL_DIR}/meta.json" <<'PY'
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

python3 - "${MODEL_DIR}/meta.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

order = data.get('order', [])
shapes = data.get('shapes', {})
required = ['conv1.weight', 'conv1.bias', 'fc.weight', 'fc.bias']
missing = [k for k in required if k not in order or k not in shapes]
if missing:
    raise SystemExit(f'meta.json missing required entries: {missing}')
if len(order) < 100:
    raise SystemExit(f'meta.json order seems incomplete for full ResNet-50: len={len(order)}')
PY

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
  local watchdog_pid=""
  ./semi2k-party.x -pn "${PORT}" 0 "${PROGRAM}" >"${log0}" 2>&1 &
  P0=$!
  ./semi2k-party.x -pn "${PORT}" 1 "${PROGRAM}" >"${log1}" 2>&1 &
  P1=$!
  if [[ -n "${RUN_TIMEOUT_SECONDS}" ]]; then
    (
      sleep "${RUN_TIMEOUT_SECONDS}"
      kill "${P0}" "${P1}" >/dev/null 2>&1 || true
    ) &
    watchdog_pid=$!
  fi
  set +e
  wait "${P0}"; rc0=$?
  wait "${P1}"; rc1=$?
  set -e
  if [[ -n "${watchdog_pid}" ]]; then
    kill "${watchdog_pid}" >/dev/null 2>&1 || true
    wait "${watchdog_pid}" >/dev/null 2>&1 || true
  fi
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

echo "=== ResNet50 MPC Run ===" | tee "${RUN_DIR}/run.info"
echo "run_dir=${RUN_DIR}" | tee -a "${RUN_DIR}/run.info"
echo "model_dir=${MODEL_DIR}" | tee -a "${RUN_DIR}/run.info"
echo "input_file=${INPUT_FILE}" | tee -a "${RUN_DIR}/run.info"
echo "port=${PORT}" | tee -a "${RUN_DIR}/run.info"
echo "fractional_bits=${FRACTIONAL_BITS}" | tee -a "${RUN_DIR}/run.info"

./compile.py --ring 64 "${PROGRAM}" 2>&1 | tee "${COMPILE_LOG}"

RUN_STATUS=0
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
    if ! run_protocol_once "${TMP_P0_SECOND}" "${TMP_P1_SECOND}"; then
      RUN_STATUS=1
    fi
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
    RUN_STATUS=1
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
  --fractional-bits "${FRACTIONAL_BITS}"
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

if [[ "${RUN_STATUS}" -ne 0 ]]; then
  echo "Run ended without successful completion. See logs in ${RUN_DIR}."
  exit 1
fi
