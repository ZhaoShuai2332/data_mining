#!/bin/bash
set -euo pipefail

PROGRAM="fc2_mnist_infer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MP_SPDZ_DIR="${SCRIPT_DIR}/../MP-SPDZ"
ALT_MP_SPDZ_DIR="${SCRIPT_DIR}/../mp-spdz-0.4.2"

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
  local log0 log1 rc0 rc1
  log0="$(mktemp)"
  log1="$(mktemp)"
  ./semi2k-party.x -pn "${PORT}" 0 "${PROGRAM}" >"${log0}" 2>&1 &
  P0=$!
  ./semi2k-party.x -pn "${PORT}" 1 "${PROGRAM}" >"${log1}" 2>&1 &
  P1=$!
  wait "${P0}"; rc0=$?
  wait "${P1}"; rc1=$?
  cat "${log0}"
  cat "${log1}"
  LAST_LOG0="${log0}"
  LAST_LOG1="${log1}"
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

./compile.py --ring 64 "${PROGRAM}"

prepare_inputs_dir_layout
if ! run_protocol_once; then
  if grep -Eq "not enough inputs in Player-Data/Input-P0-0|not enough inputs in Player-Data/Input-P1-0" "${LAST_LOG0}" "${LAST_LOG1}"; then
    echo "Detected flat input layout in this MP-SPDZ build, retrying with flat files."
    prepare_inputs_flat_layout
    run_protocol_once
  else
    exit 1
  fi
fi
