#!/usr/bin/env bash
set -euo pipefail

host_uid="${HOST_UID:-1000}"
host_gid="${HOST_GID:-1000}"

cleanup() {
    chown -R "${host_uid}:${host_gid}" \
        /workspace/forecasting/artifacts \
        /workspace/forecasting/data \
        /workspace/mlflow-data \
        /workspace/mlruns
}

trap cleanup EXIT

python -m forecasting.train "$@"
