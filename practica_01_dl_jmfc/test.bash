#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

check_endpoint() {
  local endpoint="$1"
  echo "Checking ${BASE_URL}${endpoint}"
  curl --silent --show-error --fail "${BASE_URL}${endpoint}" > /dev/null
}

check_endpoint "/v2/health/live"
check_endpoint "/v2/health/ready"
check_endpoint "/v2/models/vgg11/ready"

echo "Triton health checks OK."
