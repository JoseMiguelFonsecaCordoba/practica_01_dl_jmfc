#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:5002}"
VALID_PAYLOAD='{"values":[10,11,12,13,14,15,16,17,18,19,20,19,18,17,16,15,14,13,12,11,10,9,10,11]}'
INVALID_PAYLOAD='{"values":[10,11,12]}'

echo "Test 1/3: health"
health_response="$(curl --fail --silent "${BASE_URL}/health")"
printf '%s\n' "$health_response"
grep -q '"status":"ok"' <<<"$health_response"

echo "Test 2/3: valid prediction"
prediction_response="$(
    curl --fail --silent \
        -X POST \
        -H 'Content-Type: application/json' \
        -d "$VALID_PAYLOAD" \
        "${BASE_URL}/predict"
)"
printf '%s\n' "$prediction_response"
grep -q '"model":"' <<<"$prediction_response"
grep -q '"input_points":24' <<<"$prediction_response"

echo "Test 3/3: invalid sequence"
status_code="$(
    curl --silent --output /tmp/forecasting-invalid-response.json \
        --write-out '%{http_code}' \
        -X POST \
        -H 'Content-Type: application/json' \
        -d "$INVALID_PAYLOAD" \
        "${BASE_URL}/predict"
)"
cat /tmp/forecasting-invalid-response.json
test "$status_code" = "422"

echo "All inference curl tests passed."
