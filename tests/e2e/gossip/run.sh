#!/bin/bash
# Run the gossip E2E test with Docker Compose.
#
# Usage:
#   ./run.sh              # dev flavor (from repo source)
#   ./run.sh dev          # explicit dev
#   ./run.sh prod         # prod flavor (from TestPyPI)
#
# For prod, set JAATO_SDK_VERSION and JAATO_SERVER_VERSION env vars.
set -euo pipefail

FLAVOR="${1:-dev}"
cd "$(dirname "$0")"

echo "Running gossip E2E test (flavor: ${FLAVOR})"

# Determine the test-runner service name for exit code tracking
if [ "$FLAVOR" = "prod" ]; then
    TEST_RUNNER="test-runner-prod"
else
    TEST_RUNNER="test-runner"
fi

docker compose --profile "$FLAVOR" up --build --abort-on-container-exit --exit-code-from "$TEST_RUNNER"
rc=$?

docker compose --profile "$FLAVOR" down -v

exit $rc
