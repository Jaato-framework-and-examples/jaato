#!/bin/bash
# compensation-endpoint-check.sh
# Tier 3 validation for mod-code-020-compensation
# Validates ERI constraints: compensation-endpoint-exists, correlation-id-header-required

SERVICE_DIR="${1:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

echo "=== Compensation Endpoint Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Find controller files
CONTROLLERS=$(find "$SERVICE_DIR" -name "*Controller.java" -type f 2>/dev/null)

if [ -z "$CONTROLLERS" ]; then
    fail "No Controller classes found"
    exit 1
fi

ENDPOINT_FOUND=0

for CONTROLLER in $CONTROLLERS; do
    CONTROLLER_NAME=$(basename "$CONTROLLER")
    
    # Check for /compensate endpoint
    if grep -q 'PostMapping.*"/compensate"' "$CONTROLLER" 2>/dev/null || \
       grep -q "@PostMapping.*compensate" "$CONTROLLER" 2>/dev/null; then
        
        pass "$CONTROLLER_NAME has /compensate endpoint"
        ENDPOINT_FOUND=$((ENDPOINT_FOUND + 1))
        
        echo "--- Checking $CONTROLLER_NAME ---"
        
        # Check for @PostMapping
        if grep -q "@PostMapping" "$CONTROLLER" 2>/dev/null; then
            pass "Uses @PostMapping (correct HTTP method)"
        else
            fail "Compensation endpoint should use @PostMapping"
        fi
        
        # Check for CompensationRequest parameter
        if grep -A5 'PostMapping.*compensate' "$CONTROLLER" 2>/dev/null | grep -q "CompensationRequest"; then
            pass "Accepts CompensationRequest"
        else
            warn "Should accept CompensationRequest as parameter"
        fi
        
        # Check for @Valid annotation
        if grep -A5 'PostMapping.*compensate' "$CONTROLLER" 2>/dev/null | grep -q "@Valid"; then
            pass "Request is validated with @Valid"
        else
            warn "Should validate request with @Valid"
        fi
        
        # Check for X-Correlation-ID header
        if grep -A10 'PostMapping.*compensate' "$CONTROLLER" 2>/dev/null | grep -q "X-Correlation-ID"; then
            pass "Requires X-Correlation-ID header"
        else
            fail "Must require X-Correlation-ID header"
        fi
        
        # Check for CompensationResult return type
        if grep -B2 'PostMapping.*compensate' "$CONTROLLER" 2>/dev/null | grep -q "CompensationResult\|ResponseEntity"; then
            pass "Returns CompensationResult"
        else
            warn "Should return CompensationResult"
        fi
        
        # Check for proper HTTP status handling
        if grep -A20 'compensate.*CompensationRequest' "$CONTROLLER" 2>/dev/null | grep -q "status(404)\|status(500)\|ResponseEntity.ok"; then
            pass "Has proper HTTP status handling"
        else
            warn "Should return proper HTTP status codes (200, 404, 500)"
        fi
    fi
done

if [ "$ENDPOINT_FOUND" -eq 0 ]; then
    warn "No /compensate endpoint found (may be OK if not participating in SAGAs)"
fi

echo ""
echo "=== Compensation Endpoint Check Complete ==="
echo "Errors: $ERRORS"
echo "Endpoints found: $ENDPOINT_FOUND"

exit $ERRORS
