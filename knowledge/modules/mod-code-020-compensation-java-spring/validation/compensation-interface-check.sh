#!/bin/bash
# compensation-interface-check.sh
# Tier 3 validation for mod-code-020-compensation
# Validates ERI constraints: compensable-interface-implemented, compensate-method-exists

SERVICE_DIR="${1:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

echo "=== Compensation Interface Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Check 1: Compensable interface exists
COMPENSABLE=$(find "$SERVICE_DIR" -name "Compensable.java" -type f 2>/dev/null | head -1)
if [ -n "$COMPENSABLE" ]; then
    pass "Compensable.java interface exists"
    
    # Check interface definition
    if grep -q "interface Compensable" "$COMPENSABLE" 2>/dev/null; then
        pass "Compensable is an interface"
    else
        fail "Compensable should be an interface"
    fi
    
    # Check for compensate method
    if grep -q "CompensationResult compensate" "$COMPENSABLE" 2>/dev/null; then
        pass "Compensable has compensate() method"
    else
        fail "Compensable interface missing compensate() method"
    fi
else
    fail "Compensable.java interface not found"
fi

# Check 2: Find services that implement Compensable
echo ""
echo "--- Service Implementation Check ---"

SERVICES=$(find "$SERVICE_DIR" -name "*ApplicationService.java" -type f 2>/dev/null)

if [ -z "$SERVICES" ]; then
    warn "No ApplicationService classes found"
else
    IMPLEMENTING_COUNT=0
    
    for SERVICE in $SERVICES; do
        SERVICE_NAME=$(basename "$SERVICE")
        
        if grep -q "implements Compensable" "$SERVICE" 2>/dev/null; then
            pass "$SERVICE_NAME implements Compensable"
            IMPLEMENTING_COUNT=$((IMPLEMENTING_COUNT + 1))
            
            # Check for compensate method implementation
            if grep -q "public CompensationResult compensate" "$SERVICE" 2>/dev/null; then
                pass "$SERVICE_NAME has compensate() implementation"
            else
                fail "$SERVICE_NAME implements Compensable but missing compensate() method"
            fi
            
            # Check for @Transactional on compensate
            if grep -B2 "public CompensationResult compensate" "$SERVICE" 2>/dev/null | grep -q "@Transactional"; then
                pass "$SERVICE_NAME compensate() is @Transactional"
            else
                warn "$SERVICE_NAME compensate() should be @Transactional"
            fi
        fi
    done
    
    if [ "$IMPLEMENTING_COUNT" -eq 0 ]; then
        warn "No services implement Compensable (may be OK if not participating in SAGAs)"
    fi
fi

# Check 3: CompensationResult structure
echo ""
echo "--- CompensationResult Check ---"

COMP_RESULT=$(find "$SERVICE_DIR" -name "CompensationResult.java" -type f 2>/dev/null | head -1)
if [ -n "$COMP_RESULT" ]; then
    pass "CompensationResult.java exists"
    
    # Check for factory methods
    if grep -q "static CompensationResult compensated" "$COMP_RESULT" 2>/dev/null; then
        pass "CompensationResult has compensated() factory"
    else
        warn "CompensationResult missing compensated() factory method"
    fi
    
    if grep -q "static CompensationResult alreadyCompensated" "$COMP_RESULT" 2>/dev/null; then
        pass "CompensationResult has alreadyCompensated() factory"
    else
        fail "CompensationResult missing alreadyCompensated() factory (required for idempotency)"
    fi
    
    if grep -q "static CompensationResult notFound" "$COMP_RESULT" 2>/dev/null; then
        pass "CompensationResult has notFound() factory"
    else
        warn "CompensationResult missing notFound() factory method"
    fi
else
    fail "CompensationResult.java not found"
fi

# Check 4: CompensationStatus enum
echo ""
echo "--- CompensationStatus Check ---"

COMP_STATUS=$(find "$SERVICE_DIR" -name "CompensationStatus.java" -type f 2>/dev/null | head -1)
if [ -n "$COMP_STATUS" ]; then
    pass "CompensationStatus.java exists"
    
    # Check for required values
    for STATUS in "COMPENSATED" "ALREADY_COMPENSATED" "NOT_FOUND" "FAILED"; do
        if grep -q "$STATUS" "$COMP_STATUS" 2>/dev/null; then
            pass "CompensationStatus has $STATUS"
        else
            fail "CompensationStatus missing $STATUS value"
        fi
    done
else
    fail "CompensationStatus.java enum not found"
fi

echo ""
echo "=== Compensation Interface Check Complete ==="
echo "Errors: $ERRORS"

exit $ERRORS
