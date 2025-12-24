#!/bin/bash
# transaction-log-check.sh
# Tier 3 validation for mod-code-020-compensation
# Validates ERI constraint: transaction-log-entity-exists

SERVICE_DIR="${1:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; WARNINGS=$((WARNINGS + 1)); }

ERRORS=0
WARNINGS=0

echo "=== Transaction Log Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Check 1: TransactionLog entity exists
TX_LOG=$(find "$SERVICE_DIR" -name "TransactionLog.java" -type f 2>/dev/null | head -1)
if [ -n "$TX_LOG" ]; then
    pass "TransactionLog.java exists"
    
    # Check for required fields
    echo ""
    echo "--- TransactionLog Fields ---"
    
    if grep -q "transactionId" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has transactionId"
    else
        fail "TransactionLog missing transactionId field"
    fi
    
    if grep -q "operationType" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has operationType"
    else
        fail "TransactionLog missing operationType field"
    fi
    
    if grep -q "entityId" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has entityId"
    else
        fail "TransactionLog missing entityId field"
    fi
    
    if grep -q "operationData" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has operationData"
    else
        warn "TransactionLog missing operationData field"
    fi
    
    # Check for markCompensated method
    if grep -q "markCompensated" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has markCompensated() method"
    else
        fail "TransactionLog missing markCompensated() method"
    fi
    
    # Check for isCompensated method
    if grep -q "isCompensated" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has isCompensated() method"
    else
        fail "TransactionLog missing isCompensated() method"
    fi
    
    # Check for status enum
    if grep -q "TransactionLogStatus" "$TX_LOG" 2>/dev/null; then
        pass "TransactionLog has status enum"
    else
        warn "TransactionLog should have status enum"
    fi
else
    warn "TransactionLog.java not found (may use alternative tracking)"
fi

# Check 2: TransactionLogRepository exists
echo ""
echo "--- TransactionLogRepository Check ---"

TX_REPO=$(find "$SERVICE_DIR" -name "TransactionLogRepository.java" -type f 2>/dev/null | head -1)
if [ -n "$TX_REPO" ]; then
    pass "TransactionLogRepository.java exists"
    
    # Check for required methods
    if grep -q "findByTransactionId" "$TX_REPO" 2>/dev/null; then
        pass "Repository has findByTransactionId()"
    else
        fail "Repository missing findByTransactionId() method"
    fi
    
    if grep -q "save" "$TX_REPO" 2>/dev/null; then
        pass "Repository has save()"
    else
        fail "Repository missing save() method"
    fi
else
    if [ -n "$TX_LOG" ]; then
        warn "TransactionLogRepository.java not found (required if using TransactionLog)"
    fi
fi

# Check 3: Application services log transactions
echo ""
echo "--- Transaction Logging Usage ---"

SERVICES=$(find "$SERVICE_DIR" -name "*ApplicationService.java" -type f 2>/dev/null)
LOGGING_COUNT=0

for SERVICE in $SERVICES; do
    SERVICE_NAME=$(basename "$SERVICE")
    
    if grep -q "TransactionLog" "$SERVICE" 2>/dev/null; then
        pass "$SERVICE_NAME uses TransactionLog"
        LOGGING_COUNT=$((LOGGING_COUNT + 1))
    fi
done

if [ "$LOGGING_COUNT" -eq 0 ] && [ -n "$TX_LOG" ]; then
    warn "TransactionLog exists but no services use it"
fi

echo ""
echo "=== Transaction Log Check Complete ==="
echo "Errors: $ERRORS, Warnings: $WARNINGS"

exit $ERRORS
