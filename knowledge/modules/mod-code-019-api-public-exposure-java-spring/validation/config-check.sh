#!/bin/bash
# config-check.sh
# Tier 3 validation for mod-code-019-api-public-exposure
# Validates ERI constraint: default-page-size, max-page-size, zero-indexed-pagination

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

echo "=== Pagination Configuration Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Find application.yml files
APP_CONFIGS=$(find "$SERVICE_DIR" -name "application*.yml" -o -name "application*.yaml" 2>/dev/null)

if [ -z "$APP_CONFIGS" ]; then
    fail "No application.yml configuration files found"
    exit 1
fi

# Combine all configs for checking
COMBINED_CONFIG=$(cat $APP_CONFIGS 2>/dev/null)

# Check 1: Default page size (ADR-001 requires 20)
if echo "$COMBINED_CONFIG" | grep -q "default-page-size.*20"; then
    pass "Default page size is 20 (per ADR-001)"
elif echo "$COMBINED_CONFIG" | grep -q "default-page-size"; then
    PAGE_SIZE=$(echo "$COMBINED_CONFIG" | grep "default-page-size" | head -1)
    warn "Default page size configured but not 20: $PAGE_SIZE"
else
    warn "default-page-size not configured (will use Spring default)"
fi

# Check 2: Max page size (ADR-001 requires 100)
if echo "$COMBINED_CONFIG" | grep -q "max-page-size.*100"; then
    pass "Max page size is 100 (per ADR-001)"
elif echo "$COMBINED_CONFIG" | grep -q "max-page-size"; then
    MAX_SIZE=$(echo "$COMBINED_CONFIG" | grep "max-page-size" | head -1)
    warn "Max page size configured but not 100: $MAX_SIZE"
else
    warn "max-page-size not configured (will use Spring default)"
fi

# Check 3: Zero-indexed pagination (ADR-001 requires false for one-indexed)
if echo "$COMBINED_CONFIG" | grep -q "one-indexed-parameters.*false"; then
    pass "Pagination is zero-indexed (per ADR-001)"
elif echo "$COMBINED_CONFIG" | grep -q "one-indexed-parameters.*true"; then
    fail "Pagination is one-indexed but ADR-001 requires zero-indexed"
else
    pass "one-indexed-parameters not set (defaults to false/zero-indexed)"
fi

# Check 4: HATEOAS HAL format
if echo "$COMBINED_CONFIG" | grep -q "use-hal-as-default-json-media-type.*true"; then
    pass "HATEOAS configured to use HAL format"
else
    warn "HATEOAS HAL format not explicitly configured"
fi

# Check 5: PageableConfig.java exists
PAGEABLE_CONFIG=$(find "$SERVICE_DIR" -name "PageableConfig.java" -type f 2>/dev/null | head -1)
if [ -n "$PAGEABLE_CONFIG" ]; then
    pass "PageableConfig.java exists"
    
    # Check for EnableSpringDataWebSupport
    if grep -q "EnableSpringDataWebSupport" "$PAGEABLE_CONFIG" 2>/dev/null; then
        pass "PageableConfig has @EnableSpringDataWebSupport"
    else
        warn "PageableConfig missing @EnableSpringDataWebSupport annotation"
    fi
else
    warn "PageableConfig.java not found (pagination may use defaults)"
fi

echo ""
echo "=== Configuration Check Complete ==="
echo "Errors: $ERRORS, Warnings: $WARNINGS"

exit $ERRORS
