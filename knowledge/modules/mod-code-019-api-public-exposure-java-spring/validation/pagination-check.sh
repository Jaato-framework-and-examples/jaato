#!/bin/bash
# pagination-check.sh
# Tier 3 validation for mod-code-019-api-public-exposure
# Validates ERI constraint: page-response-structure, page-metadata-fields

SERVICE_DIR="${1:-.}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

echo "=== Pagination Structure Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Check 1: PageResponse.java exists
PAGE_RESPONSE=$(find "$SERVICE_DIR" -name "PageResponse.java" -type f 2>/dev/null | head -1)
if [ -n "$PAGE_RESPONSE" ]; then
    pass "PageResponse.java exists"
    
    # Check 1a: PageResponse has content field
    if grep -q "List<T> content" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageResponse has 'content' field"
    else
        fail "PageResponse missing 'content' field (List<T>)"
    fi
    
    # Check 1b: PageResponse has page metadata
    if grep -q "PageMetadata page" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageResponse has 'page' metadata field"
    else
        fail "PageResponse missing 'page' metadata field"
    fi
    
    # Check 1c: PageResponse has _links
    if grep -q "_links" "$PAGE_RESPONSE" 2>/dev/null || grep -q "Links links" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageResponse has '_links' field"
    else
        fail "PageResponse missing '_links' field for HATEOAS"
    fi
else
    fail "PageResponse.java not found"
fi

# Check 2: PageMetadata structure
if [ -n "$PAGE_RESPONSE" ]; then
    echo ""
    echo "--- PageMetadata Validation ---"
    
    # Check for required fields in PageMetadata
    if grep -q "int number" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageMetadata has 'number' field"
    else
        fail "PageMetadata missing 'number' field"
    fi
    
    if grep -q "int size" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageMetadata has 'size' field"
    else
        fail "PageMetadata missing 'size' field"
    fi
    
    if grep -q "long totalElements" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageMetadata has 'totalElements' field"
    else
        fail "PageMetadata missing 'totalElements' field"
    fi
    
    if grep -q "int totalPages" "$PAGE_RESPONSE" 2>/dev/null; then
        pass "PageMetadata has 'totalPages' field"
    else
        fail "PageMetadata missing 'totalPages' field"
    fi
fi

echo ""
echo "=== Pagination Check Complete ==="
echo "Errors: $ERRORS"

exit $ERRORS
