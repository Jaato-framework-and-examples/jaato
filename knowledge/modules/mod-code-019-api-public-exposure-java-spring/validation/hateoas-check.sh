#!/bin/bash
# hateoas-check.sh
# Tier 3 validation for mod-code-019-api-public-exposure
# Validates ERI constraint: hateoas-self-link, model-assembler-exists

SERVICE_DIR="${1:-.}"
PACKAGE_PATH="${2:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

echo "=== HATEOAS Structure Check ==="
echo "Service directory: $SERVICE_DIR"
echo ""

# Check 1: Find ModelAssembler classes
ASSEMBLERS=$(find "$SERVICE_DIR" -name "*ModelAssembler.java" -type f 2>/dev/null)
ASSEMBLER_COUNT=$(echo "$ASSEMBLERS" | grep -c "ModelAssembler" 2>/dev/null || echo "0")

if [ "$ASSEMBLER_COUNT" -gt 0 ]; then
    pass "Found $ASSEMBLER_COUNT ModelAssembler class(es)"
    
    # Check each assembler
    for ASSEMBLER in $ASSEMBLERS; do
        ASSEMBLER_NAME=$(basename "$ASSEMBLER")
        echo ""
        echo "--- Checking: $ASSEMBLER_NAME ---"
        
        # Check 1a: Extends RepresentationModelAssemblerSupport
        if grep -q "RepresentationModelAssemblerSupport" "$ASSEMBLER" 2>/dev/null; then
            pass "$ASSEMBLER_NAME extends RepresentationModelAssemblerSupport"
        else
            fail "$ASSEMBLER_NAME does not extend RepresentationModelAssemblerSupport"
        fi
        
        # Check 1b: Has @Component annotation
        if grep -q "@Component" "$ASSEMBLER" 2>/dev/null; then
            pass "$ASSEMBLER_NAME has @Component annotation"
        else
            fail "$ASSEMBLER_NAME missing @Component annotation"
        fi
        
        # Check 1c: Adds self link
        if grep -q "withSelfRel" "$ASSEMBLER" 2>/dev/null; then
            pass "$ASSEMBLER_NAME adds self link (withSelfRel)"
        else
            fail "$ASSEMBLER_NAME missing self link (withSelfRel)"
        fi
        
        # Check 1d: Uses WebMvcLinkBuilder
        if grep -q "WebMvcLinkBuilder" "$ASSEMBLER" 2>/dev/null || grep -q "linkTo" "$ASSEMBLER" 2>/dev/null; then
            pass "$ASSEMBLER_NAME uses WebMvcLinkBuilder"
        else
            warn "$ASSEMBLER_NAME may not use WebMvcLinkBuilder"
        fi
    done
else
    fail "No ModelAssembler classes found"
fi

# Check 2: Response DTOs extend RepresentationModel
echo ""
echo "--- Response DTO Check ---"

RESPONSE_DTOS=$(find "$SERVICE_DIR" -name "*Response.java" -path "*/dto/*" -type f 2>/dev/null)

for DTO in $RESPONSE_DTOS; do
    DTO_NAME=$(basename "$DTO")
    
    # Skip PageResponse as it uses Links directly
    if [[ "$DTO_NAME" == "PageResponse.java" ]]; then
        continue
    fi
    
    if grep -q "RepresentationModel" "$DTO" 2>/dev/null; then
        pass "$DTO_NAME extends RepresentationModel"
    else
        warn "$DTO_NAME does not extend RepresentationModel (may not support HATEOAS links)"
    fi
done

echo ""
echo "=== HATEOAS Check Complete ==="
echo "Errors: $ERRORS"

exit $ERRORS
