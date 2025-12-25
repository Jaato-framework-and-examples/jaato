#!/bin/bash
# validate.sh
# Orchestrates validation for skill-021-api-rest-java-spring
# EXTENDS skill-020 validation + adds Tier 3 for mod-019, mod-020

set -e

SERVICE_DIR="${1:-.}"
API_LAYER="${2:-domain}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(dirname "$SCRIPT_DIR")"
SKILL_020_DIR="${SKILL_020_DIR:-/home/claude/enablement-2.0/skills/code/soi/skill-020-microservice-java-spring}"
MODULES_DIR="${MODULES_DIR:-/home/claude/enablement-2.0/modules}"

# Counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

run_check() {
    local name="$1"
    local script="$2"
    local args="${3:-}"
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    echo -e "${BLUE}Running:${NC} $name"
    
    if [ -f "$script" ]; then
        if bash "$script" $args; then
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
            echo -e "${GREEN}✅ PASSED:${NC} $name"
        else
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            echo -e "${RED}❌ FAILED:${NC} $name"
        fi
    else
        echo -e "${YELLOW}⚠️  SKIPPED:${NC} $name (script not found: $script)"
    fi
    
    echo ""
}

echo "=============================================="
echo "skill-021-api-rest-java-spring Validation"
echo "=============================================="
echo "Service: $SERVICE_DIR"
echo "API Layer: $API_LAYER"
echo "Extends: skill-020-microservice-java-spring"
echo ""

# =============================================================================
# INHERITED VALIDATION (from skill-020)
# =============================================================================
echo "=== INHERITED: skill-020 Validation ==="
echo ""

if [ -f "$SKILL_020_DIR/validation/validate.sh" ]; then
    echo "Running parent skill validation..."
    bash "$SKILL_020_DIR/validation/validate.sh" "$SERVICE_DIR"
    INHERITED_ERRORS=$?
    FAILED_CHECKS=$((FAILED_CHECKS + INHERITED_ERRORS))
    echo ""
else
    echo -e "${YELLOW}⚠️  Parent skill-020 validation not found, running standalone${NC}"
    echo ""
fi

# =============================================================================
# ADDITIONAL TIER 3: mod-019 (API Public Exposure)
# =============================================================================
echo "=== ADDITIONAL: Tier 3 mod-019 ==="
echo ""

run_check "Pagination Structure (mod-019)" \
    "$MODULES_DIR/mod-code-019-api-public-exposure-java-spring/validation/pagination-check.sh" \
    "$SERVICE_DIR"

run_check "Pagination Config (mod-019)" \
    "$MODULES_DIR/mod-code-019-api-public-exposure-java-spring/validation/config-check.sh" \
    "$SERVICE_DIR"

# HATEOAS (experience, domain only)
if [[ "$API_LAYER" == "experience" || "$API_LAYER" == "domain" ]]; then
    run_check "HATEOAS Structure (mod-019)" \
        "$MODULES_DIR/mod-code-019-api-public-exposure-java-spring/validation/hateoas-check.sh" \
        "$SERVICE_DIR"
else
    echo -e "${YELLOW}⚠️  SKIPPED:${NC} HATEOAS (not required for $API_LAYER layer)"
    echo ""
fi

# =============================================================================
# ADDITIONAL TIER 3: mod-020 (Compensation) - Domain only
# =============================================================================
if [[ "$API_LAYER" == "domain" ]]; then
    echo "=== ADDITIONAL: Tier 3 mod-020 (Domain layer) ==="
    echo ""
    
    run_check "Compensation Interface (mod-020)" \
        "$MODULES_DIR/mod-code-020-compensation-java-spring/validation/compensation-interface-check.sh" \
        "$SERVICE_DIR"
    
    run_check "Compensation Endpoint (mod-020)" \
        "$MODULES_DIR/mod-code-020-compensation-java-spring/validation/compensation-endpoint-check.sh" \
        "$SERVICE_DIR"
    
    run_check "Transaction Log (mod-020)" \
        "$MODULES_DIR/mod-code-020-compensation-java-spring/validation/transaction-log-check.sh" \
        "$SERVICE_DIR"
else
    echo -e "${YELLOW}⚠️  SKIPPED:${NC} Compensation checks (not required for $API_LAYER layer)"
    echo ""
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo "=============================================="
echo "VALIDATION SUMMARY"
echo "=============================================="
echo "Total Checks: $TOTAL_CHECKS"
echo -e "Passed: ${GREEN}$PASSED_CHECKS${NC}"
echo -e "Failed: ${RED}$FAILED_CHECKS${NC}"
echo ""

if [ "$FAILED_CHECKS" -gt 0 ]; then
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    exit $FAILED_CHECKS
else
    echo -e "${GREEN}✅ ALL VALIDATIONS PASSED${NC}"
    exit 0
fi
