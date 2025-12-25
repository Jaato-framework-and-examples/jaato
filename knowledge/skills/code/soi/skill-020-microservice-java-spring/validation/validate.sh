#!/bin/bash
# validate.sh - Main validation orchestrator for skill-code-020-generate-microservice-java-spring
#
# This script orchestrates validations across tiers for CODE domain:
#   Tier 1 Universal: Traceability - ALWAYS
#   Tier 1 Code: Project structure, naming - ALWAYS (CODE domain)
#   Tier 2: Tech Stack (Maven, Spring Boot, Docker) - CONDITIONAL
#   Tier 3: Module (Hexagonal architecture, Circuit Breaker if enabled) - CONDITIONAL
#   Tier 4: Runtime (CI/CD tests) - FUTURE (not implemented here)
#
# Usage: ./validate.sh <service-dir> <base-package-path> [input-json]
# Example: ./validate.sh ./customer-service com/company/customer
#
# Exit codes:
#   0 - All validations passed
#   1 - One or more validations failed

SERVICE_DIR=${1:-.}
BASE_PACKAGE_PATH=${2:-""}
INPUT_JSON=${3:-""}

if [ ! -d "$SERVICE_DIR" ]; then
    echo "Error: Service directory not found: $SERVICE_DIR"
    echo "Usage: $0 <service-dir> <base-package-path> [input-json]"
    exit 1
fi

if [ -z "$BASE_PACKAGE_PATH" ]; then
    echo "Error: Base package path required"
    echo "Usage: $0 <service-dir> <base-package-path> [input-json]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KNOWLEDGE_BASE="$SCRIPT_DIR/../../.."
VALIDATORS="$KNOWLEDGE_BASE/validators"

echo "════════════════════════════════════════════════════════════"
echo "  SKILL-CODE-020 VALIDATION"
echo "  Generate Microservice - Java/Spring"
echo "  Service: $SERVICE_DIR"
echo "  Package: $BASE_PACKAGE_PATH"
echo "════════════════════════════════════════════════════════════"
echo ""

ERRORS=0

# ==========================================
# TIER 1 UNIVERSAL: TRACEABILITY
# ==========================================
echo "═══════════════════════════════════════════════════════════"
echo "TIER 1 UNIVERSAL: Traceability"
echo "═══════════════════════════════════════════════════════════"
echo ""

if bash "$VALIDATORS/tier-1-universal/traceability/traceability-check.sh" "$SERVICE_DIR"; then
    : # pass
else
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ==========================================
# TIER 1 CODE: INFRASTRUCTURE VALIDATIONS
# ==========================================
echo "═══════════════════════════════════════════════════════════"
echo "TIER 1 CODE: Infrastructure Validations"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Project structure
if bash "$VALIDATORS/tier-1-universal/code-projects/project-structure/project-structure-check.sh" "$SERVICE_DIR"; then
    : # pass
else
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Naming conventions
if bash "$VALIDATORS/tier-1-universal/code-projects/naming-conventions/naming-conventions-check.sh" "$SERVICE_DIR"; then
    : # pass
else
    ERRORS=$((ERRORS + 1))
fi

echo ""

# ==========================================
# TIER 2: ARTIFACT VALIDATIONS
# ==========================================
echo "═══════════════════════════════════════════════════════════"
echo "TIER 2: Artifact Validations"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Java Spring
if [ -f "$SERVICE_DIR/pom.xml" ]; then
    echo "--- Java Spring Stack ---"
    
    for script in "$VALIDATORS/tier-2-technology/code-projects/java-spring/"*-check.sh; do
        if [ -f "$script" ]; then
            if bash "$script" "$SERVICE_DIR"; then
                : # pass
            else
                ERRORS=$((ERRORS + 1))
            fi
        fi
    done
    
    echo ""
fi

# Docker
if [ -f "$SERVICE_DIR/Dockerfile" ]; then
    echo "--- Docker Deployment ---"
    
    if bash "$VALIDATORS/tier-2-technology/deployments/docker/dockerfile-check.sh" "$SERVICE_DIR"; then
        : # pass
    else
        ERRORS=$((ERRORS + 1))
    fi
    
    echo ""
fi

echo ""

# ==========================================
# TIER 3: MODULE VALIDATIONS
# ==========================================
echo "═══════════════════════════════════════════════════════════"
echo "TIER 3: Module Validations"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Hexagonal architecture (always for skill-code-020)
echo "--- Hexagonal Architecture Module ---"

if bash "$KNOWLEDGE_BASE/skills/modules/mod-015-hexagonal-base-java-spring/validation/hexagonal-structure-check.sh" "$SERVICE_DIR" "$BASE_PACKAGE_PATH"; then
    : # pass
else
    ERRORS=$((ERRORS + 1))
fi

echo ""

# Circuit Breaker (if enabled in input)
if [ -n "$INPUT_JSON" ] && [ -f "$INPUT_JSON" ]; then
    CB_ENABLED=$(jq -r '.features.circuit_breaker // false' "$INPUT_JSON" 2>/dev/null)
    
    if [ "$CB_ENABLED" = "true" ]; then
        echo "--- Circuit Breaker Module (Feature Enabled) ---"
        
        if bash "$KNOWLEDGE_BASE/skills/modules/mod-001-circuit-breaker-java-resilience4j/validation/circuit-breaker-check.sh" "$SERVICE_DIR"; then
            : # pass
        else
            ERRORS=$((ERRORS + 1))
        fi
        
        echo ""
    fi
fi

echo ""

# ==========================================
# SUMMARY
# ==========================================
echo "════════════════════════════════════════════════════════════"
echo "  VALIDATION SUMMARY"
echo "════════════════════════════════════════════════════════════"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "\033[0;32m✅ ALL VALIDATIONS PASSED\033[0m"
    echo ""
    echo "Validated:"
    echo "  - Tier 1 Universal: Traceability"
    echo "  - Tier 1 Code: Project structure, naming conventions"
    echo "  - Tier 2: Java Spring, Docker (if applicable)"
    echo "  - Tier 3: Hexagonal architecture (ADR-009)"
    
    if [ -n "$INPUT_JSON" ] && [ -f "$INPUT_JSON" ]; then
        CB_ENABLED=$(jq -r '.features.circuit_breaker // false' "$INPUT_JSON" 2>/dev/null)
        if [ "$CB_ENABLED" = "true" ]; then
            echo "  - Tier 3: Circuit Breaker (Resilience4j)"
        fi
    fi
    
    exit 0
else
    echo -e "\033[0;31m❌ VALIDATION FAILED\033[0m"
    echo "   Errors: $ERRORS"
    echo ""
    echo "Review the output above for details."
    exit 1
fi
