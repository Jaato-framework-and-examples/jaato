#!/bin/bash
# hexagonal-structure-check.sh
# Validates Hexagonal Light architecture (ADR-009 compliance)
#
# Version: 1.1.0
# Updated: 2025-12-03
# Fix: Exclude comments from annotation detection (false positives)

SERVICE_DIR=${1:-.}
BASE_PACKAGE_PATH=${2:-""}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✅ PASS:${NC} $1"; }
fail() { echo -e "${RED}❌ FAIL:${NC} $1"; ERRORS=$((ERRORS + 1)); }
warn() { echo -e "${YELLOW}⚠️  WARN:${NC} $1"; }

ERRORS=0

# Determine paths
if [ -n "$BASE_PACKAGE_PATH" ]; then
    DOMAIN_PATH="$SERVICE_DIR/src/main/java/$BASE_PACKAGE_PATH/domain"
    APP_PATH="$SERVICE_DIR/src/main/java/$BASE_PACKAGE_PATH/application"
    ADAPTER_PATH="$SERVICE_DIR/src/main/java/$BASE_PACKAGE_PATH/adapter"
    INFRA_PATH="$SERVICE_DIR/src/main/java/$BASE_PACKAGE_PATH/infrastructure"
else
    # Try to find hexagonal structure automatically
    DOMAIN_PATH=$(find "$SERVICE_DIR/src/main/java" -type d -name "domain" 2>/dev/null | head -1)
    APP_PATH=$(find "$SERVICE_DIR/src/main/java" -type d -name "application" 2>/dev/null | head -1)
    ADAPTER_PATH=$(find "$SERVICE_DIR/src/main/java" -type d -name "adapter" 2>/dev/null | head -1)
    INFRA_PATH=$(find "$SERVICE_DIR/src/main/java" -type d -name "infrastructure" 2>/dev/null | head -1)
fi

echo "════════════════════════════════════════════════════════════"
echo "  TIER 3 - HEXAGONAL ARCHITECTURE VALIDATION (mod-015)"
echo "════════════════════════════════════════════════════════════"
echo "  Target: $SERVICE_DIR"
echo "  ADR: ADR-009 (Hexagonal Light)"
echo "════════════════════════════════════════════════════════════"

# ==========================================
# 1. STRUCTURE VALIDATION
# ==========================================

echo ""
echo "── Structure ──────────────────────────────────────────────"

# Check domain layer exists
if [ -d "$DOMAIN_PATH" ]; then
    pass "Domain layer exists"
else
    fail "Domain layer missing: $DOMAIN_PATH"
fi

# Check application layer exists
if [ -d "$APP_PATH" ]; then
    pass "Application layer exists"
else
    fail "Application layer missing: $APP_PATH"
fi

# Check adapter layer exists
if [ -d "$ADAPTER_PATH" ]; then
    pass "Adapter layer exists"
else
    fail "Adapter layer missing: $ADAPTER_PATH"
fi

# Check infrastructure exists (optional)
if [ -d "$INFRA_PATH" ]; then
    pass "Infrastructure layer exists"
else
    warn "Infrastructure layer not found (optional)"
fi

# ==========================================
# 2. DOMAIN LAYER PURITY (CRITICAL - ADR-009)
# ==========================================

echo ""
echo "── Domain Purity (ADR-009) ────────────────────────────────"

if [ -d "$DOMAIN_PATH" ]; then
    # Check for Spring annotations in domain (excluding comments)
    # Only match lines where annotation is actual code, not in comments
    SPRING_IN_DOMAIN=0
    for file in $(find "$DOMAIN_PATH" -name "*.java" -type f 2>/dev/null); do
        # Remove single-line comments, then search for annotations at line start
        matches=$(sed 's|//.*||g' "$file" | \
                  grep -cE "^\s*@(Service|Component|Autowired|Repository|Bean)\b" 2>/dev/null)
        matches=${matches:-0}
        SPRING_IN_DOMAIN=$((SPRING_IN_DOMAIN + matches))
    done
    
    if [ "$SPRING_IN_DOMAIN" -gt 0 ]; then
        fail "Spring annotations found in domain layer (ADR-009 violation)"
        # Show actual violations (not in comments)
        for file in $(find "$DOMAIN_PATH" -name "*.java" -type f 2>/dev/null); do
            sed 's|//.*||g' "$file" | grep -nE "^\s*@(Service|Component|Autowired|Repository|Bean)\b" 2>/dev/null | \
                while read -r line; do echo "  $file:$line"; done | head -3
        done
    else
        pass "Domain layer is Spring-free (ADR-009 compliant)"
    fi
    
    # Check for JPA annotations in domain model (excluding comments)
    DOMAIN_MODEL_PATH="$DOMAIN_PATH/model"
    if [ -d "$DOMAIN_MODEL_PATH" ]; then
        JPA_IN_MODEL=0
        for file in $(find "$DOMAIN_MODEL_PATH" -name "*.java" -type f 2>/dev/null); do
            matches=$(sed 's|//.*||g' "$file" | \
                      grep -cE "^\s*@(Entity|Table|Column|Id|GeneratedValue)\b" 2>/dev/null)
            matches=${matches:-0}
            JPA_IN_MODEL=$((JPA_IN_MODEL + matches))
        done
        
        if [ "$JPA_IN_MODEL" -gt 0 ]; then
            fail "JPA annotations found in domain model (ADR-009 violation)"
            for file in $(find "$DOMAIN_MODEL_PATH" -name "*.java" -type f 2>/dev/null); do
                sed 's|//.*||g' "$file" | grep -nE "^\s*@(Entity|Table|Column|Id|GeneratedValue)\b" 2>/dev/null | \
                    while read -r line; do echo "  $file:$line"; done | head -3
            done
        else
            pass "Domain model is JPA-free (ADR-009 compliant)"
        fi
    fi
    
    # Check repository is interface in domain
    REPO_PATH="$DOMAIN_PATH/repository"
    if [ -d "$REPO_PATH" ]; then
        REPO_IS_INTERFACE=$(grep -r "public interface.*Repository" "$REPO_PATH" 2>/dev/null | wc -l || echo "0")
        if [ "$REPO_IS_INTERFACE" -gt 0 ]; then
            pass "Repository is interface in domain layer"
        else
            fail "Repository should be an interface in domain layer"
        fi
        
        # Check repository has NO @Repository annotation (excluding comments)
        REPO_ANNOTATION=0
        for file in $(find "$REPO_PATH" -name "*.java" -type f 2>/dev/null); do
            matches=$(sed 's|//.*||g' "$file" | \
                      grep -cE "^\s*@Repository\b" 2>/dev/null)
            matches=${matches:-0}
            REPO_ANNOTATION=$((REPO_ANNOTATION + matches))
        done
        
        if [ "$REPO_ANNOTATION" -gt 0 ]; then
            fail "@Repository annotation in domain layer (should be in adapter)"
        else
            pass "Repository interface has no Spring annotations"
        fi
    fi
fi

# ==========================================
# 3. APPLICATION LAYER VALIDATION
# ==========================================

echo ""
echo "── Application Layer ──────────────────────────────────────"

if [ -d "$APP_PATH" ]; then
    # Application services should have @Service
    SERVICE_COUNT=$(find "$APP_PATH" -name "*Service.java" -type f 2>/dev/null | wc -l)
    if [ "$SERVICE_COUNT" -gt 0 ]; then
        SERVICE_WITH_ANNOTATION=0
        for file in $(find "$APP_PATH" -name "*Service.java" -type f 2>/dev/null); do
            if sed 's|//.*||g' "$file" | grep -qE "^\s*@Service\b" 2>/dev/null; then
                SERVICE_WITH_ANNOTATION=$((SERVICE_WITH_ANNOTATION + 1))
            fi
        done
        
        if [ "$SERVICE_WITH_ANNOTATION" -gt 0 ]; then
            pass "Application services have @Service annotation"
        else
            warn "Application services missing @Service annotation"
        fi
    else
        warn "No application services found"
    fi
else
    warn "Application layer not found"
fi

# ==========================================
# 4. ADAPTER LAYER VALIDATION
# ==========================================

echo ""
echo "── Adapter Layer ──────────────────────────────────────────"

if [ -d "$ADAPTER_PATH" ]; then
    # Check for adapter implementations
    ADAPTER_IMPL=$(find "$ADAPTER_PATH" -name "*Adapter.java" -type f 2>/dev/null | wc -l)
    if [ "$ADAPTER_IMPL" -gt 0 ]; then
        pass "Adapter implementations found ($ADAPTER_IMPL adapters)"
    fi
    
    # Check repository adapter implements domain interface
    IMPL_REPO=$(grep -r "implements.*Repository" "$ADAPTER_PATH" 2>/dev/null | wc -l || echo "0")
    if [ "$IMPL_REPO" -gt 0 ]; then
        pass "Repository adapter implements domain interface"
    else
        warn "No repository adapter found implementing domain interface"
    fi
    
    # Check for JPA entities in adapter layer (if persistence/entity exists)
    JPA_ENTITY_PATH="$ADAPTER_PATH/persistence/entity"
    if [ -d "$JPA_ENTITY_PATH" ]; then
        JPA_ENTITIES=0
        for file in $(find "$JPA_ENTITY_PATH" -name "*.java" -type f 2>/dev/null); do
            matches=$(sed 's|//.*||g' "$file" | grep -cE "^\s*@Entity\b" 2>/dev/null)
            matches=${matches:-0}
            JPA_ENTITIES=$((JPA_ENTITIES + matches))
        done
        
        if [ "$JPA_ENTITIES" -gt 0 ]; then
            pass "JPA @Entity annotations in adapter layer (correct location)"
        else
            warn "Expected JPA entities in adapter/persistence/entity"
        fi
    fi
    
    # Check for System API adapter (alternative persistence)
    SYSTEMAPI_PATH="$ADAPTER_PATH/systemapi"
    if [ -d "$SYSTEMAPI_PATH" ]; then
        pass "System API adapter found (alternative persistence)"
    fi
else
    fail "Adapter layer not found"
fi

# ==========================================
# 5. TEST VALIDATION
# ==========================================

echo ""
echo "── Test Structure ─────────────────────────────────────────"

DOMAIN_TEST_PATH="$SERVICE_DIR/src/test/java"
if [ -n "$BASE_PACKAGE_PATH" ]; then
    DOMAIN_TEST_PATH="$SERVICE_DIR/src/test/java/$BASE_PACKAGE_PATH/domain"
else
    DOMAIN_TEST_PATH=$(find "$SERVICE_DIR/src/test/java" -type d -name "domain" 2>/dev/null | head -1)
fi

if [ -d "$DOMAIN_TEST_PATH" ]; then
    # Check domain tests don't use Spring (excluding comments)
    SPRING_IN_DOMAIN_TESTS=0
    for file in $(find "$DOMAIN_TEST_PATH" -name "*.java" -type f 2>/dev/null); do
        matches=$(sed 's|//.*||g' "$file" | \
                  grep -cE "^\s*@(SpringBootTest|DataJpaTest)\b" 2>/dev/null)
        matches=${matches:-0}
        SPRING_IN_DOMAIN_TESTS=$((SPRING_IN_DOMAIN_TESTS + matches))
    done
    
    if [ "$SPRING_IN_DOMAIN_TESTS" -gt 0 ]; then
        fail "Domain tests should not use @SpringBootTest (ADR-009 violation)"
        for file in $(find "$DOMAIN_TEST_PATH" -name "*.java" -type f 2>/dev/null); do
            sed 's|//.*||g' "$file" | grep -nE "^\s*@(SpringBootTest|DataJpaTest)\b" 2>/dev/null | \
                while read -r line; do echo "  $file:$line"; done | head -3
        done
    else
        pass "Domain tests don't use Spring context (ADR-009 compliant)"
    fi
    
    # Check domain tests use Mockito (excluding comments)
    MOCKITO_IN_TESTS=0
    for file in $(find "$DOMAIN_TEST_PATH" -name "*.java" -type f 2>/dev/null); do
        if sed 's|//.*||g' "$file" | grep -qE "^\s*@(ExtendWith|Mock)\b" 2>/dev/null; then
            MOCKITO_IN_TESTS=$((MOCKITO_IN_TESTS + 1))
        fi
    done
    
    if [ "$MOCKITO_IN_TESTS" -gt 0 ]; then
        pass "Domain tests use Mockito (recommended)"
    else
        warn "Domain tests should use Mockito for mocking"
    fi
else
    warn "Domain test directory not found"
fi

# ==========================================
# SUMMARY
# ==========================================

echo ""
echo "════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo -e "  ${GREEN}✅ HEXAGONAL ARCHITECTURE: VALIDATED${NC}"
    echo "     ADR-009 Compliant"
    echo "════════════════════════════════════════════════════════════"
    exit 0
else
    echo -e "  ${RED}❌ HEXAGONAL ARCHITECTURE: VALIDATION FAILED${NC}"
    echo "     Errors: $ERRORS (ADR-009 violations)"
    echo "════════════════════════════════════════════════════════════"
    exit 1
fi
