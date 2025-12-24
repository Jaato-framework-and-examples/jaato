#!/bin/bash
# =============================================================================
# MOD-016: JPA Persistence Validation Script
# Tier 3 validation for JPA persistence implementation
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

TARGET_DIR="${1:-.}"

echo "=============================================="
echo "MOD-016: JPA Persistence Validation"
echo "Target: $TARGET_DIR"
echo "=============================================="

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((ERRORS++))
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    ((WARNINGS++))
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

info() {
    echo -e "[INFO] $1"
}

echo ""
echo "--- Structural Constraints ---"

# Check 1: JPA entities NOT in domain
info "Checking JPA annotations not in domain layer..."
if grep -r "@Entity\|@Table\|@Id" "$TARGET_DIR/src/main/java" 2>/dev/null | grep -q "/domain/"; then
    error "JPA annotations found in domain layer - MUST be in adapter/persistence/"
else
    success "No JPA annotations in domain layer"
fi

# Check 2: JPA entities in correct location
info "Checking JPA entities in adapter/persistence/entity/..."
if ls "$TARGET_DIR/src/main/java"/**/adapter/persistence/entity/*.java 2>/dev/null | grep -q .; then
    success "JPA entities found in adapter/persistence/entity/"
else
    warning "No JPA entities found in adapter/persistence/entity/"
fi

# Check 3: Repository interface in domain
info "Checking repository interface in domain/repository/..."
if ls "$TARGET_DIR/src/main/java"/**/domain/repository/*Repository.java 2>/dev/null | grep -q .; then
    success "Repository interface found in domain/repository/"
else
    error "Repository interface not found in domain/repository/"
fi

echo ""
echo "--- Configuration Constraints ---"

# Check 4: ddl-auto setting
info "Checking ddl-auto configuration..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "ddl-auto:" "$TARGET_DIR/src/main/resources/application.yml"; then
        if grep -A1 "ddl-auto:" "$TARGET_DIR/src/main/resources/application.yml" | grep -q "validate"; then
            success "ddl-auto set to validate"
        else
            warning "ddl-auto should be 'validate' in production"
        fi
    else
        warning "ddl-auto not configured"
    fi
fi

# Check 5: OSIV disabled
info "Checking open-in-view setting..."
if [ -f "$TARGET_DIR/src/main/resources/application.yml" ]; then
    if grep -q "open-in-view: false" "$TARGET_DIR/src/main/resources/application.yml"; then
        success "Open Session In View disabled"
    else
        warning "open-in-view should be false for better performance"
    fi
fi

echo ""
echo "--- Dependency Constraints ---"

# Check 6: Spring Data JPA dependency
info "Checking Spring Data JPA dependency..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "spring-boot-starter-data-jpa" "$TARGET_DIR/pom.xml"; then
        success "Spring Data JPA dependency found"
    else
        error "spring-boot-starter-data-jpa dependency not found"
    fi
fi

# Check 7: Database driver
info "Checking database driver..."
if [ -f "$TARGET_DIR/pom.xml" ]; then
    if grep -q "postgresql\|mysql\|h2" "$TARGET_DIR/pom.xml"; then
        success "Database driver found"
    else
        warning "No database driver found in pom.xml"
    fi
fi

echo ""
echo "--- Testing Constraints ---"

# Check 8: Adapter tests exist
info "Checking persistence adapter tests..."
if find "$TARGET_DIR/src/test/java" -name "*PersistenceAdapter*Test.java" 2>/dev/null | grep -q .; then
    success "Persistence adapter tests found"
else
    warning "No persistence adapter tests found"
fi

echo ""
echo "=============================================="
echo "Validation Summary"
echo "=============================================="
echo -e "Errors:   ${RED}$ERRORS${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}VALIDATION FAILED${NC}"
    exit 1
else
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}VALIDATION PASSED WITH WARNINGS${NC}"
    else
        echo -e "${GREEN}VALIDATION PASSED${NC}"
    fi
    exit 0
fi
