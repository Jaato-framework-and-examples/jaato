# Tier 3 Validation: mod-code-020-compensation

## Overview

Validates that generated code complies with ERI-015 (Distributed Transactions) constraints.

## Checks

| Script | Severity | Description |
|--------|----------|-------------|
| `compensation-interface-check.sh` | ERROR | Validates Compensable interface and implementation |
| `compensation-endpoint-check.sh` | ERROR | Validates /compensate endpoint exists |
| `transaction-log-check.sh` | WARNING | Validates TransactionLog entity and repository |

## ERI Constraint Mapping

| ERI Constraint | Script | Check |
|----------------|--------|-------|
| compensable-interface-implemented | compensation-interface-check.sh | Service implements Compensable<T> |
| compensate-method-exists | compensation-interface-check.sh | compensate() method exists |
| compensation-endpoint-exists | compensation-endpoint-check.sh | @PostMapping("/compensate") exists |
| correlation-id-header-required | compensation-endpoint-check.sh | @RequestHeader("X-Correlation-ID") |
| transaction-log-entity-exists | transaction-log-check.sh | TransactionLog.java exists |
| idempotency-test | compensation-interface-check.sh | Tests verify ALREADY_COMPENSATED |

## Usage

```bash
# Run all checks
./compensation-interface-check.sh /path/to/service
./compensation-endpoint-check.sh /path/to/service
./transaction-log-check.sh /path/to/service

# Exit codes
# 0 = All checks passed
# N = Number of failed checks
```

## Integration with Skills

Skills should run these validators after code generation:

```bash
# In skill's generate.sh or validate.sh
source "$MODULE_DIR/validation/compensation-interface-check.sh" "$OUTPUT_DIR"
source "$MODULE_DIR/validation/compensation-endpoint-check.sh" "$OUTPUT_DIR"
source "$MODULE_DIR/validation/transaction-log-check.sh" "$OUTPUT_DIR"
```
