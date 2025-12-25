# Tier 3 Validation: mod-code-019-api-public-exposure

## Overview

Validates that generated code complies with ERI-014 (API Public Exposure) constraints.

## Checks

| Script | Severity | Description |
|--------|----------|-------------|
| `pagination-check.sh` | ERROR | Validates PageResponse structure and metadata |
| `hateoas-check.sh` | ERROR | Validates HATEOAS assembler and links |
| `config-check.sh` | WARNING | Validates pagination configuration values |

## ERI Constraint Mapping

| ERI Constraint | Script | Check |
|----------------|--------|-------|
| page-response-structure | pagination-check.sh | PageResponse.java exists with content, page, _links |
| page-metadata-fields | pagination-check.sh | PageMetadata has number, size, totalElements, totalPages |
| hateoas-self-link | hateoas-check.sh | ModelAssembler adds withSelfRel() |
| model-assembler-exists | hateoas-check.sh | {Entity}ModelAssembler.java exists |
| default-page-size | config-check.sh | application.yml has default-page-size: 20 |
| max-page-size | config-check.sh | application.yml has max-page-size: 100 |
| zero-indexed-pagination | config-check.sh | one-indexed-parameters: false |

## Usage

```bash
# Run all checks
./pagination-check.sh /path/to/service
./hateoas-check.sh /path/to/service com/company/customer
./config-check.sh /path/to/service

# Exit codes
# 0 = All checks passed
# N = Number of failed checks
```

## Integration with Skills

Skills should run these validators after code generation:

```bash
# In skill's generate.sh or validate.sh
source "$MODULE_DIR/validation/pagination-check.sh" "$OUTPUT_DIR"
source "$MODULE_DIR/validation/hateoas-check.sh" "$OUTPUT_DIR" "$PACKAGE_PATH"
source "$MODULE_DIR/validation/config-check.sh" "$OUTPUT_DIR"
```
