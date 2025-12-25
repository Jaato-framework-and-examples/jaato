# Validation: mod-code-018-api-integration-rest-java-spring

## Overview

Tier-3 validation for REST API integration clients.

## Checks

### integration-check.sh

| Check | Severity | Description |
|-------|----------|-------------|
| Correlation headers | ERROR | X-Correlation-ID must be propagated |
| Base URL externalized | ERROR | Base URL must use environment variable |
| Source system header | WARNING | X-Source-System should be set |
| Error handling | WARNING | IntegrationException should be used |
| Logging | WARNING | Debug logging should be present |

## Usage

```bash
./integration-check.sh /path/to/generated/code
```

## Exit Codes

- 0: All checks passed
- 1: Errors found (blocking)
- 2: Warnings only (non-blocking)
