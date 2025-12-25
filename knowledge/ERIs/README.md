# Enterprise Reference Implementations (ERIs)

**Version:** 1.3  
**Last Updated:** 2025-12-01

---

## Overview

ERIs are **complete, production-ready reference implementations** of patterns and standards defined in ADRs. They serve as:

- Starting points for code generation skills
- Reference for developers understanding implementations
- Validation targets for compliance checking
- Training resources

**Key Requirement:** All ERIs MUST include a machine-readable `eri_constraints` annex.

---

## Naming Convention

```
eri-{domain}-XXX-{pattern}-{framework}-{library}/
└── ERI.md
```

| Component | Description | Values |
|-----------|-------------|--------|
| `{domain}` | Primary domain | `code`, `design`, `qa`, `gov` |
| `XXX` | Sequential number within domain | `001`-`999` |
| `{pattern}` | Pattern being implemented | `hexagonal-light`, `circuit-breaker`, etc. |
| `{framework}` | Technology framework | `java`, `nodejs`, `python`, etc. |
| `{library}` | Specific library (if applicable) | `spring`, `resilience4j`, etc. |

---

## Current ERIs

### CODE Domain

| ERI | Pattern | Framework | ADR | Status |
|-----|---------|-----------|-----|--------|
| [eri-code-001-hexagonal-light-java-spring](./eri-code-001-hexagonal-light-java-spring/) | Hexagonal Architecture | Java/Spring | ADR-009 | ✅ Active |
| [eri-code-008-circuit-breaker-java-resilience4j](./eri-code-008-circuit-breaker-java-resilience4j/) | Circuit Breaker | Java/Resilience4j | ADR-004 | ✅ Active |
| [eri-code-009-retry-java-resilience4j](./eri-code-009-retry-java-resilience4j/) | Retry | Java/Resilience4j | ADR-004 | ✅ Active |
| [eri-code-010-timeout-java-resilience4j](./eri-code-010-timeout-java-resilience4j/) | Timeout | Java/Resilience4j | ADR-004 | ✅ Active |
| [eri-code-011-rate-limiter-java-resilience4j](./eri-code-011-rate-limiter-java-resilience4j/) | Rate Limiter | Java/Resilience4j | ADR-004 | ✅ Active |
| [eri-code-012-persistence-patterns-java-spring](./eri-code-012-persistence-patterns-java-spring/) | Persistence (JPA + System API) | Java/Spring | ADR-011 | ✅ Active |

### Other Domains

| Domain | Status |
|--------|--------|
| DESIGN | ⏳ No ERIs yet |
| QA | ⏳ No ERIs yet |
| GOV | ⏳ No ERIs yet |

---

## ERI → Module → Skill Chain

ERIs are part of the knowledge chain:

```
ADR (Decision)
  ↓ implements
ERI (Reference Implementation)
  ↓ encapsulates
Module (Reusable Template)
  ↓ orchestrates
Skill (Executable Generation)
```

---

## Cross-Domain Usage

ERIs have a **primary domain** but may be used by other domains:

| ERI | Primary Domain | Secondary Usage |
|-----|----------------|-----------------|
| eri-code-001-hexagonal | CODE | QA (compliance validation) |
| eri-code-008-circuit-breaker | CODE | QA (resilience validation) |
| eri-code-009-retry | CODE | QA (resilience validation) |
| eri-code-010-timeout | CODE | QA (resilience validation) |
| eri-code-011-rate-limiter | CODE | QA (resilience validation) |
| eri-code-012-persistence | CODE | QA (persistence validation) |

The `cross_domain_usage` field in ERI metadata documents secondary uses.

---

## Related

- `model/standards/ASSET-STANDARDS-v1.4.md` - ERI structure definition
- `model/standards/authoring/ERI.md` - How to create ERIs
- `ADRs/` - Architectural decisions ERIs implement
- `skills/modules/` - Modules derived from ERIs

---

**Last Updated:** 2025-12-01
