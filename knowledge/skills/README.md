# Skills

> Executable units that agents orchestrate

## Purpose

Skills are the **primary executable units** in Enablement 2.0. Each skill:
- Has a clear input/output specification
- References modules for template processing
- Follows a domain-specific execution flow
- Includes prompts for agent orchestration

## Structure (v1.4)

Skills are organized by domain and, for CODE domain, by architectural layer:

```
skills/
├── README.md           # This file
├── code/
│   ├── README.md      # CODE domain overview
│   ├── soe/           # System of Engagement (frontend)
│   ├── soi/           # System of Integration (microservices)
│   │   ├── skill-001-circuit-breaker-java-resilience4j/
│   │   └── skill-020-microservice-java-spring/
│   └── sor/           # System of Record (mainframe)
├── design/            # DESIGN domain (flat)
├── qa/                # QA domain (flat)
└── governance/        # GOVERNANCE domain (flat)
```

## Layer Taxonomy (CODE Domain)

| Layer | Name | Technologies |
|-------|------|--------------|
| `soe` | System of Engagement | Angular, React, Vue, Microfrontends |
| `soi` | System of Integration | Java Spring, Node.js, Quarkus |
| `sor` | System of Record | COBOL, CICS, DB2, JCL |

## Skill Files

| File | Purpose | Consumer |
|------|---------|----------|
| **SKILL.md** | Complete specification | Agent execution |
| **OVERVIEW.md** | Lightweight summary | Agent discovery ⭐ |
| **prompts/** | Execution instructions | Agent orchestrator |
| **validation/** | Validation orchestrator | Post-execution |

## Naming Convention

```
skills/{domain}/{layer}/skill-{NNN}-{name}/
```

| Component | Example |
|-----------|---------|
| `domain` | code, design, qa, governance |
| `layer` | soe, soi, sor (CODE only) |
| `NNN` | 001, 020 (3-digit unique ID) |
| `name` | circuit-breaker-java-resilience4j, microservice-java-spring |

## Current Skills

| Location | Skill | Flow | Purpose |
|----------|-------|------|---------|
| code/soi | skill-001-circuit-breaker-java-resilience4j | ADD | Add circuit breaker to existing code |
| code/soi | skill-020-microservice-java-spring | GENERATE | Generate new microservice project |

## Discovery

Skills are indexed in `runtime/discovery/skill-index.yaml` for efficient discovery.

Discovery flow:
1. Identify domain (CODE, DESIGN, QA, GOVERNANCE)
2. Identify layer (for CODE: SoE, SoI, SoR)
3. Query index for filtered candidates
4. Read OVERVIEW.md of candidates
5. Select best match

## Execution Flow

Each skill follows a flow defined in `/runtime/flows/{domain}/{TYPE}.md`:

1. **Discovery**: OVERVIEW.md + skill-index.yaml help select the right skill
2. **Load**: SKILL.md provides full specification
3. **Prompts**: prompts/ provides agent instructions
4. **Flow**: runtime/flows/ provides execution steps
5. **Modules**: /modules/ provides templates
6. **Validation**: validators verify output

## Related

- Modules (templates): `/modules/`
- Flows (execution): `/runtime/flows/`
- Skill Index: `/runtime/discovery/skill-index.yaml`
- Model (authoring guide): `/model/standards/authoring/SKILL.md`
- Standards: `/model/standards/ASSET-STANDARDS-v1.4.md`
