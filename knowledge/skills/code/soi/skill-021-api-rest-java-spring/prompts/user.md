# User Prompt Template: REST API Generation

## Request

Generate a REST API with the following specification:

```json
{{generation-request}}
```

## Instructions

1. **Read the specification** and identify:
   - API layer (experience, composable, domain, system)
   - Entities to generate
   - Features to enable

2. **Consult modules** based on layer:
   - Always: mod-015 (hexagonal), mod-019 (pagination, HATEOAS)
   - Domain layer: mod-020 (compensation)
   - If resilience enabled: mod-001 to 004
   - Based on persistence type: mod-016 or mod-017

3. **Generate the complete API** including:
   - All Java source files with traceability headers
   - Configuration files (application.yml, pom.xml)
   - OpenAPI specification
   - Unit and integration tests
   - Traceability manifest

4. **Apply layer-specific patterns**:
   - Experience: Full HATEOAS, caching
   - Composable: Pagination only
   - Domain: Full HATEOAS + Compensation
   - System: Simple pagination

5. **Validate output** against:
   - Tier 1: Structure, naming, traceability
   - Tier 2: Java compilation, OpenAPI validity
   - Tier 3: Module-specific requirements

## Expected Output

A complete, production-ready REST API project that:
- Compiles without errors
- Passes all validation tiers
- Follows Hexagonal Light architecture
- Implements ADR-001 pagination and HATEOAS standards
- Includes compensation (Domain layer only)
- Has complete traceability

## Output Format

Provide the generated code organized by file path, with each file including:
1. Traceability header
2. Complete, compilable code
3. Appropriate comments

End with the manifest.json containing full traceability information.
