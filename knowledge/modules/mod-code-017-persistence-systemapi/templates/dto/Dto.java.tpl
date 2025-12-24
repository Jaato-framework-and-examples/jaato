// Template: Dto.java.tpl
// Output: {{basePackage}}/adapter/out/persistence/dto/{{Entity}}SystemApiResponse.java
// Purpose: Response DTO for System API communication
// STRUCTURE: MUST be record (per DETERMINISM-RULES.md)
// NOTE: NO Lombok - records are preferred for DTOs

package {{basePackage}}.adapter.out.persistence.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Response DTO from {{SystemApiName}} System API.
 * 
 * Maps to external contract (e.g., COBOL copybook).
 * Uses record for immutability.
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-017-persistence-systemapi
 */
public record {{Entity}}SystemApiResponse(
{{#fields}}
    @JsonProperty("{{jsonField}}") {{type}} {{fieldName}}{{^last}},{{/last}}
{{/fields}}
) {}
