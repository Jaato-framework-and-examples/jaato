// Template: SystemApiMapper.java.tpl
// Output: {{basePackage}}/adapter/out/persistence/mapper/{{Entity}}SystemApiMapper.java
// Purpose: Maps between domain entities and System API DTOs
// IMPORTANT: Code mapping (enum ↔ external codes) is done HERE, not in enums

package {{basePackage}}.adapter.out.persistence.mapper;

import {{basePackage}}.adapter.out.persistence.dto.{{Entity}}SystemApiResponse;
import {{basePackage}}.adapter.out.persistence.dto.{{Entity}}SystemApiRequest;
import {{basePackage}}.domain.model.*;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

/**
 * Mapper between {{Entity}} domain model and System API DTOs.
 * 
 * Handles all transformations including:
 * - UUID ↔ Mainframe ID format
 * - Proper case ↔ Uppercase
 * - ISO dates ↔ System date formats
 * - Enum ↔ Single character codes (IMPORTANT: mapping here, NOT in enum)
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-017-persistence-systemapi
 */
@Component
public class {{Entity}}SystemApiMapper {
    
    private static final DateTimeFormatter SYSTEM_DATE_FORMAT = 
        DateTimeFormatter.ofPattern("yyyy-MM-dd");
    
    /**
     * Maps System API response to domain entity.
     */
    public {{Entity}} toDomain({{Entity}}SystemApiResponse response) {
        if (response == null) {
            return null;
        }
        
        return new {{Entity}}(
            toEntityId(response.{{idField}}()),
            {{#fields}}
            {{#if isString}}toProperCase(response.{{fieldName}}()){{/if}}{{#if isEnum}}to{{EnumType}}(response.{{fieldName}}()){{/if}}{{#if isDate}}parseDate(response.{{fieldName}}()){{/if}}{{#if isOther}}response.{{fieldName}}(){{/if}}{{^last}},{{/last}}
            {{/fields}}
        );
    }
    
    /**
     * Maps domain entity to System API request.
     */
    public {{Entity}}SystemApiRequest toRequest({{Entity}} entity) {
        if (entity == null) {
            return null;
        }
        
        return new {{Entity}}SystemApiRequest(
            {{#fields}}
            {{#if isString}}entity.get{{FieldName}}().toUpperCase(){{/if}}{{#if isEnum}}toCode(entity.get{{FieldName}}()){{/if}}{{#if isDate}}formatDate(entity.get{{FieldName}}()){{/if}}{{#if isOther}}entity.get{{FieldName}}(){{/if}}{{^last}},{{/last}}
            {{/fields}}
        );
    }
    
    /**
     * Converts mainframe ID to domain EntityId.
     */
    public {{Entity}}Id toEntityId(String mainframeId) {
        if (mainframeId == null || mainframeId.isBlank()) {
            return null;
        }
        // Mainframe IDs are uppercase without hyphens
        String formatted = mainframeId.toLowerCase();
        String uuid = String.format("%s-%s-%s-%s-%s",
            formatted.substring(0, 8),
            formatted.substring(8, 12),
            formatted.substring(12, 16),
            formatted.substring(16, 20),
            formatted.substring(20));
        return {{Entity}}Id.of(UUID.fromString(uuid));
    }
    
    /**
     * Converts domain EntityId to mainframe format.
     */
    public String toMainframeId({{Entity}}Id id) {
        if (id == null) {
            return null;
        }
        return id.value().toString().replace("-", "").toUpperCase();
    }
    
    // ========== Status Code Mapping (Enum ↔ Code) ==========
    // IMPORTANT: This is where external code mapping belongs, NOT in the enum
    
    {{#statusEnum}}
    private {{StatusEnum}} to{{StatusEnum}}(String code) {
        if (code == null) {
            return null;
        }
        return switch (code) {
            {{#statusMappings}}
            case "{{code}}" -> {{StatusEnum}}.{{value}};
            {{/statusMappings}}
            default -> throw new IllegalArgumentException("Unknown {{StatusEnum}} code: " + code);
        };
    }
    
    private String toCode({{StatusEnum}} status) {
        if (status == null) {
            return null;
        }
        return switch (status) {
            {{#statusMappings}}
            case {{value}} -> "{{code}}";
            {{/statusMappings}}
        };
    }
    {{/statusEnum}}
    
    // ========== String Conversion ==========
    
    private String toProperCase(String value) {
        if (value == null || value.isEmpty()) {
            return value;
        }
        return value.substring(0, 1).toUpperCase() + 
               value.substring(1).toLowerCase();
    }
    
    // ========== Date Conversion ==========
    
    private LocalDate parseDate(String systemDate) {
        if (systemDate == null || systemDate.isBlank()) {
            return null;
        }
        return LocalDate.parse(systemDate.substring(0, 10), SYSTEM_DATE_FORMAT);
    }
    
    private String formatDate(LocalDate date) {
        if (date == null) {
            return "";
        }
        return date.format(SYSTEM_DATE_FORMAT);
    }
}
