// Template: Response-hateoas.java.tpl
// Output: {{basePackage}}/application/dto/{{Entity}}Response.java
// Purpose: Response DTO for {{Entity}} with HATEOAS support
// VARIANT: hateoas (used when features.hateoas = true)
// NOTE: Requires class instead of record because of extends RepresentationModel

package {{basePackage}}.application.dto;

import {{basePackage}}.domain.model.{{Entity}};
import org.springframework.hateoas.RepresentationModel;
import org.springframework.hateoas.server.core.Relation;
import java.time.Instant;

/**
 * Response DTO for {{Entity}} with HATEOAS hypermedia links.
 * 
 * Uses class (not record) because HATEOAS requires extending RepresentationModel.
 * 
 * @generated {{skillId}} v{{skillVersion}}
 * @module mod-code-015-hexagonal-base-java-spring
 * @variant hateoas
 */
@Relation(collectionRelation = "{{entityPlural}}", itemRelation = "{{entity}}")
public class {{Entity}}Response extends RepresentationModel<{{Entity}}Response> {
    
    {{#fields}}
    private {{type}} {{fieldName}};
    {{/fields}}
    
    /**
     * Default constructor for serialization.
     */
    public {{Entity}}Response() {}
    
    /**
     * All-args constructor.
     */
    public {{Entity}}Response({{#fields}}{{type}} {{fieldName}}{{^last}}, {{/last}}{{/fields}}) {
        {{#fields}}
        this.{{fieldName}} = {{fieldName}};
        {{/fields}}
    }
    
    /**
     * Factory method to create response from domain entity.
     */
    public static {{Entity}}Response from({{Entity}} entity) {
        if (entity == null) {
            return null;
        }
        return new {{Entity}}Response(
            {{#fields}}
            entity.get{{FieldName}}(){{^last}},{{/last}}
            {{/fields}}
        );
    }
    
    // ========== Getters ==========
    
    {{#fields}}
    public {{type}} get{{FieldName}}() {
        return {{fieldName}};
    }
    
    {{/fields}}
    
    // ========== Setters ==========
    
    {{#fields}}
    public void set{{FieldName}}({{type}} {{fieldName}}) {
        this.{{fieldName}} = {{fieldName}};
    }
    
    {{/fields}}
}
