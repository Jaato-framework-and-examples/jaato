// Template: EntityModelAssembler.java.tpl
// Output: {{basePackagePath}}/adapter/in/rest/assembler/{{entityName}}ModelAssembler.java
// Purpose: Converts {{entityName}} to HATEOAS-enabled responses

package {{basePackage}}.adapter.in.rest.assembler;

import {{basePackage}}.adapter.in.rest.{{entityName}}Controller;
import {{basePackage}}.adapter.in.rest.dto.{{entityName}}Response;
import {{basePackage}}.domain.model.{{entityName}};
import org.springframework.hateoas.server.mvc.RepresentationModelAssemblerSupport;
import org.springframework.stereotype.Component;

import static org.springframework.hateoas.server.mvc.WebMvcLinkBuilder.*;

/**
 * Assembler that converts {{entityName}} domain entities to {{entityName}}Response DTOs
 * with HATEOAS links per ADR-001.
 */
@Component
public class {{entityName}}ModelAssembler 
        extends RepresentationModelAssemblerSupport<{{entityName}}, {{entityName}}Response> {
    
    public {{entityName}}ModelAssembler() {
        super({{entityName}}Controller.class, {{entityName}}Response.class);
    }
    
    @Override
    public {{entityName}}Response toModel({{entityName}} entity) {
        {{entityName}}Response response = mapToResponse(entity);
        
        // Self link
        response.add(linkTo(methodOn({{entityName}}Controller.class)
            .getById(entity.getId().toString(), null))
            .withSelfRel());
        
        // Collection link
        response.add(linkTo(methodOn({{entityName}}Controller.class)
            .list(null, null, null, null))
            .withRel("collection"));
        
        // Add state-specific action links
        addActionLinks(response, entity);
        
        return response;
    }
    
    private {{entityName}}Response mapToResponse({{entityName}} entity) {
        // Map entity fields to response DTO
        return new {{entityName}}Response(
            entity.getId().toString()
            // Add other field mappings here
        );
    }
    
    private void addActionLinks({{entityName}}Response response, {{entityName}} entity) {
        // Add conditional action links based on entity state
        // Example:
        // if ("ACTIVE".equals(entity.getStatus().name())) {
        //     response.add(linkTo(methodOn({{entityName}}Controller.class)
        //         .deactivate(entity.getId().toString(), null))
        //         .withRel("deactivate"));
        // }
    }
}
