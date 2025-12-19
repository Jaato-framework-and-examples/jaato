package com.bank.customer.controller;

import com.bank.customer.model.CreateCustomerRequest;
import com.bank.customer.model.Customer;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.enums.ParameterIn;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.Valid;
import java.util.UUID;

@Validated
@Tag(name = "customers", description = "Customer operations")
public interface CustomerApi {

    @Operation(
        summary = "Create a new customer",
        description = "Creates a new customer record in the system",
        operationId = "createCustomer",
        responses = {
            @ApiResponse(responseCode = "201", description = "Customer created successfully", content = {
                @Content(mediaType = "application/json", schema = @Schema(implementation = Customer.class))
            }),
            @ApiResponse(responseCode = "400", description = "Invalid request parameters"),
            @ApiResponse(responseCode = "409", description = "Resource already exists"),
            @ApiResponse(responseCode = "500", description = "Internal server error"),
            @ApiResponse(responseCode = "503", description = "Service temporarily unavailable")
        }
    )
    @PostMapping(value = "/api/v1/customers",
        produces = {"application/json"},
        consumes = {"application/json"})
    ResponseEntity<Customer> createCustomer(
        @Parameter(description = "Customer data to create", required = true)
        @Valid @RequestBody CreateCustomerRequest createCustomerRequest
    );

    @Operation(
        summary = "Get customer by ID",
        description = "Retrieves a customer by their unique identifier",
        operationId = "getCustomerById",
        parameters = {
            @Parameter(name = "customerId", in = ParameterIn.PATH, description = "Unique customer identifier (UUID)", required = true),
            @Parameter(name = "X-Correlation-ID", in = ParameterIn.HEADER, description = "Correlation ID for distributed tracing")
        },
        responses = {
            @ApiResponse(responseCode = "200", description = "Customer found", content = {
                @Content(mediaType = "application/json", schema = @Schema(implementation = Customer.class))
            }),
            @ApiResponse(responseCode = "404", description = "Resource not found"),
            @ApiResponse(responseCode = "500", description = "Internal server error"),
            @ApiResponse(responseCode = "503", description = "Service temporarily unavailable")
        }
    )
    @GetMapping(value = "/api/v1/customers/{customerId}",
        produces = {"application/json"})
    ResponseEntity<Customer> getCustomerById(
        @PathVariable("customerId") UUID customerId,
        @RequestHeader(value = "X-Correlation-ID", required = false) UUID correlationId
    );

}
