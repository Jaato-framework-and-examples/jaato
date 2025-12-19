package com.bank.customer.repository.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class PartyCreateRequest {

    @JsonProperty("CUST_FNAME")
    private String firstName;

    @JsonProperty("CUST_LNAME")
    private String lastName;

    @JsonProperty("CUST_EMAIL_ADDR")
    private String emailAddress;

    @JsonProperty("CUST_DOB")
    private String dateOfBirth;
}
