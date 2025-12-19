package com.bank.customer.repository.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class PartyResponse {

    @JsonProperty("CUST_ID")
    private String customerId;

    @JsonProperty("CUST_FNAME")
    private String firstName;

    @JsonProperty("CUST_LNAME")
    private String lastName;

    @JsonProperty("CUST_EMAIL_ADDR")
    private String emailAddress;

    @JsonProperty("CUST_DOB")
    private String dateOfBirth;

    @JsonProperty("CUST_STATUS")
    private String status;

    @JsonProperty("CUST_CRT_TS")
    private String createdAt;

    @JsonProperty("CUST_UPD_TS")
    private String updatedAt;

    @JsonProperty("SYS_RC")
    private String systemReturnCode;

    @JsonProperty("SYS_MSG")
    private String systemMessage;
}
