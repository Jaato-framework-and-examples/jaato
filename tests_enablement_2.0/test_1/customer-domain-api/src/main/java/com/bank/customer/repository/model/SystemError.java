package com.bank.customer.repository.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class SystemError {

    @JsonProperty("SYS_RC")
    private String returnCode;

    @JsonProperty("SYS_MSG")
    private String message;

    @JsonProperty("SYS_ABEND_CD")
    private String abendCode;

    @JsonProperty("SYS_TIMESTAMP")
    private String timestamp;
}
