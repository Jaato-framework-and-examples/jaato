package com.bank.customer.service.mapper;

import com.bank.customer.model.CreateCustomerRequest;
import com.bank.customer.model.Customer;
import com.bank.customer.model.CustomerStatus;
import com.bank.customer.repository.model.PartyCreateRequest;
import com.bank.customer.repository.model.PartyResponse;
import org.mapstruct.Mapper;
import org.mapstruct.Mapping;
import org.mapstruct.Named;

import java.time.LocalDate;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.UUID;

@Mapper(componentModel = "spring")
public interface CustomerMapper {

    @Mapping(source = "customerId", target = "id", qualifiedByName = "toUUID")
    @Mapping(source = "firstName", target = "firstName")
    @Mapping(source = "lastName", target = "lastName")
    @Mapping(source = "emailAddress", target = "email")
    @Mapping(source = "dateOfBirth", target = "dateOfBirth", qualifiedByName = "toDate")
    @Mapping(source = "status", target = "status", qualifiedByName = "toCustomerStatus")
    @Mapping(source = "createdAt", target = "createdAt", qualifiedByName = "toOffsetDateTime")
    @Mapping(source = "updatedAt", target = "updatedAt", qualifiedByName = "toOffsetDateTime")
    Customer toCustomer(PartyResponse partyResponse);

    @Mapping(source = "firstName", target = "firstName")
    @Mapping(source = "lastName", target = "lastName")
    @Mapping(source = "email", target = "emailAddress")
    @Mapping(source = "dateOfBirth", target = "dateOfBirth", qualifiedByName = "toStringDate")
    PartyCreateRequest toPartyCreateRequest(CreateCustomerRequest createCustomerRequest);

    @Named("toUUID")
    default UUID toUUID(String uuid) {
        if (uuid == null || uuid.isEmpty()) {
            return null;
        }
        return UUID.fromString(uuid.replaceFirst(
                "([0-9a-fA-F]{8})([0-9a-fA-F]{4})([0-9a-fA-F]{4})([0-9a-fA-F]{4})([0-9a-fA-F]{12})",
                "$1-$2-$3-$4-$5"
        ));
    }

    @Named("toDate")
    default LocalDate toDate(String date) {
        if (date == null || date.isEmpty()) {
            return null;
        }
        return LocalDate.parse(date, DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }

    @Named("toStringDate")
    default String toStringDate(LocalDate date) {
        if (date == null) {
            return null;
        }
        return date.format(DateTimeFormatter.ofPattern("yyyy-MM-dd"));
    }

    @Named("toOffsetDateTime")
    default OffsetDateTime toOffsetDateTime(String dateTime) {
        if (dateTime == null || dateTime.isEmpty()) {
            return null;
        }
        return OffsetDateTime.parse(dateTime, DateTimeFormatter.ofPattern("yyyy-MM-dd-HH.mm.ss.SSSSSS"));
    }

    @Named("toCustomerStatus")
    default CustomerStatus toCustomerStatus(String status) {
        if (status == null) {
            return null;
        }
        return switch (status) {
            case "A" -> CustomerStatus.ACTIVE;
            case "I" -> CustomerStatus.INACTIVE;
            case "B" -> CustomerStatus.BLOCKED;
            case "P" -> CustomerStatus.PENDING_VERIFICATION;
            default -> null;
        };
    }
}
