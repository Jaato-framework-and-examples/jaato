package com.bank.customer.repository;

import com.bank.customer.repository.model.PartyCreateRequest;
import com.bank.customer.repository.model.PartyResponse;

import java.util.Optional;
import java.util.UUID;

public interface PartiesRepository {

    Optional<PartyResponse> findPartyById(UUID partyId);

    PartyResponse createParty(PartyCreateRequest partyCreateRequest);

}
