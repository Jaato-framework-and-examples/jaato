package com.bank.customer.repository;

import com.bank.customer.repository.model.PartyCreateRequest;
import com.bank.customer.repository.model.PartyResponse;
import io.github.resilience4j.circuitbreaker.annotation.CircuitBreaker;
import io.github.resilience4j.retry.annotation.Retry;
import io.github.resilience4j.timelimiter.annotation.TimeLimiter;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.RestClient;

import java.util.Optional;
import java.util.UUID;

@Repository
public class PartiesRepositoryImpl implements PartiesRepository {

    private final RestClient restClient;

    public PartiesRepositoryImpl(RestClient restClient) {
        this.restClient = restClient;
    }

    @Override
    @CircuitBreaker(name = "parties")
    @Retry(name = "parties")
    @TimeLimiter(name = "parties")
    public Optional<PartyResponse> findPartyById(UUID partyId) {
        return Optional.ofNullable(restClient.get()
                .uri("/parties/{partyId}", partyId.toString().replace("-", ""))
                .retrieve()
                .body(PartyResponse.class));
    }

    @Override
    @CircuitBreaker(name = "parties")
    @Retry(name = "parties")
    @TimeLimiter(name = "parties")
    public PartyResponse createParty(PartyCreateRequest partyCreateRequest) {
        return restClient.post()
                .uri("/parties")
                .body(partyCreateRequest)
                .retrieve()
                .body(PartyResponse.class);
    }
}
