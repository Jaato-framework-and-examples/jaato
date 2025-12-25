package com.company.customer.domain.service;

import com.company.customer.domain.exception.CustomerNotFoundException;
import com.company.customer.domain.exception.DuplicateEmailException;
import com.company.customer.domain.model.Customer;
import com.company.customer.domain.model.CustomerId;
import com.company.customer.domain.model.CustomerRegistration;
import com.company.customer.domain.repository.CustomerRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Optional;

import static org.assertj.core.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

/**
 * Unit tests for CustomerDomainService.
 * 
 * IMPORTANT: Uses @ExtendWith(MockitoExtension.class), NOT @SpringBootTest.
 * Domain tests should run without Spring context for fast feedback.
 */
@ExtendWith(MockitoExtension.class)
class CustomerDomainServiceTest {
    
    @Mock
    private CustomerRepository repository;
    
    private CustomerDomainService domainService;
    
    @BeforeEach
    void setUp() {
        domainService = new CustomerDomainService(repository);
    }
    
    @Nested
    @DisplayName("registerCustomer")
    class RegisterCustomerTests {
        
        @Test
        @DisplayName("should create customer when email is unique")
        void registerCustomer_WithUniqueEmail_CreatesCustomer() {
            // Given
            var registration = new CustomerRegistration("John Doe", "john@example.com", 30);
            when(repository.existsByEmail("john@example.com")).thenReturn(false);
            when(repository.save(any(Customer.class))).thenAnswer(inv -> inv.getArgument(0));
            
            // When
            Customer result = domainService.registerCustomer(registration);
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getName()).isEqualTo("John Doe");
            assertThat(result.getEmail()).isEqualTo("john@example.com");
            assertThat(result.getAge()).isEqualTo(30);
            assertThat(result.getId()).isNotNull();
            
            verify(repository).existsByEmail("john@example.com");
            verify(repository).save(any(Customer.class));
        }
        
        @Test
        @DisplayName("should throw exception when email already exists")
        void registerCustomer_WithDuplicateEmail_ThrowsException() {
            // Given
            var registration = new CustomerRegistration("John Doe", "existing@example.com", 30);
            when(repository.existsByEmail("existing@example.com")).thenReturn(true);
            
            // When/Then
            assertThatThrownBy(() -> domainService.registerCustomer(registration))
                    .isInstanceOf(DuplicateEmailException.class)
                    .hasMessageContaining("existing@example.com");
            
            verify(repository).existsByEmail("existing@example.com");
            verify(repository, never()).save(any());
        }
    }
    
    @Nested
    @DisplayName("getCustomer")
    class GetCustomerTests {
        
        @Test
        @DisplayName("should return customer when exists")
        void getCustomer_WhenExists_ReturnsCustomer() {
            // Given
            var customerId = CustomerId.generate();
            var customer = Customer.create(new CustomerRegistration("Jane", "jane@example.com", 25));
            when(repository.findById(customerId)).thenReturn(Optional.of(customer));
            
            // When
            Customer result = domainService.getCustomer(customerId);
            
            // Then
            assertThat(result).isNotNull();
            assertThat(result.getName()).isEqualTo("Jane");
            verify(repository).findById(customerId);
        }
        
        @Test
        @DisplayName("should throw exception when customer not found")
        void getCustomer_WhenNotFound_ThrowsException() {
            // Given
            var customerId = CustomerId.generate();
            when(repository.findById(customerId)).thenReturn(Optional.empty());
            
            // When/Then
            assertThatThrownBy(() -> domainService.getCustomer(customerId))
                    .isInstanceOf(CustomerNotFoundException.class)
                    .hasMessageContaining(customerId.toString());
        }
    }
    
    @Nested
    @DisplayName("deleteCustomer")
    class DeleteCustomerTests {
        
        @Test
        @DisplayName("should delete customer when exists")
        void deleteCustomer_WhenExists_DeletesSuccessfully() {
            // Given
            var customerId = CustomerId.generate();
            when(repository.existsById(customerId)).thenReturn(true);
            
            // When
            domainService.deleteCustomer(customerId);
            
            // Then
            verify(repository).existsById(customerId);
            verify(repository).deleteById(customerId);
        }
        
        @Test
        @DisplayName("should throw exception when customer not found")
        void deleteCustomer_WhenNotFound_ThrowsException() {
            // Given
            var customerId = CustomerId.generate();
            when(repository.existsById(customerId)).thenReturn(false);
            
            // When/Then
            assertThatThrownBy(() -> domainService.deleteCustomer(customerId))
                    .isInstanceOf(CustomerNotFoundException.class);
            
            verify(repository, never()).deleteById(any());
        }
    }
}
