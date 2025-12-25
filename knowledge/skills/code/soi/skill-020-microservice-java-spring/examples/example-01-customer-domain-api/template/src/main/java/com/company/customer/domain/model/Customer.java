package com.company.customer.domain.model;

import java.time.LocalDateTime;
import java.util.Objects;

/**
 * Domain entity representing a Customer.
 * Pure POJO - NO framework annotations allowed.
 */
public class Customer {
    
    private final CustomerId id;
    private String name;
    private String email;
    private int age;
    private String tier;
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    
    /**
     * Factory method to create a new Customer from registration data.
     */
    public static Customer create(CustomerRegistration registration) {
        Objects.requireNonNull(registration, "registration must not be null");
        
        return new Customer(
            CustomerId.generate(),
            registration.name(),
            registration.email(),
            registration.age(),
            null,
            LocalDateTime.now()
        );
    }
    
    /**
     * Reconstitution constructor for loading from persistence.
     */
    public Customer(CustomerId id, String name, String email, int age, 
                    String tier, LocalDateTime createdAt) {
        this.id = Objects.requireNonNull(id, "id must not be null");
        this.name = Objects.requireNonNull(name, "name must not be null");
        this.email = Objects.requireNonNull(email, "email must not be null");
        this.age = age;
        this.tier = tier;
        this.createdAt = createdAt;
        this.updatedAt = createdAt;
    }
    
    // ============ Business Methods ============
    
    /**
     * Updates customer profile information.
     */
    public void updateProfile(String name, String email, int age) {
        this.name = Objects.requireNonNull(name, "name must not be null");
        this.email = Objects.requireNonNull(email, "email must not be null");
        this.age = age;
        this.updatedAt = LocalDateTime.now();
    }
    
    /**
     * Assigns a loyalty tier to the customer.
     */
    public void assignTier(String tier) {
        this.tier = tier;
        this.updatedAt = LocalDateTime.now();
    }
    
    // ============ Getters ============
    
    public CustomerId getId() {
        return id;
    }
    
    public String getName() {
        return name;
    }
    
    public String getEmail() {
        return email;
    }
    
    public int getAge() {
        return age;
    }
    
    public String getTier() {
        return tier;
    }
    
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }
    
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }
    
    // ============ Equality based on ID ============
    
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Customer customer = (Customer) o;
        return Objects.equals(id, customer.id);
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
    
    @Override
    public String toString() {
        return "Customer{" +
                "id=" + id +
                ", name='" + name + '\'' +
                ", email='" + email + '\'' +
                ", age=" + age +
                ", tier='" + tier + '\'' +
                '}';
    }
}
