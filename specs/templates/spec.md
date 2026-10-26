# Specification: <Capability Title>

## Capability: `<capability-slug>`

### 1. Functional Requirements

- **REQ-<DOMAIN>-1 (<Short Name>)**: The system MUST <exact verifiable requirement>.
- **REQ-<DOMAIN>-2 (<Short Name>)**: When <condition>, the system MUST <exact response>.

### 2. Non-Functional Requirements & Budgets

- **NFR-<DOMAIN>-1 (Latency / Throughput)**: <Operation> MUST complete within `<budget>`.
- **NFR-<DOMAIN>-2 (Resource / Allocation)**: <Memory, I/O, or concurrency constraint>.

### 3. Assumptions & Edge Cases

- **ASM-<DOMAIN>-1**: <Domain-specific assumption>.
- **Edge Case 1**: <Empty input / boundary value / malformed payload> $\rightarrow$ <Deterministic behavior or error>.

---

### 4. Behavioral Scenarios (Given / When / Then)

#### Scenario 4.1: <Primary Happy Path>
- **Given** <initial state>
- **When** <action is triggered>
- **Then** <expected outcome matching REQ-*>

#### Scenario 4.2: <Boundary / Error Condition>
- **Given** <boundary or invalid state>
- **When** <action is triggered>
- **Then** <deterministic error or clamped state>

---

## Review Gate
- [ ] **Gate 2 Approval**: Specifications reviewed and approved by human before technical design.
