# Research & Technical Tradeoffs: <Feature Title>

## 1. Intent & Ground Truth
<2–3 sentences on the core problem, verified constraints, and facts confirmed via codebase inspection or official documentation.>

### Verified Constraints & Prior Art
- **Existing System Constraints**: <What exists in the codebase today, APIs, data formats, invariants>
- **Ecosystem / Prior Art**: <How standard tools, libraries, or external systems solve this>

---

## 2. Options Explored

### Option A: <Short Name, e.g., Standard / Idiomatic Library Path>
- **Description**: <Brief summary of how this works>
- **Pros**:
  - <Pro 1>
  - <Pro 2>
- **Cons**:
  - <Con 1>
  - <Con 2>

### Option B: <Short Name, e.g., In-House / Zero-Dependency / Minimal Path>
- **Description**: <Brief summary of how this works>
- **Pros**:
  - <Pro 1>
  - <Pro 2>
- **Cons**:
  - <Con 1>
  - <Con 2>

### Option C: <Short Name, e.g., Alternative / Specialized Architecture> *(Optional)*
- **Description**: <Brief summary of how this works>
- **Pros / Cons**: <Brief summary>

---

## 3. Comparative Tradeoff Matrix

| Criteria | Option A (<Name>) | Option B (<Name>) | Option C (<Name>) |
| :--- | :--- | :--- | :--- |
| **Complexity** | Low / Med / High | Low / Med / High | Low / Med / High |
| **New Dependencies** | <e.g., none / 1 crate> | <e.g., none> | <e.g., external service> |
| **Performance & Latency** | <budget fit> | <budget fit> | <budget fit> |
| **Maintenance Burden** | Low / Med / High | Low / Med / High | Low / Med / High |
| **Blast Radius** | Localized / Wide | Localized / Wide | Localized / Wide |

---

## 4. Recommendation & Rationale
<1–2 sentences recommending the optimal option based on YAGNI, simplicity, and performance constraints.>

---

## 5. Breaking Changes & Deprecations
- **Affected Living Requirements**:
  - `REQ-<DOMAIN>-<ID>`: <Explain whether this requirement is modified, superseded, or deprecated by the proposed options>
  - `NFR-<DOMAIN>-<ID>`: <Any budget change or obsolete invariant>
- **Zero Breaking Changes**: *(Check if none)*
  - [ ] This capability introduces strictly additive behavior with zero modifications to existing living specs.

---

## Review Gate: Developer Decision
- [ ] **Decision**:
  - **Go**: Proceed with Option A | Option B | Option C $\rightarrow$ Open Gate 1 (Proposal)
  - **No-Go**: Reject all options $\rightarrow$ via a `chore/roadmap-<id>` branch on `main`: commit `research.md` to `specs/archive/<id>-<slug>/` and record `❌ Dropped` + rationale in `specs/ROADMAP.md`; then delete the feature branch.
- [ ] **Developer Sign-off**: Approved by human.
