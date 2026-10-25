# Spec-Driven, Testing-Railed Development (SDTRD) Workflow

## 1. Branching & Worktree Strategy

### Rule 1: Always Use a New Feature Branch
- **NEVER** work directly on `main` / `master`.
- Every `specs/` feature proposal MUST originate on a dedicated feature branch:
  ```bash
  rtk git checkout -b feat/<feature-id>-<feature-name>
  ```
- All proposal, specs, design, tasks, and vertical-slice work commits to this feature branch.

### Rule 2: Multi-Agent Parallel Streams via Git Worktrees
- When tasks/slices have **no mutual dependencies** (orthogonal modules, distinct capabilities, or decoupled tests):
  1. **Pre-Requisite Contract Lock**: Any shared domain models, interfaces, or contracts (e.g. `core/models.py`) MUST be implemented, tested, and committed to the parent feature branch *before* spawning parallel worktrees.
  2. **Worktree Hygiene**: Ensure `.worktrees/` is present in root `.gitignore`. Check for and clean any stale `.git/index.lock` prior to worktree operations:
     ```bash
     rtk rm -f .git/index.lock
     ```
  3. **Spawn Isolated Worktree**:
     ```bash
     rtk git worktree add .worktrees/<feature-id>-<stream-name> -b feat/<feature-id>/<stream-name>
     ```
  4. **Environment & Path Resolution**:
     - Tests and modules MUST resolve repository and dataset roots dynamically (e.g. iterating `Path(__file__).parents` until root marker like `.git` or `pyproject.toml` is found).
     - Reuse parent virtual environment (`.venv`) inside the worktree without reinstallation.
  5. **Concurrently Dispatched Subagents**:
     - Dispatch a dedicated subagent (`invoke_subagent`) per stream, passing the worktree directory.
     - Each agent executes its vertical slice in isolation:
       - BDD Scenarios (Gate 5.x)
       - Unit Tests (Red)
       - Minimal Implementation (Green)
       - Coverage ($\ge 95\%$ line & branch)
       - Targeted Slice Mutation Testing ($100\%$ kill score on slice module)
  6. **Rebase & Merge Sequence**:
     - The first completed stream merges into the feature branch via fast-forward:
       ```bash
       rtk git checkout feat/<feature-id>-<feature-name>
       rtk git merge --ff-only feat/<feature-id>/<stream-name>
       rtk git worktree remove .worktrees/<feature-id>-<stream-name>
       rtk git branch -d feat/<feature-id>/<stream-name>
       ```
     - Subsequent streams MUST rebase onto the updated feature branch before merging:
       ```bash
       rtk git -C .worktrees/<feature-id>-<stream-2> rebase feat/<feature-id>-<feature-name>
       rtk git checkout feat/<feature-id>-<feature-name>
       rtk git merge --ff-only feat/<feature-id>/<stream-2>
       rtk git worktree remove .worktrees/<feature-id>-<stream-2>
       rtk git branch -d feat/<feature-id>/<stream-2>
       ```

### Rule 3: Fast-Track Bugfix Protocol (Zero-Contamination Bug Isolation)
- **Scope Distinction**:
  - **In-Flight Slice Defect**: A defect discovered inside the active slice being developed is resolved directly within that slice's branch/worktree via an immediate red-first regression test. No extra triage gates needed.
  - **Cross-Cutting / Main / Regression Defect**: A defect affecting already-merged modules or discovered during dogfooding MUST follow the full Fast-Track Protocol below.
- **HARD PROHIBITION**: NEVER modify production code on `main` or within a dirty feature working directory without an isolated worktree and red test.
- **Protocol Steps**:
  1. **Log Defect**: File a bug report in `specs/bug-reports/<date>-<slug>.md` detailing root cause, reproduction steps, and expected behavior.
  2. **Spawn Worktree**: Create an isolated bugfix worktree branching off target base (e.g., `main`):
     ```bash
     rtk git worktree add .worktrees/fix-<bug-slug> -b fix/<bug-slug> main
     ```
  3. **Mandatory Red-First Regression Test**:
     - Write a targeted test replicating the defect *before touching any production code*.
     - Run the test in the worktree: MUST fail (`Red`).
  4. **Minimal Fix (Green)**:
     - Apply the minimal surgical fix in the worktree to turn the test passing (`Green`).
     - Run the full test suite to guarantee zero regressions.
  5. **Merge & Prune**:
     - Fast-forward merge the fix into `main`:
       ```bash
       rtk git checkout main
       rtk git merge --ff-only fix/<bug-slug>
       rtk git worktree remove .worktrees/fix-<bug-slug>
       rtk git branch -d fix/<bug-slug>
       ```
  6. **Rebase Active Feature Branches**:
     - Rebase active feature branches and worktrees onto updated `main`.

---

## 2. Feature Lifecycle & Human Gates

```mermaid
flowchart TD
    B0[Checkout feat/branch] --> G1[Gate 1: Proposal Review]
    G1 --> G2[Gate 2: Specs Review]
    G2 --> G3[Gate 3: Architecture & Design Review]
    G3 --> G4[Gate 4: Tasks Breakdown Review]
    G4 --> DAG{Analyze Task DAG}

    DAG -- Independent Streams --> S_A[Agent Stream A: Worktree A]
    DAG -- Independent Streams --> S_B[Agent Stream B: Worktree B]
    DAG -- Sequential / Dependent --> S_Seq[Sequential Slice Execution]

    subgraph StreamA_Loop [Stream A: Vertical Slice]
        A1[BDD Scenarios] --> GA[Gate 5.A Review]
        GA --> A2[Unit Tests Red]
        A2 --> A3[Implement Green]
        A3 --> A4[Coverage >= 95%]
        A4 --> A5[Mutation 100% Kill]
    end

    subgraph StreamB_Loop [Stream B: Vertical Slice]
        B1[BDD Scenarios] --> GB[Gate 5.B Review]
        GB --> B2[Unit Tests Red]
        B2 --> B3[Implement Green]
        B3 --> B4[Coverage >= 95%]
        B4 --> B5[Mutation 100% Kill]
    end

    S_A --> StreamA_Loop
    S_B --> StreamB_Loop
    S_Seq --> StreamA_Loop

    StreamA_Loop --> MA[Merge Worktree A --ff-only]
    StreamB_Loop --> MB[Rebase & Merge Worktree B]

    MA & MB --> G6[Gate 6: Spec Sync & Archive]
```

### Gate 1: Proposal Review
- **Artifact**: `specs/features/<feature>/proposal.md`
- **Mandatory Sections**: Intent, Problem Statement, Scope (In-Scope / Out-of-Scope YAGNI), Success Metrics, Assumptions (`ASM-GEN-*`).
- **Gate**: Human approval required before drafting specs.

### Gate 2: Specifications Review
- **Artifacts**: `specs/features/<feature>/specs/<capability>/spec.md`
- **Mandatory Sections**: Functional Requirements (`REQ-*`), Non-Functional Requirements & Performance Budgets (`NFR-*`), Assumptions (`ASM-*`), Edge Cases & Error Conditions, Behavioral Scenarios (`Given / When / Then`).
- **Traceability**: Every requirement MUST be uniquely keyed (`REQ-<DOMAIN>-<ID>`, `NFR-<DOMAIN>-<ID>`).
- **Gate**: Human approval required before technical design.

### Gate 3: Architecture & Design Review
- **Artifact**: `specs/features/<feature>/design.md`
- **Mandatory Sections**:
  - Architectural Overview & Invariants
  - Domain Models, State & Interface Contracts
  - Architecture Diagrams & Data Flow
  - Boundary Rules (Pure Core vs I/O Shell)
  - Concurrency, Latency & Resource Budgets (matching `NFR-*` specs)
  - Testing Strategy
- **Domain-Specific Baselines**:
  - **UI / Interactive Systems**: Define render frame deadlines (e.g., $< 16\text{ms}$), input latency budgets, and background I/O offloading rules.
  - **ML / Data / Heuristics**: Mandate a deterministic Tier 0 Dumb Baseline before complex models or external APIs.
- **Gate**: Human approval required before task breakdown.

### Gate 4: Tasks Breakdown Review
- **Artifact**: `specs/features/<feature>/tasks.md`
- **Mandatory Sections**:
  - **Dependency DAG**: Explicit mapping of slice prerequisites.
  - **Multi-Agent Streams**: Orthogonal slices partitioned into streams (`Stream A`, `Stream B`, etc.) with designated Git worktrees (`.worktrees/<feature>-<stream>`).
  - **Order of Execution**: Test-first vertical slice steps per stream (BDD $\rightarrow$ Gate 5.x $\rightarrow$ Unit Red $\rightarrow$ Code Green $\rightarrow$ Coverage $\rightarrow$ Mutation).
  - **Integration & Rebase Plan**: Merge sequence and synchronization checkpoints across streams.
- **Gate**: Human approval required before spawning worktrees or writing test code.

---

## 3. Bug Lifecycle & Fast-Track Gates

```mermaid
flowchart TD
    BR[Bug Encountered / Dogfood Failure] --> GB1[Gate B1: Bug Report & Triage]
    GB1 --> WT[Spawn .worktrees/fix-slug]
    WT --> T_Red[Write Regression Test: Red]
    T_Red --> GB2[Gate B2: Reproduction Confirmed]
    GB2 --> C_Green[Minimal Fix: Green]
    C_Green --> V_Suite[Full Regression Suite Passes]
    V_Suite --> GB3[Gate B3: Integration Sign-off]
    GB3 --> Merge[Merge --ff-only & Prune Worktree]
    Merge --> Rebase[Rebase Active Feature Streams]
```

### Gate B1: Bug Report & Triage
- **Artifact**: `specs/bug-reports/<date>-<slug>.md`
- **Mandatory Sections**: Status & Metadata, Summary, Steps to Reproduce, Expected vs Actual Result, Root Cause Analysis, Remediation Plan.
- **Gate**: Human approval required before code execution.

### Gate B2: Reproduction Confirmed (Red-First)
- Spawn isolated worktree off target branch (`rtk git worktree add .worktrees/fix-<slug> -b fix/<slug> main`).
- Write targeted regression test replicating the defect *before touching any production code*.
- Test MUST fail (`Red`) on unmodified code.
- **Gate**: Human verifies the failing test reproduces the real defect.

### Gate B3: Integration Sign-off
- Apply minimal surgical fix (`Green`) and run full regression suite.
- Human inspects final diff and test output.
- Fast-forward merge into `main`, prune worktree, mark bug report `Resolved` with commit hash, and rebase active feature branches.

---

## 4. Apply Phase: Vertical-Slice Test-Driven Implementation

Execute each vertical slice through three strict sub-steps:

### Step 5.x-A: BDD Behavior Scenarios (100% Behavior Coverage)
- Write Gherkin feature scenarios (`tests/bdd/features/*.feature` or stack equivalent) covering all `REQ-*`, `NFR-*`, and boundary conditions.
- **Gate 5.x Presentation**: Present scenarios to human in ultra-compact bullet format (scenario title, key assertions, boundary checks). Wait for explicit confirmation before writing tests.

### Step 5.x-B: Unit Tests & Red-Green-Refactor ($\ge 95\%$ Coverage)
- **Red**: Write failing unit tests first. Verify test failure for the expected reason.
- **Green**: Implement minimal production code to make tests pass (strictly YAGNI).
- **Refactor**: Clean up duplication and structure under green test protection.
- **Coverage**: Enforce line and branch coverage $\ge 95\%$ on the target module.
- **Performance Verification**: Verify any `NFR-*` latency or allocation budgets with deterministic benchmarks/assertions.

### Step 5.x-C: Mutation Testing (100% Kill Score)
- **Scoped Run**: Scope mutation runner strictly to the active slice module (e.g., `mutmut run --paths-to-mutate src/<slice>/`) to avoid re-mutating completed slices. Full codebase mutation run is reserved for Gate 6.
- **Surviving Mutant Protocol**: Every surviving mutant is a blocking defect. Resolve by:
  1. Adding a targeted test assertion that kills the mutant, OR
  2. Deleting redundant/unreachable code (Ponytail / YAGNI rule).
- **Target**: $100\%$ mutation kill rate on slice domain and core logic.

---

## 5. Coding Agents & Review Harness Integration

- **Anchor Directives**: Honor line-anchored feedback directives (`#ref file:line <directive>`) emitted by code review harnesses (such as `lgtm`). Acknowledge directive, resolve at exact target line, verify test pass, and commit.
- **Review Inbox**: Check `.lgtm/reviews/` and inline user directives on every turn; resolve review items before advancing gates.
- **Test Integrity (Anti-Fraud)**:
  - NEVER delete assertions or weaken tests to force a green suite.
  - NEVER write vacuous assertions or tautological mocks.
  - NEVER skip/ignore failing tests without explicit human sign-off.
- **Architecture Boundaries & Hygiene**:
  - Keep domain core pure (zero direct I/O, network, or framework coupling).
  - Restrict file modifications strictly to the active vertical slice (minimal blast radius).
  - Leave zero scratch files, debug dumps, or `.env` artifacts in the working tree.

---

## 6. Gate 6: Living Spec Sync, Archive & Merge

1. Run the full test suite and verify $100\%$ pass rate + $\ge 95\%$ line and branch coverage.
2. Run the full mutation suite across the entire project and verify $100\%$ kill score on core logic.
3. Ensure all parallel stream worktrees are merged (`--ff-only`) into the feature branch and pruned.
4. Sync feature delta specs (`specs/features/<feature>/specs/`) into the canonical living specifications directory (`specs/living/`).
5. Check off all items in `specs/features/<feature>/tasks.md` and record Gate 6 completion.
6. Archive feature proposal and tasks to `specs/archive/<feature>/`.
7. Fast-forward merge feature branch into `main`:
   ```bash
   rtk git checkout main
   rtk git merge --ff-only feat/<feature-id>-<feature-name>
   ```
