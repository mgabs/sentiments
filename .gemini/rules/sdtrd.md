# SDTRD (Spec-Driven, Testing-Railed Development) Rules

When working in this repository:

1. **Pre-Execution Invariant (Zero Direct Edits to main)**:
   - Before invoking `edit_file`, `write_file`, or running mutating scripts, ALWAYS inspect the current branch:
     ```bash
     rtk git branch --show-current
     ```
   - If the active branch is `main` or `master`, HALT immediately. Checkout a dedicated branch first.

2. **Pick a Lane First**:
   - **Feature Track**: `feat/<id>-<slug>` (Gates 0–6: research → proposal → spec → design → tasks → BDD/unit/mutation → sync/merge).
   - **Bugfix Track**: `fix/<slug>` (Gates B1–B3: bug-report → red regression test in worktree → minimal green fix → merge).
   - **Chore / Build Track**: `chore/<slug>` or `build/<slug>` (tooling, dependencies, Makefiles, dataset runners; zero behavior change; tests must pass).
   - **Docs Track**: `docs/<slug>` (spec guides, wiki, README changes).

3. **Strict Human Gates**:
   - Never proceed to code or the next gate without explicit human clearance ("k", "proceed", or approval).
   - Gate 2 specs must be a **full copy** (never delta).

4. **Testing Rails**:
   - Red before Green.
   - $\ge 95\%$ line and branch coverage; $100\%$ mutation kill score on core domain logic.
   - Never delete assertions, skip failing tests, or write vacuous tests.

5. **Commit & Merge Hygiene**:
   - Verify `rtk git status` and `rtk git diff` before committing.
   - Never bypass pre-commit hooks (`--no-verify` is forbidden).
   - Merge to `main` strictly via `--ff-only`.
