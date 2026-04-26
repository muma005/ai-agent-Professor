# V2 Refactoring Guardrails: Rules for Safe Evolution

This document serves as the absolute authority for all "Professor v2.0" refactoring efforts. These rules are designed to prevent the destructive failures, "Identity Crisis" type errors, and architectural blindness that occurred during the first refactoring attempt.

---

## 1. The Golden Rule of Refactoring
**Functional Parity is Non-Negotiable.** 
*   **NEVER** delete existing production logic (math, edge cases, specialized data handling) unless specifically instructed to replace it with a superior verified alternative.
*   **SURGICAL UPDATES ONLY:** Refactoring means changing the *structure* of the code (how state is accessed, how errors are caught), not deleting the *meat* of the work.
*   **3000 -> 164 is a failure, not a success.** If a file shrinks by more than 20% during a v2.0 refactor, you have likely deleted essential intelligence.

## 2. State Management Protocol (The "Identity Crisis" Shield)
*   **Object Preservation:** When updating state, you MUST preserve the `ProfessorState` object type.
*   **NO RECURSIVE DUMPING:** Never call `state.model_dump()` in a way that converts the entire state back to a dictionary (e.g., in retry loops or escalations). This lobotomizes the system.
*   **USE MODEL_COPY:** For Pydantic v2 updates, use `state.model_copy(update=updates)`. This preserves nested objects like `config` and `ProfessorConfig`.
*   **LEGACY COMPATIBILITY:** The `ProfessorState` must implement the Mapping protocol (`__getitem__`, `get`, `keys`) to ensure that downstream code using `state['key']` does not crash during the migration.
*   **FIELD OWNERSHIP:** Respect `_FIELD_OWNERS`. If an agent needs to write to a field it doesn't own, the architecture is wrong. Fix the flow, don't hack the validator.

## 3. Orchestration Integrity (The "Trunk" Rule)
*   **MAP THE TRUNK FIRST:** Before touching an agent (a "leaf"), you MUST read the orchestration files (`core/professor.py` and `agents/semantic_router.py`).
*   **STRICT NAMING:** Do not change Node IDs or internal variable names that are hard-coded in the graph. If the graph expects `ml_optimizer`, do not name your node `MLOptimizer`.
*   **VARIABLE STABILITY:** Do not rename keys (e.g., `target_col` to `target_column`) unless you are updating every single reference in the entire codebase simultaneously. "Split brain" state kills the pipeline.

## 4. Skills & Expertise Protection
*   **THE SKILLS FOLDER IS IMMUTABLE:** The `skills/` folder contains the agent's brain. Do not delete, move, or "slim down" these files. They define the LLM's identity and expert workflows.
*   **PROMPT OVER HARD-CODE:** Prioritize rich, expert prompt templates over hard-coded Python logic. The agent's power comes from its ability to act as a Kaggle Grandmaster, not a collection of `if/else` blocks.

## 5. Workspace Hygiene
*   **NO JUNK FILES:** Do not create planning files (`v2_0.md`), diagnostic scripts (`diag.py`), or redundant utilities in the root directory. Use the existing `docs/` and `tests/` folders.
*   **SOURCE OF TRUTH:** Verify if a utility (like JSON parsing or metric verification) already exists in `tools/` before creating a new one. Duplicate logic creates import loops and maintenance rot.
*   **WINDOWS COMPATIBILITY:** Use the provided tools (`grep_search`, `read_file`) instead of `run_shell_command` with Linux-specific binaries like `tail` or `grep` that may fail on the host environment.

## 6. Testing & Validation Workflow
*   **CONTRACTS FIRST:** Contract tests are the source of truth. They define the binary "Pass/Fail" state of the system. 
*   **NO PUSHING FAILURES:** Never push to origin if contract tests are failing. You are not a "build server"; verify locally first.
*   **SUBJECTIVITY IS SECONDARY:** Do not rely on "quality tests" to prove a refactor is working. Quality is subjective; Contracts are objective.

## 7. Anti-Feature Creep
*   **STRICT ADHERENCE:** Only build what is instructed. 
*   **UNASKED COMPLEXITY:** "Holographic Fast-Track," "State Size Budgeting," and "Provider Rotation" were uninstructed and harmful. If it's not in the plan, don't build it.

---

**Any agent entering this workspace MUST read and acknowledge these guardrails before making a single edit.**
