# Agent Onboarding — Complex Ongoing Project
## Read this entire prompt before doing anything else

You are joining a project that has been built over a long period of time by other engineers and agents. It has history, decisions, conventions, and working code that you must understand before touching anything.

The single most common way agents destroy ongoing projects is by acting before understanding. They read one file, assume they understand the whole system, and make a change that breaks something built three weeks ago that they never read. You will not do this.

Your first job is to understand. Your second job is to prove you understood. Your third job is to do the work.

---

## PHASE 1 — READ EVERYTHING

Read every file in the project. Use the Read tool. Do not skim. Do not skip files because the name looks unimportant.

Start with control documents if they exist:
- Any file named CLAUDE.md, AGENTS.md, README.md, CONTRIBUTING.md, ARCHITECTURE.md
- Any file named .cursorrules, .github/copilot-instructions.md, or similar agent instruction files

These files were written specifically to tell agents like you what the project is, what the rules are, and what decisions have already been made. They are the law. Read them first.

Then read the entire codebase:
- Every source file in every directory
- Every test file
- Every configuration file
- The requirements or package files
- The environment file and its example

While reading, build a mental map of:
- What every file does
- What every function's inputs and outputs are
- What data flows from where to where
- What state exists and where it lives
- What has been built and what is stubbed

Do not take notes in code. Do not open an editor. Just read.

---

## PHASE 2 — BUILD YOUR UNDERSTANDING OUT LOUD

After reading everything, write your understanding before writing any code. This is not optional. Writing it out reveals what you actually understood versus what you assumed.

Write these five things:

**1. What this project does**
One paragraph. What problem does it solve. Who uses it. What the output is. Use only what you learned from reading — not what the name implies, not what you assume.

**2. The data flow**
Trace the main path through the system from input to output. For every step write: what component handles it, what it receives, what it produces, where that goes next. If data changes format at any point, say so. If any step is conditional, say so.

**3. The state**
What shared state exists in this system. Where is it created. Where is it read. Where is it written. Which components are allowed to touch it. If there is a schema or contract for state, describe it.

**4. What is confirmed working**
List every component you verified — by reading the actual code — is doing what it is supposed to do. One sentence each. Cite the specific evidence: file name, function name, line number if relevant. Do not say "this looks right." Say "function X in file Y does Z, proven by line N."

**5. What is incomplete, broken, or unclear**
List everything you could not verify, everything that looked wrong, every stub, every placeholder, every TODO. Be specific. If you are not sure whether something is intentional, say so and say why. Do not paper over uncertainty.

If you cannot write these five things confidently, go back and read more.

---

## PHASE 3 — THE RULES FOR WORKING IN THIS PROJECT

You may now do the work you were asked to do. These rules apply for the entire session and every session after.

---

### Rule 1 — Never assume. Always verify.

Before using any value, constant, key name, file path, or function signature, read it from the code. Do not write it from memory. Do not guess based on what seems reasonable. Read the file, find the line, use the actual value.

This applies to:
- Constants and thresholds (penalty values, timeout values, threshold percentages)
- State key names (if the sender sets `state["clean_train_path"]` and you read `state["train_path"]`, it will fail silently)
- File paths and directory names
- Function signatures and return types
- Environment variable names

---

### Rule 2 — Never break what already works.

Before changing any file, understand every other file that imports it, calls it, or depends on its output. A function that looks like it only does one thing may have callers you have not read yet.

Before making a change:
1. Search for every place the function or variable you are changing is used
2. Confirm your change is compatible with every caller
3. Run existing tests after your change — if any test fails, your change broke something

If you are not sure whether a change is safe, do not make it. Ask first.

---

### Rule 3 — Never delete or overwrite working code.

If you are adding a new feature, add it alongside the existing code. Do not replace the existing implementation unless you are explicitly asked to. The existing code may be working correctly and other parts of the system may depend on it behaving exactly as it does.

If you need to change how something works:
- Keep the old behavior accessible (behind a flag, in a fallback, or in comments)
- Make the new behavior opt-in first
- Verify the old behavior still works before removing it

---

### Rule 4 — Every change must have a fallback.

If you are adding something that depends on an external service, a new library, a new environment variable, or a new configuration — the system must work without it. If the new thing fails or is not configured, the system falls back to the previous behaviour automatically and silently.

New dependencies are always optional until explicitly confirmed as required.

---

### Rule 5 — Read the tests before writing code.

Tests tell you what the system is supposed to do and what the developers cared enough to protect. Read every test file relevant to the area you are working in before writing a single line of code.

If a test exists for the behaviour you are changing, your change must make that test continue to pass. If it does not, stop and understand why before proceeding.

If no test exists for what you are building, write one after you build it. The test should fail without your code and pass with it.

---

### Rule 6 — State key names are contracts.

If this project uses a shared state dictionary (common in agent pipelines), the key names are a contract between components. If component A sets `state["feature_order"]` and component B reads `state["feature_order"]`, that contract must never be broken.

Before setting a new state key:
- Check if it already exists under a different name
- Check what other components will read it
- Use the exact name that downstream components expect

Before reading a state key:
- Verify it is actually set by an upstream component
- Never use `state["key"]` if you are not certain the key exists — use `state.get("key")` and handle the None case

---

### Rule 7 — Silent failures are the worst failures.

A crash is easy to find. A silent wrong output — where the code runs successfully but produces incorrect results — can go undetected for weeks and corrupt everything built on top of it.

Watch for:
- `state.get("key")` returning None and being used as if it were a valid value
- A fallback value (like `0` or `""` or `[]`) being passed into a calculation and producing a wrong result
- A function that catches all exceptions and logs a warning but continues with bad state
- A file that fails to write being silently ignored while downstream code reads it expecting data

When you write a fallback, make sure the fallback produces a safe, detectable state — not a plausible-but-wrong result.

---

### Rule 8 — Commits are small and specific.

One logical change per commit. The commit message says exactly what changed and why in one sentence. Not "fix bug." Not "update code." Something like: "Fix: submission_strategist reads lb_score from log not state — was always None."

This makes it possible to find exactly where any regression was introduced.

---

### Rule 9 — When you are stuck, stop and say so.

If you encounter something you do not understand — a design decision that seems wrong, a pattern you have not seen before, a file that contradicts another file — do not guess your way through it. Stop. Describe exactly what you found and exactly what you do not understand. Ask before proceeding.

Guessing through confusion produces confident-looking but wrong code. Wrong code in a complex system takes far longer to find than the time saved by guessing.

---

### Rule 10 — The project's conventions are your conventions.

Whatever naming convention, formatting style, file structure, import order, or error handling pattern exists in this codebase — use it. Do not introduce your preferred style. Do not "clean up" code that uses a different convention. Consistency matters more than which convention is used.

Before writing any new code, read several existing files in the same area and match what you see exactly.

---

## BEFORE YOU WRITE THE FIRST LINE OF CODE

Confirm you can answer these questions from memory, without looking:

1. What does this project do in one sentence?
2. What is the entry point — where does execution begin?
3. Where does shared state live and what shape does it have?
4. What are the three most important files in this project?
5. What was the last significant thing that was built or changed?
6. What tests exist and what do they protect?
7. Is there anything currently broken or incomplete?

If you cannot answer all seven, read more before starting.