# Technical Debt Tracker

**Project:** ai-agent-Professor
**Last Updated:** 2026-03-25
**Status:** 🟡 ACTIVE

---

## Overview

This document tracks technical debt, TODOs, and future improvements for the Professor project. Items are prioritized and tracked to prevent accumulation of unchecked debt.

---

## Debt Categories

| Category | Description | Priority |
|----------|-------------|----------|
| **P0** | Critical - Must fix before next release | 🔴 |
| **P1** | High - Should fix in next sprint | 🟠 |
| **P2** | Medium - Schedule for future sprint | 🟡 |
| **P3** | Low - Nice to have | 🟢 |

---

## Active Technical Debt

### P1: High Priority

#### DEBT-001: CV-Safe Target Encoding Integration
- **Created:** 2026-03-25
- **Component:** `agents/ml_optimizer.py`
- **Description:** Target encoding functions created but not integrated into CV loop
- **Impact:** Potential data leakage if not fixed
- **Fix:** Integrate `_apply_target_encoding_cv_safe()` into CV fold loop
- **Estimated Effort:** 2-3 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

#### DEBT-002: Pseudo-Label Agent Integration
- **Created:** 2026-03-25
- **Component:** `core/professor.py`
- **Description:** Pseudo-label agent not in pipeline graph
- **Impact:** Feature not utilized
- **Fix:** Add `pseudo_label_agent` node to LangGraph
- **Estimated Effort:** 1-2 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

---

### P2: Medium Priority

#### DEBT-003: LangFuse Integration
- **Created:** 2026-03-25
- **Component:** `core/professor.py`
- **Description:** LangFuse observability stubbed but not fully integrated
- **Impact:** Limited observability
- **Fix:** Complete LangFuse integration or remove stub
- **Estimated Effort:** 3-4 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

#### DEBT-004: Ensemble Architect Integration
- **Created:** 2026-03-25
- **Component:** `core/professor.py`
- **Description:** Ensemble architect not in pipeline graph
- **Impact:** Model blending not used
- **Fix:** Add `ensemble_architect` node to LangGraph
- **Estimated Effort:** 2-3 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

#### DEBT-005: Null Importance Cache
- **Created:** 2026-03-25
- **Component:** `tools/null_importance.py`
- **Description:** Null importance results not cached between runs
- **Impact:** Recomputation on every run
- **Fix:** Add disk caching for null importance results
- **Estimated Effort:** 2-3 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

---

### P3: Low Priority

#### DEBT-006: Round 2 LLM Prompt Improvement
- **Created:** 2026-03-25
- **Component:** `agents/feature_factory.py`
- **Description:** LLM generates invalid Python for feature expressions
- **Impact:** Round 2 features suppressed
- **Fix:** Improve prompt with valid Polars expression examples
- **Estimated Effort:** 1-2 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

#### DEBT-007: Test Coverage Gaps
- **Created:** 2026-03-25
- **Component:** Multiple
- **Description:** Some modules have <80% test coverage
- **Impact:** Potential undetected bugs
- **Fix:** Add tests for uncovered branches
- **Estimated Effort:** 4-6 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

#### DEBT-008: Documentation Gaps
- **Created:** 2026-03-25
- **Component:** Multiple
- **Description:** Some public functions lack docstrings
- **Impact:** Poor IDE support, unclear usage
- **Fix:** Add missing docstrings
- **Estimated Effort:** 3-4 hours
- **Assigned To:** Unassigned
- **Status:** ⏳ TODO

---

## Completed Debt

| ID | Description | Completed | Effort |
|----|-------------|-----------|--------|
| DEBT-009 | Data leakage fixes | 2026-03-25 | 8 hours |
| DEBT-010 | State validation | 2026-03-25 | 6 hours |
| DEBT-011 | Performance monitoring | 2026-03-25 | 5 hours |
| DEBT-012 | Memory profiling | 2026-03-25 | 4 hours |
| DEBT-013 | Seed management | 2026-03-25 | 3 hours |
| DEBT-014 | Reproducibility checks | 2026-03-25 | 4 hours |
| DEBT-015 | API security | 2026-03-25 | 4 hours |
| DEBT-016 | Model comparison framework | 2026-03-25 | 6 hours |
| DEBT-017 | Data quality tests | 2026-03-25 | 5 hours |
| DEBT-018 | Security tests | 2026-03-25 | 5 hours |

---

## Debt Metrics

### Current Status
- **Active Debt Items:** 8
- **P0 (Critical):** 0
- **P1 (High):** 2
- **P2 (Medium):** 3
- **P3 (Low):** 3

### Velocity
- **Debt Created (Last 30 Days):** 15 items
- **Debt Resolved (Last 30 Days):** 10 items
- **Net Change:** +5 items

### Effort Estimate
- **Total Estimated Effort:** 18-23 hours
- **Available Capacity (Sprint):** 40 hours
- **Debt as % of Capacity:** 45-58%

---

## Process

### Adding New Debt

1. Create new entry with unique ID (DEBT-XXX)
2. Categorize by priority (P0-P3)
3. Estimate effort
4. Assign to sprint or backlog

### Resolving Debt

1. Create branch: `fix/debt-XXX`
2. Implement fix
3. Add regression tests
4. Update this document
5. Merge via PR

### Review Cadence

- **Weekly:** Review new debt items
- **Sprint:** Resolve at least 2 debt items
- **Monthly:** Audit debt metrics

---

## Prevention

### Code Review Checklist
- [ ] No TODOs without corresponding debt entry
- [ ] No commented-out code
- [ ] Tests cover new functionality
- [ ] Documentation updated

### Definition of Done
- [ ] Code passes all tests
- [ ] Code passes linting
- [ ] Type hints added
- [ ] Docstrings complete
- [ ] No new P0/P1 debt created

---

## Related Documents

- [CODE_STYLE.md](CODE_STYLE.md) - Code style guide
- [PHASE_4_IMPLEMENTATION_PLAN.md](PHASE_4_IMPLEMENTATION_PLAN.md) - Phase 4 plan
- [BUG_TRACKER.md](BUG_TRACKER.md) - Bug tracker

---

**Maintained By:** Development Team
**Review Cadence:** Weekly
