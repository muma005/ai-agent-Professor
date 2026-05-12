# Quick tests (run on every commit)
test-quick:
	pytest tests/contracts/ tests/unit/ -v --tb=short -q

# Full tests (run before merge)
test-full:
	pytest tests/contracts/ tests/unit/ tests/fault_injection/ tests/performance/ -v --tb=short

# Smoke test (run after build)
test-smoke:
	pytest tests/smoke/ -v --timeout=300

# Regression tests (run weekly or before release)
test-regression:
	PROFESSOR_REGRESSION_TESTS=1 pytest tests/regression/ -v --timeout=1800

# All tests
test-all:
	PROFESSOR_REGRESSION_TESTS=1 pytest tests/ -v --timeout=1800

# Contract tests only (fastest — run during development)
test-contracts:
	pytest tests/contracts/ -v -q
