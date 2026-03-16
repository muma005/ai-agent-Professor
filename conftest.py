import sys
import os

# Ensure the project root is on sys.path so that
# `import core`, `import agents`, `import tools`, etc. work in pytest.
sys.path.insert(0, os.path.dirname(__file__))

# Force fakeredis for ALL tests — prevents Windows fatal socket crash
# when real Redis is not running (code 0x80320012 from getaddrinfo).
import fakeredis
import memory.redis_state as _redis_state_module
_redis_state_module._redis_client = fakeredis.FakeRedis(decode_responses=True)

import pytest

@pytest.fixture(autouse=True)
def reset_graph_singleton():
    """Ensures each test gets a fresh graph if it modifies graph structure."""
    yield
    # Only clear if professor module was already imported (avoid expensive import on every test)
    prof = sys.modules.get("core.professor")
    if prof is not None:
        prof.get_graph_cache_clear()

