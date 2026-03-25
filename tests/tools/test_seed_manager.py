"""
Tests for seed management.

FLAW-10.1: Seed Management
"""
import pytest
import os
import random
from tools.seed_manager import (
    SeedManager,
    get_seed_manager,
    initialize_seeds,
    get_seed,
    get_seeds,
    reseed_all,
    DEFAULT_SEED,
)


class TestSeedManager:
    """Test SeedManager class."""

    def test_seed_manager_creation(self):
        """Test SeedManager can be created."""
        manager = SeedManager(seed=42)
        
        assert manager.base_seed == 42
        assert manager.get_seed() == 42

    def test_seed_manager_default_seed(self):
        """Test default seed is 42."""
        manager = SeedManager()
        
        assert manager.base_seed == DEFAULT_SEED
        assert DEFAULT_SEED == 42

    def test_seed_offset(self):
        """Test seed offset works correctly."""
        manager = SeedManager(seed=42)
        
        assert manager.get_seed(offset=0) == 42
        assert manager.get_seed(offset=1) == 43
        assert manager.get_seed(offset=10) == 52
        assert manager.get_seed(offset=-5) == 37

    def test_get_multiple_seeds(self):
        """Test generating multiple seeds."""
        manager = SeedManager(seed=100)
        
        seeds = manager.get_seeds(count=5)
        
        assert len(seeds) == 5
        assert seeds == [100, 101, 102, 103, 104]

    def test_get_seeds_with_offset(self):
        """Test generating seeds with start offset."""
        manager = SeedManager(seed=100)
        
        seeds = manager.get_seeds(count=3, start_offset=10)
        
        assert len(seeds) == 3
        assert seeds == [110, 111, 112]

    def test_reseed(self):
        """Test reseeding changes base seed."""
        manager = SeedManager(seed=42)
        
        assert manager.get_seed() == 42
        
        manager.reseed(999)
        
        assert manager.base_seed == 999
        assert manager.get_seed() == 999

    def test_to_dict(self):
        """Test conversion to dict."""
        manager = SeedManager(seed=42)
        
        result = manager.to_dict()
        
        assert result["base_seed"] == 42
        assert "initialized_at" in result
        assert result["python_seed"] == 42
        assert result["numpy_seed"] == 42
        assert "torch_seed" in result  # May be None if torch not installed

    def test_seed_manager_sets_numpy(self):
        """Test NumPy is seeded."""
        import numpy as np
        
        manager = SeedManager(seed=42)
        
        # Generate random numbers
        nums1 = np.random.rand(5)
        
        # Reseed and generate again
        manager.reseed(42)
        nums2 = np.random.rand(5)
        
        # Should be identical
        assert np.allclose(nums1, nums2)

    def test_seed_manager_sets_python_random(self):
        """Test Python random module is seeded."""
        manager = SeedManager(seed=123)
        
        # Generate random numbers
        nums1 = [random.random() for _ in range(5)]
        
        # Reseed and generate again
        manager.reseed(123)
        nums2 = [random.random() for _ in range(5)]
        
        # Should be identical
        assert nums1 == nums2


class TestGlobalSeedManager:
    """Test global seed manager functions."""

    def test_get_seed_manager_singleton(self):
        """Test get_seed_manager returns same instance."""
        # Clear global state
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        manager1 = get_seed_manager(seed=42)
        manager2 = get_seed_manager()
        
        assert manager1 is manager2
        assert manager1.base_seed == 42

    def test_initialize_seeds(self):
        """Test initialize_seeds creates new instance."""
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        manager = initialize_seeds(seed=999)
        
        assert manager.base_seed == 999
        assert seed_manager._seed_manager is manager

    def test_get_seed_convenience(self):
        """Test get_seed convenience function."""
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        initialize_seeds(seed=100)
        
        assert get_seed() == 100
        assert get_seed(offset=5) == 105

    def test_get_seeds_convenience(self):
        """Test get_seeds convenience function."""
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        initialize_seeds(seed=100)
        
        seeds = get_seeds(count=3)
        
        assert len(seeds) == 3
        assert seeds == [100, 101, 102]

    def test_reseed_all(self):
        """Test reseed_all convenience function."""
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        initialize_seeds(seed=42)
        
        assert get_seed() == 42
        
        reseed_all(777)
        
        assert get_seed() == 777


class TestReproducibility:
    """Test reproducibility guarantees."""

    def test_reproducible_numpy(self):
        """Test NumPy operations are reproducible."""
        import numpy as np
        
        # First run
        initialize_seeds(seed=42)
        arr1 = np.random.randn(10)
        result1 = np.mean(arr1)
        
        # Second run
        initialize_seeds(seed=42)
        arr2 = np.random.randn(10)
        result2 = np.mean(arr2)
        
        # Should be identical
        assert np.allclose(arr1, arr2)
        assert result1 == result2

    def test_reproducible_python_random(self):
        """Test Python random operations are reproducible."""
        # First run
        initialize_seeds(seed=42)
        nums1 = [random.randint(1, 100) for _ in range(10)]
        
        # Second run
        initialize_seeds(seed=42)
        nums2 = [random.randint(1, 100) for _ in range(10)]
        
        # Should be identical
        assert nums1 == nums2

    def test_different_seeds_produce_different_results(self):
        """Test different seeds produce different results."""
        import numpy as np
        
        initialize_seeds(seed=42)
        result1 = np.random.rand()
        
        initialize_seeds(seed=999)
        result2 = np.random.rand()
        
        # Should be different
        assert result1 != result2


class TestEnvironmentConfiguration:
    """Test environment variable configuration."""

    def test_default_seed_from_env(self, monkeypatch):
        """Test default seed can be set via env var."""
        from tools import seed_manager
        seed_manager._seed_manager = None
        
        monkeypatch.setenv("PROFESSOR_SEED", "12345")
        
        # Reload module to pick up new env var
        import importlib
        importlib.reload(seed_manager)
        
        assert seed_manager.DEFAULT_SEED == 12345
        
        manager = seed_manager.get_seed_manager()
        assert manager.base_seed == 12345

    def test_env_seed_default_is_42(self):
        """Test default seed is 42 when env var not set."""
        # Ensure env var is not set
        if "PROFESSOR_SEED" in os.environ:
            del os.environ["PROFESSOR_SEED"]
        
        from tools import seed_manager
        import importlib
        importlib.reload(seed_manager)
        
        assert seed_manager.DEFAULT_SEED == 42
