# tools/seed_manager.py

"""
Centralized seed management for reproducible experiments.

FLAW-10.1 FIX: Seed Management
- Single source of truth for all random seeds
- Configurable via environment variables
- Automatic seeding of all random libraries
- Seed documentation in run metadata
"""

import os
import random
import logging
from typing import Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Default seed (configurable via env)
DEFAULT_SEED = int(os.environ.get("PROFESSOR_SEED", "42"))


class SeedManager:
    """
    Centralized seed manager for reproducible experiments.
    
    Ensures all random operations across the pipeline use consistent seeds.
    """
    
    def __init__(self, seed: int = DEFAULT_SEED):
        """
        Initialize seed manager with a base seed.
        
        Args:
            seed: Base seed for all random operations
        """
        self.base_seed = seed
        self.seed_offset = 0
        self._set_all_seeds(seed)
        logger.info(f"[SeedManager] Initialized with base seed: {seed}")
    
    def _set_all_seeds(self, seed: int) -> None:
        """
        Set seeds for all random libraries.
        
        Libraries seeded:
        - Python random module
        - NumPy
        - PyTorch (if available)
        - Environment PYTHONHASHSEED
        """
        # Python built-in random
        random.seed(seed)
        
        # NumPy
        try:
            import numpy as np
            np.random.seed(seed)
            logger.debug(f"[SeedManager] NumPy seeded with {seed}")
        except ImportError:
            pass
        
        # PyTorch
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            logger.debug(f"[SeedManager] PyTorch seeded with {seed}")
        except ImportError:
            pass
        
        # Environment hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
    
    def get_seed(self, offset: int = 0) -> int:
        """
        Get a seed value with optional offset.
        
        Useful for generating different seeds for different components
        while maintaining reproducibility.
        
        Args:
            offset: Offset from base seed (default: 0)
        
        Returns:
            Seed value (base_seed + offset)
        """
        return self.base_seed + offset
    
    def get_seeds(self, count: int, start_offset: int = 0) -> list[int]:
        """
        Get multiple seed values.
        
        Args:
            count: Number of seeds to generate
            start_offset: Starting offset
        
        Returns:
            List of seed values
        """
        seeds = [self.base_seed + start_offset + i for i in range(count)]
        logger.debug(f"[SeedManager] Generated {count} seeds: {seeds}")
        return seeds
    
    def reseed(self, seed: Optional[int] = None) -> None:
        """
        Reseed all libraries with a new base seed.
        
        Args:
            seed: New base seed (default: use original base_seed)
        """
        if seed is not None:
            self.base_seed = seed
        self._set_all_seeds(self.base_seed)
        logger.info(f"[SeedManager] Reseeded with base seed: {self.base_seed}")
    
    def to_dict(self) -> dict:
        """Convert to serializable dict for logging/state."""
        return {
            "base_seed": self.base_seed,
            "initialized_at": datetime.now(timezone.utc).isoformat(),
            "python_seed": self.base_seed,
            "numpy_seed": self.base_seed,
            "torch_seed": self.base_seed if self._torch_available() else None,
            "hash_seed": self.base_seed,
        }
    
    @staticmethod
    def _torch_available() -> bool:
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False


# Global seed manager instance (lazy initialization)
_seed_manager: Optional[SeedManager] = None


def get_seed_manager(seed: Optional[int] = None) -> SeedManager:
    """
    Get or create the global seed manager instance.
    
    Args:
        seed: Seed to use (only used on first call)
    
    Returns:
        Global SeedManager instance
    """
    global _seed_manager
    
    if _seed_manager is None:
        _seed_manager = SeedManager(seed if seed is not None else DEFAULT_SEED)
    elif seed is not None:
        # Allow reseeding if explicitly requested
        _seed_manager.reseed(seed)
    
    return _seed_manager


def initialize_seeds(seed: Optional[int] = None) -> SeedManager:
    """
    Initialize seeds at pipeline start.
    
    This should be called once at the very beginning of the pipeline
    to ensure all random operations are reproducible.
    
    Args:
        seed: Base seed (default: from PROFESSOR_SEED env var or 42)
    
    Returns:
        SeedManager instance
    """
    global _seed_manager
    
    # Clear any existing instance
    _seed_manager = None
    
    # Create new instance
    _seed_manager = get_seed_manager(seed)
    
    logger.info(
        f"[SeedManager] Seeds initialized -- base: {_seed_manager.base_seed}, "
        f"env: {os.environ.get('PROFESSOR_SEED', 'default=42')}"
    )
    
    return _seed_manager


def get_seed(offset: int = 0) -> int:
    """
    Convenience function to get a seed value.
    
    Args:
        offset: Offset from base seed
    
    Returns:
        Seed value
    """
    manager = get_seed_manager()
    return manager.get_seed(offset)


def get_seeds(count: int, start_offset: int = 0) -> list[int]:
    """
    Convenience function to get multiple seed values.
    
    Args:
        count: Number of seeds to generate
        start_offset: Starting offset
    
    Returns:
        List of seed values
    """
    manager = get_seed_manager()
    return manager.get_seeds(count, start_offset)


def reseed_all(seed: int) -> None:
    """
    Reseed all libraries with a new seed.
    
    Use with caution - this affects all subsequent random operations.
    
    Args:
        seed: New base seed
    """
    manager = get_seed_manager()
    manager.reseed(seed)
