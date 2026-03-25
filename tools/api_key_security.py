# tools/api_key_security.py

"""
API key security utilities.

FLAW-7.3 FIX: API Key Security
- Validates API keys are present
- Prevents key leakage in logs
- Masks keys in error messages
- Secure key loading from environment
"""

import os
import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

# API key patterns to detect and mask
KEY_PATTERNS = {
    "fireworks": r"fw[_-]?\w{20,}",
    "gemini": r"AIza\w{35}",
    "groq": r"gsk_\w{20,}",
    "openai": r"sk-[a-zA-Z0-9]{48}",
    "anthropic": r"sk-ant-[a-zA-Z0-9]{95}",
    "generic": r"(?i)(api[_-]?key|apikey|secret[_-]?key)\s*[=:]\s*['\"]?[\w-]{20,}",
}

# Keys to validate (from .env)
REQUIRED_KEYS = [
    "FIREWORKS_API_KEY",
    "FIREWORKS_GLM_API_KEY",
]

OPTIONAL_KEYS = [
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "LANGCHAIN_API_KEY",
    "KAGGLE_USERNAME",
    "KAGGLE_KEY",
]


def mask_key(key: str, visible_chars: int = 8) -> str:
    """
    Mask API key for safe logging.
    
    Args:
        key: API key to mask
        visible_chars: Number of characters to show at start
    
    Returns:
        Masked key (e.g., "fw_abc123...***")
    """
    if not key or len(key) <= visible_chars:
        return "***"
    
    return f"{key[:visible_chars]}...***"


def validate_api_keys() -> dict:
    """
    Validate that required API keys are present.
    
    Returns:
        Validation result dict
    """
    result = {
        "valid": True,
        "missing_required": [],
        "missing_optional": [],
        "present": [],
    }
    
    # Check required keys
    for key in REQUIRED_KEYS:
        if os.environ.get(key):
            result["present"].append(key)
        else:
            result["missing_required"].append(key)
            result["valid"] = False
    
    # Check optional keys
    for key in OPTIONAL_KEYS:
        if os.environ.get(key):
            result["present"].append(key)
        else:
            result["missing_optional"].append(key)
    
    return result


def sanitize_for_logging(text: str) -> str:
    """
    Remove or mask API keys from text for safe logging.
    
    Args:
        text: Text to sanitize
    
    Returns:
        Sanitized text with keys masked
    """
    sanitized = text
    
    # Mask known key patterns
    for pattern_name, pattern in KEY_PATTERNS.items():
        try:
            sanitized = re.sub(pattern, f"[{pattern_name}_KEY_MASKED]", sanitized)
        except re.error:
            logger.debug(f"[APIKeySecurity] Invalid regex pattern: {pattern_name}")
    
    # Mask environment variable values that look like keys
    for key_name in REQUIRED_KEYS + OPTIONAL_KEYS:
        key_value = os.environ.get(key_name)
        if key_value and len(key_value) > 8:
            masked = mask_key(key_value)
            sanitized = sanitized.replace(key_value, masked)
    
    return sanitized


class SecureKeyLoader:
    """Secure API key loader with validation."""
    
    def __init__(self):
        self._loaded_keys = {}
        self._validation_result = None
    
    def load_key(self, key_name: str, required: bool = True) -> Optional[str]:
        """
        Load API key from environment.
        
        Args:
            key_name: Environment variable name
            required: Whether key is required
        
        Returns:
            Key value or None
        """
        key = os.environ.get(key_name)
        
        if not key:
            if required:
                logger.error(f"[APIKeySecurity] Missing required key: {key_name}")
                raise ValueError(f"Missing required API key: {key_name}")
            else:
                logger.debug(f"[APIKeySecurity] Optional key not set: {key_name}")
                return None
        
        # Validate key format (basic check)
        if len(key) < 10:
            logger.error(f"[APIKeySecurity] Key too short: {key_name}")
            raise ValueError(f"Invalid API key format: {key_name}")
        
        # Store masked version for logging
        self._loaded_keys[key_name] = mask_key(key)
        
        return key
    
    def validate_all(self) -> dict:
        """
        Validate all loaded keys.
        
        Returns:
            Validation result
        """
        self._validation_result = validate_api_keys()
        
        if self._validation_result["valid"]:
            logger.info(
                f"[APIKeySecurity] Validation passed: "
                f"{len(self._validation_result['present'])} keys present"
            )
        else:
            logger.error(
                f"[APIKeySecurity] Validation failed: "
                f"Missing required keys: {self._validation_result['missing_required']}"
            )
        
        return self._validation_result
    
    def get_loaded_keys_summary(self) -> dict:
        """
        Get summary of loaded keys (masked).
        
        Returns:
            Dict of masked key names
        """
        return {
            key_name: masked
            for key_name, masked in self._loaded_keys.items()
        }


# Global secure loader
_secure_loader: Optional[SecureKeyLoader] = None


def get_secure_loader() -> SecureKeyLoader:
    """Get or create global secure key loader."""
    global _secure_loader
    
    if _secure_loader is None:
        _secure_loader = SecureKeyLoader()
    
    return _secure_loader


def initialize_api_keys() -> dict:
    """
    Initialize and validate API keys at startup.
    
    Returns:
        Validation result
    """
    loader = get_secure_loader()
    
    # Load required keys
    for key in REQUIRED_KEYS:
        try:
            loader.load_key(key, required=True)
        except ValueError:
            pass  # Will be caught in validation
    
    # Load optional keys
    for key in OPTIONAL_KEYS:
        try:
            loader.load_key(key, required=False)
        except ValueError:
            pass
    
    # Validate
    return loader.validate_all()


def log_key_summary() -> None:
    """Log summary of API keys (masked)."""
    loader = get_secure_loader()
    summary = loader.get_loaded_keys_summary()
    
    if summary:
        logger.info("[APIKeySecurity] Loaded keys:")
        for key_name, masked in summary.items():
            logger.info(f"  - {key_name}: {masked}")
    else:
        logger.warning("[APIKeySecurity] No API keys loaded")
