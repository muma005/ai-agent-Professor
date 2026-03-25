"""
Tests for API security and validation.

FLAW-7.3: API Key Security
FLAW-8.1: API Response Validation
"""
import pytest
import os
from tools.api_key_security import (
    mask_key,
    validate_api_keys,
    sanitize_for_logging,
    SecureKeyLoader,
    get_secure_loader,
    initialize_api_keys,
)
from tools.llm_client import APIResponseValidator


class TestAPIKeyMasking:
    """Test API key masking."""

    def test_mask_key_short(self):
        """Test masking short key."""
        masked = mask_key("abc")
        assert masked == "***"

    def test_mask_key_long(self):
        """Test masking long key."""
        key = "fw_abcdefghijklmnopqrstuvwxyz"
        masked = mask_key(key)
        
        assert masked.startswith("fw_abc")
        assert masked.endswith("***")
        assert key not in masked

    def test_mask_key_custom_visible(self):
        """Test masking with custom visible chars."""
        key = "abcdefghijklmnopqrstuvwxyz"
        masked = mask_key(key, visible_chars=4)
        
        assert len(masked.split("...")[0]) == 4


class TestAPIKeyValidation:
    """Test API key validation."""

    def test_validate_api_keys_structure(self):
        """Test validation result structure."""
        result = validate_api_keys()
        
        assert "valid" in result
        assert "missing_required" in result
        assert "missing_optional" in result
        assert "present" in result
        assert isinstance(result["missing_required"], list)
        assert isinstance(result["missing_optional"], list)
        assert isinstance(result["present"], list)


class TestSanitizeForLogging:
    """Test log sanitization."""

    def test_sanitize_removes_keys(self):
        """Test sanitization removes API keys."""
        text = "API key: fw_abcdefghijklmnopqrstuvwxyz123456"
        sanitized = sanitize_for_logging(text)
        
        assert "fw_abc" not in sanitized
        assert "KEY_MASKED" in sanitized or "***" in sanitized

    def test_sanitize_preserves_normal_text(self):
        """Test sanitization preserves normal text."""
        text = "This is a normal log message without keys"
        sanitized = sanitize_for_logging(text)
        
        assert sanitized == text


class TestSecureKeyLoader:
    """Test secure key loader."""

    def test_loader_creation(self):
        """Test loader can be created."""
        loader = SecureKeyLoader()
        
        assert loader is not None
        assert loader._loaded_keys == {}

    def test_loader_loads_key(self, monkeypatch):
        """Test loader loads key from env."""
        monkeypatch.setenv("TEST_API_KEY", "test_key_123456789")
        
        loader = SecureKeyLoader()
        key = loader.load_key("TEST_API_KEY", required=False)
        
        assert key == "test_key_123456789"
        assert "TEST_API_KEY" in loader._loaded_keys

    def test_loader_masks_key(self, monkeypatch):
        """Test loader masks key in summary."""
        monkeypatch.setenv("TEST_API_KEY", "test_key_abcdefghijklmnopqrstuvwxyz")
        
        loader = SecureKeyLoader()
        loader.load_key("TEST_API_KEY", required=False)
        
        summary = loader.get_loaded_keys_summary()
        
        assert "TEST_API_KEY" in summary
        assert "test_key" in summary["TEST_API_KEY"]
        assert "***" in summary["TEST_API_KEY"]

    def test_loader_missing_required_key(self):
        """Test loader raises on missing required key."""
        loader = SecureKeyLoader()
        
        with pytest.raises(ValueError, match="Missing required"):
            loader.load_key("NONEXISTENT_KEY_12345", required=True)

    def test_loader_accepts_missing_optional_key(self):
        """Test loader accepts missing optional key."""
        loader = SecureKeyLoader()
        
        key = loader.load_key("NONEXISTENT_KEY_12345", required=False)
        
        assert key is None


class TestAPIResponseValidator:
    """Test API response validation."""

    def test_validate_empty_response(self):
        """Test empty response detection."""
        result = APIResponseValidator.validate_response("", "deepseek")
        
        assert not result["valid"]
        assert "Empty response" in result["issues"]

    def test_validate_whitespace_response(self):
        """Test whitespace-only response."""
        result = APIResponseValidator.validate_response("   \n\t  ", "deepseek")
        
        assert not result["valid"]
        assert "Empty response" in result["issues"]

    def test_validate_valid_response(self):
        """Test valid response passes."""
        result = APIResponseValidator.validate_response(
            "This is a valid response with useful content.",
            "deepseek"
        )
        
        assert result["valid"] is True
        assert result["issues"] == []

    def test_detect_error_patterns(self):
        """Test error pattern detection."""
        result = APIResponseValidator.validate_response(
            "Error: API call failed with status 500",
            "deepseek"
        )
        
        assert "Contains error pattern: 'error:'" in result["warnings"]

    def test_detect_api_key_leakage(self):
        """Test API key leakage detection."""
        result = APIResponseValidator.validate_response(
            "My API key is sk-abcdefghijklmnopqrstuvwxyz1234567890",
            "deepseek"
        )
        
        assert not result["valid"]
        assert any("API key leakage" in issue for issue in result["issues"])

    def test_detect_long_response(self):
        """Test long response warning."""
        long_text = "a" * 150000  # 150K chars
        result = APIResponseValidator.validate_response(long_text, "deepseek")
        
        # Warning includes the length
        assert any("Unusually long response" in w for w in result["warnings"])

    def test_validation_result_structure(self):
        """Test validation result structure."""
        result = APIResponseValidator.validate_response("Test", "deepseek")
        
        assert "valid" in result
        assert "issues" in result
        assert "warnings" in result
        assert "model" in result
        assert "response_length" in result
        assert result["model"] == "deepseek"


class TestIntegration:
    """Integration tests."""

    def test_initialize_api_keys(self):
        """Test API key initialization."""
        result = initialize_api_keys()
        
        assert "valid" in result
        assert "present" in result
        assert isinstance(result["present"], list)

    def test_secure_loader_global_singleton(self):
        """Test global loader is singleton."""
        loader1 = get_secure_loader()
        loader2 = get_secure_loader()
        
        assert loader1 is loader2
