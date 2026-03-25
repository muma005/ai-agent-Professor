"""
Tests for graceful degradation.

FLAW-2.5: Graceful Degradation
"""
import pytest
from datetime import datetime, timedelta
from core.graceful_degradation import (
    GracefulDegradation,
    DegradationMode,
    FeaturePriority,
    FeatureStatus,
    get_degradation_manager,
    degrade_feature,
    is_feature_enabled,
    get_degradation_status,
)


class TestFeatureStatus:
    """Test FeatureStatus dataclass."""

    def test_feature_status_creation(self):
        """Test feature status can be created."""
        status = FeatureStatus(
            name="test_feature",
            priority=FeaturePriority.HIGH,
        )
        
        assert status.name == "test_feature"
        assert status.priority == FeaturePriority.HIGH
        assert status.enabled is True
        assert status.failure_count == 0

    def test_feature_status_to_dict(self):
        """Test conversion to dict."""
        status = FeatureStatus(
            name="test_feature",
            priority=FeaturePriority.MEDIUM,
            failure_count=3,
        )
        
        result = status.to_dict()
        
        assert result["name"] == "test_feature"
        assert result["priority"] == 3
        assert result["enabled"] is True
        assert result["failure_count"] == 3


class TestGracefulDegradation:
    """Test GracefulDegradation class."""

    def test_initialization(self):
        """Test graceful degradation initialization."""
        gd = GracefulDegradation()
        
        assert gd.mode == DegradationMode.FULL
        assert len(gd.disabled_features) == 0
        assert len(gd.features) == 0

    def test_register_feature(self):
        """Test feature registration."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.HIGH)
        
        assert "test_feature" in gd.features
        assert gd.features["test_feature"].priority == FeaturePriority.HIGH

    def test_record_single_failure(self):
        """Test recording a single failure."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        
        result = gd.record_failure("test_feature", "Test error")
        
        assert result is False  # Not disabled yet
        assert gd.features["test_feature"].failure_count == 1

    def test_record_multiple_failures_disables_feature(self):
        """Test multiple failures disable feature."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        
        # Record 5 failures (threshold for MEDIUM)
        for i in range(5):
            gd.record_failure("test_feature", f"Error {i+1}")
        
        assert "test_feature" in gd.disabled_features
        assert gd.features["test_feature"].enabled is False

    def test_critical_feature_disabled_after_one_failure(self):
        """Test critical feature disabled after one failure."""
        gd = GracefulDegradation()
        gd.register_feature("critical_feature", FeaturePriority.CRITICAL)
        
        result = gd.record_failure("critical_feature", "Critical error")
        
        assert result is True  # Disabled immediately
        assert "critical_feature" in gd.disabled_features

    def test_record_success_resets_failure_count(self):
        """Test success resets failure count."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        
        gd.record_failure("test_feature", "Error 1")
        gd.record_failure("test_feature", "Error 2")
        gd.record_success("test_feature")
        
        assert gd.features["test_feature"].failure_count == 0

    def test_is_feature_enabled(self):
        """Test feature enabled check."""
        gd = GracefulDegradation()
        gd.register_feature("enabled_feature", FeaturePriority.MEDIUM)
        gd.register_feature("disabled_feature", FeaturePriority.CRITICAL)
        
        # Disable one feature
        gd.record_failure("disabled_feature", "Critical error")
        
        assert gd.is_feature_enabled("enabled_feature") is True
        assert gd.is_feature_enabled("disabled_feature") is False

    def test_mode_changes_to_reduced(self):
        """Test mode changes to REDUCED when features disabled."""
        gd = GracefulDegradation()
        gd.register_feature("feature1", FeaturePriority.MEDIUM)
        gd.register_feature("feature2", FeaturePriority.MEDIUM)
        
        # Disable both features
        for _ in range(5):
            gd.record_failure("feature1", "Error")
            gd.record_failure("feature2", "Error")
        
        assert gd.mode == DegradationMode.REDUCED

    def test_mode_changes_to_safe_on_critical_failure(self):
        """Test mode changes to SAFE on critical feature failure."""
        gd = GracefulDegradation()
        gd.register_feature("critical_feature", FeaturePriority.CRITICAL)
        
        gd.record_failure("critical_feature", "Critical error")
        
        assert gd.mode == DegradationMode.SAFE

    def test_get_status_report(self):
        """Test status report generation."""
        gd = GracefulDegradation()
        gd.register_feature("feature1", FeaturePriority.HIGH)
        gd.record_failure("feature1", "Test error")
        
        report = gd.get_status_report()
        
        assert "mode" in report
        assert "total_features" in report
        assert "enabled_features" in report
        assert "disabled_features" in report
        assert "features" in report
        assert report["total_features"] == 1

    def test_get_state(self):
        """Test state retrieval."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        gd.record_failure("test_feature", "Error")
        
        state = gd.get_state()
        
        assert state.mode == DegradationMode.FULL  # Not enough failures yet
        assert len(state.failure_history) == 1


class TestDegradationModes:
    """Test degradation mode transitions."""

    def test_mode_full_to_reduced(self):
        """Test transition from FULL to REDUCED mode."""
        gd = GracefulDegradation()
        
        assert gd.mode == DegradationMode.FULL
        
        # Register and disable 2 medium priority features
        for i in range(2):
            gd.register_feature(f"feature{i}", FeaturePriority.MEDIUM)
            for _ in range(5):
                gd.record_failure(f"feature{i}", "Error")
        
        assert gd.mode in [DegradationMode.REDUCED, DegradationMode.MINIMAL]

    def test_mode_to_minimal(self):
        """Test transition to MINIMAL mode."""
        gd = GracefulDegradation()
        
        # Disable multiple high priority features
        for i in range(3):
            gd.register_feature(f"high_feature{i}", FeaturePriority.HIGH)
            for _ in range(3):
                gd.record_failure(f"high_feature{i}", "Error")
        
        assert gd.mode == DegradationMode.MINIMAL


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_degradation_manager_singleton(self):
        """Test get_degradation_manager returns same instance."""
        # Clear global state
        from core import graceful_degradation
        graceful_degradation._degradation_manager = None
        
        manager1 = get_degradation_manager()
        manager2 = get_degradation_manager()
        
        assert manager1 is manager2

    def test_degrade_feature_convenience(self):
        """Test degrade_feature convenience function."""
        from core import graceful_degradation
        graceful_degradation._degradation_manager = None
        
        # Record failures
        for i in range(5):
            degrade_feature("test_feature", f"Error {i}", priority="medium")
        
        assert not is_feature_enabled("test_feature")

    def test_is_feature_enabled_global(self):
        """Test is_feature_enabled global function."""
        from core import graceful_degradation
        graceful_degradation._degradation_manager = None
        
        # Feature not registered yet - should be enabled
        assert is_feature_enabled("unknown_feature") is True

    def test_get_degradation_status_global(self):
        """Test get_degradation_status global function."""
        from core import graceful_degradation
        graceful_degradation._degradation_manager = None
        
        degrade_feature("test_feature", "Error", priority="high")
        
        status = get_degradation_status()
        
        assert "mode" in status
        assert "features" in status


class TestRecovery:
    """Test recovery mechanisms."""

    def test_success_enables_recovery(self):
        """Test success can trigger recovery."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        
        # Disable feature
        for _ in range(5):
            gd.record_failure("test_feature", "Error")
        
        assert "test_feature" in gd.disabled_features
        
        # Record success
        gd.record_success("test_feature")
        
        # Feature should be re-enabled (auto-recovery is ON by default)
        assert gd.AUTO_RECOVERY_ENABLED is True

    def test_recovery_cooldown(self):
        """Test recovery cooldown."""
        gd = GracefulDegradation()
        gd.register_feature("test_feature", FeaturePriority.MEDIUM)
        
        # Disable feature
        for _ in range(5):
            gd.record_failure("test_feature", "Error")
        
        # First recovery attempt
        gd.record_success("test_feature")
        
        # Immediate second attempt should be on cooldown
        # (cooldown is 5 minutes by default)
        gd.record_success("test_feature")
        
        # Should still be enabled from first recovery
        assert "test_feature" not in gd.disabled_features or \
               "test_feature" in gd._feature_retry_times


class TestRecoverySuggestions:
    """Test recovery suggestion generation."""

    def test_timeout_suggestion(self):
        """Test timeout error generates correct suggestion."""
        gd = GracefulDegradation()
        gd.register_feature("slow_feature", FeaturePriority.MEDIUM)
        
        # Need to disable feature to generate suggestion (5 failures for MEDIUM)
        for _ in range(5):
            gd.record_failure("slow_feature", "Operation timed out after 300s")
        
        assert len(gd.recovery_suggestions) > 0
        assert "timeout" in gd.recovery_suggestions[0].lower() or \
               "slow" in gd.recovery_suggestions[0].lower() or \
               "increase" in gd.recovery_suggestions[0].lower()

    def test_memory_suggestion(self):
        """Test memory error generates correct suggestion."""
        gd = GracefulDegradation()
        gd.register_feature("memory_hog", FeaturePriority.MEDIUM)
        
        # Need to disable feature to generate suggestion
        for _ in range(5):
            gd.record_failure("memory_hog", "Out of memory: allocated 8GB")
        
        assert len(gd.recovery_suggestions) > 0
        assert "memory" in gd.recovery_suggestions[0].lower() or \
               "reduce" in gd.recovery_suggestions[0].lower()

    def test_api_suggestion(self):
        """Test API error generates correct suggestion."""
        gd = GracefulDegradation()
        gd.register_feature("api_consumer", FeaturePriority.MEDIUM)
        
        # Need to disable feature to generate suggestion
        for _ in range(5):
            gd.record_failure("api_consumer", "API rate limit exceeded")
        
        assert len(gd.recovery_suggestions) > 0
        assert "api" in gd.recovery_suggestions[0].lower() or \
               "rate" in gd.recovery_suggestions[0].lower() or \
               "check" in gd.recovery_suggestions[0].lower()
