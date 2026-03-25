# core/graceful_degradation.py

"""
Graceful degradation system for Professor pipeline.

FLAW-2.5 FIX: Graceful Degradation
- Pipeline continues with fallbacks when components fail
- Automatic feature disabling based on failures
- Clear status reporting about degraded modes
- Recovery mechanisms when issues resolve
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class DegradationMode(Enum):
    """Pipeline degradation modes."""
    
    FULL = "full"              # All features enabled, 100% functionality
    REDUCED = "reduced"        # Some non-critical features disabled, 80%+ functionality
    MINIMAL = "minimal"        # Only critical features enabled, 50%+ functionality
    SAFE = "safe"              # Safe mode, manual intervention required


class FeaturePriority(Enum):
    """Feature priority levels."""
    
    CRITICAL = 1    # Pipeline cannot function without this (data_engineer, ml_optimizer, submit)
    HIGH = 2        # Important for quality (feature_factory, validation_architect)
    MEDIUM = 3      # Nice to have (eda_agent, competition_intel)
    LOW = 4         # Optional enhancements (external_data_scout)


@dataclass
class FeatureStatus:
    """Status of a single feature."""
    
    name: str
    priority: FeaturePriority
    enabled: bool = True
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    last_failure_reason: Optional[str] = None
    disabled_at: Optional[datetime] = None
    disabled_reason: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "name": self.name,
            "priority": self.priority.value,
            "enabled": self.enabled,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None,
            "last_failure_reason": self.last_failure_reason,
            "disabled_at": self.disabled_at.isoformat() if self.disabled_at else None,
            "disabled_reason": self.disabled_reason,
        }


@dataclass
class DegradationState:
    """Current state of degradation system."""
    
    mode: DegradationMode
    active_since: datetime
    disabled_features: Set[str] = field(default_factory=set)
    failure_history: List[dict] = field(default_factory=list)
    recovery_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "mode": self.mode.value,
            "active_since": self.active_since.isoformat(),
            "disabled_features": list(self.disabled_features),
            "failure_count": len(self.failure_history),
            "recovery_suggestions": self.recovery_suggestions,
        }


class GracefulDegradation:
    """
    Manages graceful degradation of Professor pipeline.
    
    Features:
    - Automatic mode switching based on failures
    - Feature-level enable/disable
    - Failure tracking and analysis
    - Recovery suggestions
    - Status reporting
    """
    
    # Configuration thresholds
    FAILURE_THRESHOLD_CRITICAL = 1    # Disable after 1 failure for critical features
    FAILURE_THRESHOLD_HIGH = 3         # Disable after 3 failures for high priority
    FAILURE_THRESHOLD_MEDIUM = 5       # Disable after 5 failures for medium priority
    FAILURE_THRESHOLD_LOW = 10         # Disable after 10 failures for low priority
    
    # Recovery settings
    AUTO_RECOVERY_ENABLED = True
    RECOVERY_COOLDOWN_MINUTES = 5
    
    def __init__(self):
        """Initialize graceful degradation system."""
        self.mode = DegradationMode.FULL
        self.active_since = datetime.now()
        self.features: Dict[str, FeatureStatus] = {}
        self.disabled_features: Set[str] = set()
        self.failure_history: List[dict] = []
        self.recovery_suggestions: List[str] = []
        self._feature_retry_times: Dict[str, datetime] = {}
        
        logger.info("[GracefulDegradation] Initialized in FULL mode")
    
    def register_feature(
        self,
        name: str,
        priority: FeaturePriority = FeaturePriority.MEDIUM,
    ) -> None:
        """
        Register a feature for degradation tracking.
        
        Args:
            name: Feature name (e.g., "eda_agent", "feature_factory")
            priority: Feature priority level
        """
        self.features[name] = FeatureStatus(
            name=name,
            priority=priority,
        )
        logger.debug(f"[GracefulDegradation] Registered feature: {name} (priority: {priority.value})")
    
    def record_failure(
        self,
        feature: str,
        error: str,
        context: Optional[dict] = None,
    ) -> bool:
        """
        Record a feature failure.
        
        Args:
            feature: Feature name
            error: Error message
            context: Optional context dict
        
        Returns:
            True if feature was disabled
        """
        now = datetime.now()
        
        # Get or create feature status
        if feature not in self.features:
            self.register_feature(feature, FeaturePriority.MEDIUM)
        
        status = self.features[feature]
        status.failure_count += 1
        status.last_failure = now
        status.last_failure_reason = error
        
        # Record in history
        self.failure_history.append({
            "timestamp": now.isoformat(),
            "feature": feature,
            "error": error,
            "context": context or {},
            "failure_count": status.failure_count,
        })
        
        # Check if feature should be disabled
        should_disable = self._should_disable_feature(status)
        
        if should_disable:
            self._disable_feature(feature, error)
            return True
        
        logger.warning(
            f"[GracefulDegradation] Feature '{feature}' failed "
            f"(count: {status.failure_count}, mode: {self.mode.value})"
        )
        
        return False
    
    def record_success(self, feature: str) -> None:
        """
        Record a feature success (can trigger recovery).
        
        Args:
            feature: Feature name
        """
        if feature not in self.features:
            return
        
        status = self.features[feature]
        
        # Reset failure count on success
        if status.failure_count > 0:
            logger.info(f"[GracefulDegradation] Feature '{feature}' succeeded, resetting failure count")
            status.failure_count = 0
            status.last_failure_reason = None
        
        # Attempt recovery if disabled
        if feature in self.disabled_features and self.AUTO_RECOVERY_ENABLED:
            self._attempt_recovery(feature)
    
    def is_feature_enabled(self, feature: str) -> bool:
        """
        Check if a feature is enabled.
        
        Args:
            feature: Feature name
        
        Returns:
            True if feature is enabled
        """
        if feature not in self.features:
            return True  # Unknown features are enabled by default
        
        return self.features[feature].enabled
    
    def get_disabled_features(self) -> List[str]:
        """Get list of disabled features."""
        return list(self.disabled_features)
    
    def get_mode(self) -> DegradationMode:
        """Get current degradation mode."""
        return self.mode
    
    def get_state(self) -> DegradationState:
        """Get current degradation state."""
        return DegradationState(
            mode=self.mode,
            active_since=self.active_since,
            disabled_features=self.disabled_features.copy(),
            failure_history=self.failure_history[-100:],  # Last 100 failures
            recovery_suggestions=self.recovery_suggestions.copy(),
        )
    
    def get_status_report(self) -> dict:
        """Get comprehensive status report."""
        enabled_count = sum(1 for f in self.features.values() if f.enabled)
        disabled_count = len(self.disabled_features)
        
        return {
            "mode": self.mode.value,
            "total_features": len(self.features),
            "enabled_features": enabled_count,
            "disabled_features": disabled_count,
            "functionality_percent": round(enabled_count / len(self.features) * 100, 1) if self.features else 100.0,
            "features": {name: status.to_dict() for name, status in self.features.items()},
            "recent_failures": self.failure_history[-10:],
            "recovery_suggestions": self.recovery_suggestions,
        }
    
    def _should_disable_feature(self, status: FeatureStatus) -> bool:
        """Check if feature should be disabled based on failures."""
        thresholds = {
            FeaturePriority.CRITICAL: self.FAILURE_THRESHOLD_CRITICAL,
            FeaturePriority.HIGH: self.FAILURE_THRESHOLD_HIGH,
            FeaturePriority.MEDIUM: self.FAILURE_THRESHOLD_MEDIUM,
            FeaturePriority.LOW: self.FAILURE_THRESHOLD_LOW,
        }
        
        threshold = thresholds.get(status.priority, self.FAILURE_THRESHOLD_MEDIUM)
        return status.failure_count >= threshold
    
    def _disable_feature(self, feature: str, reason: str) -> None:
        """
        Disable a feature.
        
        Args:
            feature: Feature name
            reason: Reason for disabling
        """
        if feature not in self.features:
            return
        
        status = self.features[feature]
        status.enabled = False
        status.disabled_at = datetime.now()
        status.disabled_reason = reason
        
        self.disabled_features.add(feature)
        
        logger.warning(
            f"[GracefulDegradation] DISABLED feature '{feature}' "
            f"(priority: {status.priority.value}, reason: {reason})"
        )
        
        # Update mode based on disabled features
        self._update_mode()
        
        # Add recovery suggestion
        self._generate_recovery_suggestion(feature, reason)
    
    def _enable_feature(self, feature: str) -> None:
        """
        Enable a previously disabled feature.
        
        Args:
            feature: Feature name
        """
        if feature not in self.features:
            return
        
        status = self.features[feature]
        status.enabled = True
        status.disabled_at = None
        status.disabled_reason = None
        status.failure_count = 0
        
        self.disabled_features.discard(feature)
        
        logger.info(f"[GracefulDegradation] ENABLED feature '{feature}'")
        
        # Update mode
        self._update_mode()
    
    def _update_mode(self) -> None:
        """Update degradation mode based on disabled features."""
        if not self.disabled_features:
            self.mode = DegradationMode.FULL
            return
        
        # Check for critical feature failures
        critical_features = {
            name for name, status in self.features.items()
            if status.priority == FeaturePriority.CRITICAL and not status.enabled
        }
        
        if critical_features:
            self.mode = DegradationMode.SAFE
            logger.error(
                f"[GracefulDegradation] SAFE MODE: Critical features disabled: {critical_features}"
            )
            return
        
        # Count disabled features by priority
        high_disabled = sum(
            1 for s in self.features.values()
            if s.priority == FeaturePriority.HIGH and not s.enabled
        )
        
        medium_disabled = sum(
            1 for s in self.features.values()
            if s.priority == FeaturePriority.MEDIUM and not s.enabled
        )
        
        # Determine mode
        if high_disabled >= 2 or len(self.disabled_features) >= 5:
            self.mode = DegradationMode.MINIMAL
        elif len(self.disabled_features) >= 2:
            self.mode = DegradationMode.REDUCED
        else:
            self.mode = DegradationMode.REDUCED
        
        logger.info(f"[GracefulDegradation] Mode updated to: {self.mode.value}")
    
    def _attempt_recovery(self, feature: str) -> None:
        """
        Attempt to recover a disabled feature.
        
        Args:
            feature: Feature name
        """
        if feature not in self.disabled_features:
            return
        
        # Check cooldown
        now = datetime.now()
        last_retry = self._feature_retry_times.get(feature)
        
        if last_retry:
            cooldown = (now - last_retry).total_seconds() / 60
            if cooldown < self.RECOVERY_COOLDOWN_MINUTES:
                logger.debug(
                    f"[GracefulDegradation] Recovery for '{feature}' on cooldown "
                    f"({cooldown:.1f} min < {self.RECOVERY_COOLDOWN_MINUTES} min)"
                )
                return
        
        # Attempt recovery
        self._feature_retry_times[feature] = now
        self._enable_feature(feature)
        
        logger.info(
            f"[GracefulDegradation] Attempting recovery for '{feature}' "
            f"(cooldown: {self.RECOVERY_COOLDOWN_MINUTES} min)"
        )
    
    def _generate_recovery_suggestion(self, feature: str, reason: str) -> None:
        """
        Generate recovery suggestion for a disabled feature.
        
        Args:
            feature: Feature name
            reason: Reason for disabling
        """
        suggestions = {
            "timeout": f"Increase timeout for {feature} or check for slow dependencies",
            "memory": f"Reduce memory usage in {feature} or increase PROFESSOR_MAX_MEMORY_GB",
            "api": f"Check API keys and rate limits for {feature}",
            "data": f"Verify data integrity for {feature} inputs",
            "default": f"Investigate {feature} failures and restart when resolved",
        }
        
        # Match suggestion based on error reason
        reason_lower = reason.lower()
        suggestion = suggestions.get("default")
        
        for keyword, sug in suggestions.items():
            if keyword in reason_lower:
                suggestion = sug
                break
        
        self.recovery_suggestions.append(f"[{feature}] {suggestion}")
        
        # Keep only last 10 suggestions
        self.recovery_suggestions = self.recovery_suggestions[-10:]


# Global degradation manager
_degradation_manager: Optional[GracefulDegradation] = None


def get_degradation_manager() -> GracefulDegradation:
    """Get or create global degradation manager."""
    global _degradation_manager
    
    if _degradation_manager is None:
        _degradation_manager = GracefulDegradation()
    
    return _degradation_manager


def degrade_feature(
    feature: str,
    error: str,
    priority: str = "medium",
    context: Optional[dict] = None,
) -> bool:
    """
    Convenience function to record feature failure.
    
    Args:
        feature: Feature name
        error: Error message
        priority: Feature priority (critical, high, medium, low)
        context: Optional context
    
    Returns:
        True if feature was disabled
    """
    manager = get_degradation_manager()
    
    # Register with priority if not already registered
    if feature not in manager.features:
        priority_map = {
            "critical": FeaturePriority.CRITICAL,
            "high": FeaturePriority.HIGH,
            "medium": FeaturePriority.MEDIUM,
            "low": FeaturePriority.LOW,
        }
        manager.register_feature(feature, priority_map.get(priority, FeaturePriority.MEDIUM))
    
    return manager.record_failure(feature, error, context)


def is_feature_enabled(feature: str) -> bool:
    """Check if a feature is enabled."""
    return get_degradation_manager().is_feature_enabled(feature)


def get_degradation_status() -> dict:
    """Get current degradation status."""
    return get_degradation_manager().get_status_report()
