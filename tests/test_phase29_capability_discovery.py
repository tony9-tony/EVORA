"""
Phase 29 — Native Capability Discovery tests.

Verifies:
1. Capability has correct structure
2. DiscoveryResult has correct structure
3. NativeCapabilityDiscovery initializes
4. NativeCapabilityDiscovery discovers from registry
5. NativeCapabilityDiscovery discovers from runtime
6. NativeCapabilityDiscovery registers capability
7. NativeCapabilityDiscovery matches capabilities
8. NativeCapabilityDiscovery gets capability by ID
9. NativeCapabilityDiscovery gets all capabilities
10. NativeCapabilityDiscovery returns metrics
11. No ModelManager dependency
12. No external dependencies
13. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.capability_discovery import (
    Capability,
    DiscoveryResult,
    NativeCapabilityDiscovery,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def capability_discovery():
    return NativeCapabilityDiscovery(logger=MagicMock())


@pytest.fixture
def discovery_with_registry():
    registry = MagicMock()
    registry.list_capabilities.return_value = [
        {"name": "code_analysis", "description": "Analyze code", "confidence": 0.8},
        {"name": "test_generation", "description": "Generate tests", "confidence": 0.7},
    ]
    return NativeCapabilityDiscovery(capability_registry=registry, logger=MagicMock())


@pytest.fixture
def discovery_with_runtime():
    runtime = MagicMock()
    runtime.get_capabilities.return_value = [
        {"name": "reasoning", "description": "Reason about tasks", "confidence": 0.9},
    ]
    return NativeCapabilityDiscovery(intelligence_runtime=runtime, logger=MagicMock())


# ---------------------------------------------------------------------------
# TestCapability
# ---------------------------------------------------------------------------

class TestCapability:
    """Test Capability."""

    def test_default_capability(self):
        cap = Capability()
        assert cap.capability_id != ""
        assert cap.confidence == 0.5

    def test_capability_to_dict(self):
        cap = Capability(name="test", description="Test capability", confidence=0.8)
        data = cap.to_dict()
        assert data["name"] == "test"
        assert data["confidence"] == 0.8


# ---------------------------------------------------------------------------
# TestNativeCapabilityDiscovery
# ---------------------------------------------------------------------------

class TestNativeCapabilityDiscovery:
    """Test NativeCapabilityDiscovery."""

    def test_discovery_initializes(self, capability_discovery):
        assert capability_discovery is not None

    def test_discover_from_registry(self, discovery_with_registry):
        result = discovery_with_registry.discover_from_registry()
        assert isinstance(result, DiscoveryResult)
        assert result.success is True
        assert result.discovered_count > 0

    def test_discover_from_runtime(self, discovery_with_runtime):
        result = discovery_with_runtime.discover_from_runtime()
        assert isinstance(result, DiscoveryResult)
        assert result.success is True

    def test_discover_from_registry_no_registry(self, capability_discovery):
        result = capability_discovery.discover_from_registry()
        assert result.success is False
        assert len(result.errors) > 0

    def test_discover_from_runtime_no_runtime(self, capability_discovery):
        result = capability_discovery.discover_from_runtime()
        assert result.success is False

    def test_register_capability(self, capability_discovery):
        cap = Capability(name="new_capability", description="A new capability")
        result = capability_discovery.register_capability(cap)
        assert result is True
        retrieved = capability_discovery.get_capability(cap.capability_id)
        assert retrieved is not None

    def test_match_capability(self, discovery_with_registry):
        discovery_with_registry.discover_from_registry()
        matches = discovery_with_registry.match_capability("code")
        assert len(matches) > 0

    def test_match_capability_no_match(self, capability_discovery):
        matches = capability_discovery.match_capability("nonexistent")
        assert len(matches) == 0

    def test_get_capability(self, discovery_with_registry):
        discovery_with_registry.discover_from_registry()
        caps = discovery_with_registry.get_all_capabilities()
        if caps:
            retrieved = discovery_with_registry.get_capability(caps[0].capability_id)
            assert retrieved is not None

    def test_get_capability_missing(self, capability_discovery):
        retrieved = capability_discovery.get_capability("nonexistent")
        assert retrieved is None

    def test_get_all_capabilities(self, discovery_with_registry):
        discovery_with_registry.discover_from_registry()
        caps = discovery_with_registry.get_all_capabilities()
        assert len(caps) > 0

    def test_get_discovery_metrics(self, discovery_with_registry):
        discovery_with_registry.discover_from_registry()
        metrics = discovery_with_registry.get_discovery_metrics()
        assert "total_capabilities" in metrics
        assert metrics["total_capabilities"] > 0


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 29 security boundaries."""

    def test_no_model_manager_in_discovery(self):
        import evora.brain.intelligence.capability_discovery as disc_mod
        source = Path(disc_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.capability_discovery as disc_mod
        source = Path(disc_mod.__file__).read_text(encoding="utf-8")
        import_section = False
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                import_section = True
            elif import_section and stripped and not stripped.startswith("#"):
                break
            if import_section:
                for forbidden in ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx", "urllib", "socket"]:
                    assert forbidden not in stripped.lower(), f"Found forbidden dependency: {forbidden}"


# ---------------------------------------------------------------------------
# TestOfflineOperation
# ---------------------------------------------------------------------------

class TestOfflineOperation:
    """Test Phase 29 works offline."""

    def test_discovery_works_offline(self, capability_discovery):
        cap = Capability(name="offline_cap", description="Offline test")
        result = capability_discovery.register_capability(cap)
        assert result is True

    def test_match_offline(self, capability_discovery):
        cap = Capability(name="test_cap", description="Test")
        capability_discovery.register_capability(cap)
        matches = capability_discovery.match_capability("test")
        assert len(matches) > 0


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 29 architecture readiness."""

    def test_native_capability_discovery_exists(self):
        from evora.brain.intelligence.capability_discovery import NativeCapabilityDiscovery
        assert NativeCapabilityDiscovery is not None

    def test_capability_exists(self):
        from evora.brain.intelligence.capability_discovery import Capability
        assert Capability is not None

    def test_discovery_result_exists(self):
        from evora.brain.intelligence.capability_discovery import DiscoveryResult
        assert DiscoveryResult is not None

    def test_discovery_reuses_capability_registry(self, discovery_with_registry):
        assert discovery_with_registry.capability_registry is not None

    def test_discovery_reuses_intelligence_runtime(self, discovery_with_runtime):
        assert discovery_with_runtime.intelligence_runtime is not None
