"""
Phase 10 — CapabilityRegistry tests.

Verifies:
* capability registration
* retrieval
* duplicate handling
* unknown capability
* capability type
* native confidence
* requires_model
* fallback metadata
* approval metadata
* deterministic listing
* bounded results
* malformed input
* offline operation
* zero ModelManager dependency
* no network dependency
* web capability metadata exists without implementing network access
"""

import inspect

import pytest

from evora.brain.intelligence import (
    CapabilityRegistry,
    CapabilityType,
    IntelligenceCapability,
)


@pytest.fixture
def empty_registry():
    return CapabilityRegistry()


@pytest.fixture
def seeded_registry():
    registry = CapabilityRegistry()
    registry.seed_default_capabilities()
    return registry


@pytest.fixture
def logger_mock():
    return pytest.MagicMock()


class TestCapabilityType:
    """Test CapabilityType enum."""

    def test_capability_type_values(self):
        assert CapabilityType.NATIVE.value == "native"
        assert CapabilityType.LOCAL_MODEL.value == "local_model"
        assert CapabilityType.EXTERNAL_MODEL.value == "external"
        assert CapabilityType.UNAVAILABLE.value == "unavailable"

    def test_capability_type_from_string(self):
        assert CapabilityType("native") == CapabilityType.NATIVE
        assert CapabilityType("local_model") == CapabilityType.LOCAL_MODEL
        assert CapabilityType("external") == CapabilityType.EXTERNAL_MODEL
        assert CapabilityType("unavailable") == CapabilityType.UNAVAILABLE


class TestIntelligenceCapability:
    """Test IntelligenceCapability dataclass."""

    def test_capability_creation_valid(self):
        cap = IntelligenceCapability(
            name="test capability",
            description="A test capability",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.8,
            requires_model=False,
            fallback_available=True,
            requires_approval=False,
        )
        assert cap.name == "test capability"
        assert cap.capability_type == CapabilityType.NATIVE
        assert cap.native_confidence == 0.8
        assert cap.requires_model is False
        assert cap.fallback_available is True
        assert cap.requires_approval is False

    def test_capability_empty_name_rejected(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="",
                description="test",
                capability_type=CapabilityType.NATIVE,
            )

    def test_capability_whitespace_name_rejected(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="   ",
                description="test",
                capability_type=CapabilityType.NATIVE,
            )

    def test_capability_empty_description_rejected(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="test",
                description="",
                capability_type=CapabilityType.NATIVE,
            )

    def test_capability_invalid_type_rejected(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="test",
                description="test",
                capability_type="invalid",
            )

    def test_capability_confidence_clamped(self):
        cap = IntelligenceCapability(
            name="test",
            description="test",
            capability_type=CapabilityType.NATIVE,
            native_confidence=1.5,
        )
        assert cap.native_confidence == 1.0

        cap2 = IntelligenceCapability(
            name="test2",
            description="test",
            capability_type=CapabilityType.NATIVE,
            native_confidence=-0.5,
        )
        assert cap2.native_confidence == 0.0

    def test_capability_requires_model_must_be_bool(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="test",
                description="test",
                capability_type=CapabilityType.NATIVE,
                requires_model="yes",
            )

    def test_capability_fallback_available_must_be_bool(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="test",
                description="test",
                capability_type=CapabilityType.NATIVE,
                fallback_available="yes",
            )

    def test_capability_requires_approval_must_be_bool(self):
        with pytest.raises(ValueError):
            IntelligenceCapability(
                name="test",
                description="test",
                capability_type=CapabilityType.NATIVE,
                requires_approval="yes",
            )

    def test_capability_serialization_roundtrip(self):
        cap = IntelligenceCapability(
            name="test capability",
            description="A test capability",
            capability_type=CapabilityType.LOCAL_MODEL,
            native_confidence=0.6,
            requires_model=True,
            fallback_available=False,
            requires_approval=True,
            metadata={"key": "value"},
        )
        data = cap.to_dict()
        restored = IntelligenceCapability.from_dict(data)
        assert restored.name == "test capability"
        assert restored.capability_type == CapabilityType.LOCAL_MODEL
        assert restored.native_confidence == 0.6
        assert restored.requires_model is True
        assert restored.fallback_available is False
        assert restored.requires_approval is True
        assert restored.metadata == {"key": "value"}

    def test_capability_from_dict_with_string_type(self):
        data = {
            "name": "test",
            "description": "test",
            "capability_type": "native",
            "native_confidence": 0.5,
        }
        cap = IntelligenceCapability.from_dict(data)
        assert cap.capability_type == CapabilityType.NATIVE


class TestCapabilityRegistryBasics:
    """Test basic CapabilityRegistry operations."""

    def test_register_capability(self, empty_registry):
        cap = IntelligenceCapability(
            name="simple reasoning",
            description="Reason using native capabilities",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
        )
        empty_registry.register(cap)
        assert empty_registry.get("simple reasoning") is not None
        assert empty_registry.get("simple reasoning").native_confidence == 0.7

    def test_register_non_capability_raises(self, empty_registry):
        with pytest.raises(TypeError):
            empty_registry.register("not a capability")

    def test_get_missing_capability(self, empty_registry):
        assert empty_registry.get("nonexistent") is None

    def test_can_handle_native(self, empty_registry):
        cap = IntelligenceCapability(
            name="knowledge retrieval",
            description="Retrieve knowledge",
            capability_type=CapabilityType.NATIVE,
        )
        empty_registry.register(cap)
        assert empty_registry.can_handle("knowledge retrieval") is True

    def test_can_handle_unavailable(self, empty_registry):
        cap = IntelligenceCapability(
            name="teleportation",
            description="Teleport objects",
            capability_type=CapabilityType.UNAVAILABLE,
        )
        empty_registry.register(cap)
        assert empty_registry.can_handle("teleportation") is False

    def test_can_handle_missing(self, empty_registry):
        assert empty_registry.can_handle("nonexistent") is False

    def test_requires_model(self, empty_registry):
        native_cap = IntelligenceCapability(
            name="simple reasoning",
            description="Reason natively",
            capability_type=CapabilityType.NATIVE,
        )
        external_cap = IntelligenceCapability(
            name="complex reasoning",
            description="Reason with model",
            capability_type=CapabilityType.EXTERNAL_MODEL,
            requires_model=True,
        )
        empty_registry.register(native_cap)
        empty_registry.register(external_cap)
        assert empty_registry.requires_model("simple reasoning") is False
        assert empty_registry.requires_model("complex reasoning") is True
        assert empty_registry.requires_model("nonexistent") is False


class TestDuplicateHandling:
    """Test duplicate capability handling."""

    def test_duplicate_name_keeps_higher_confidence(self, empty_registry):
        cap1 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v1",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.5,
        )
        cap2 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v2",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.8,
        )
        empty_registry.register(cap1)
        empty_registry.register(cap2)
        assert len(empty_registry._capabilities) == 1
        assert empty_registry.get("reasoning").native_confidence == 0.8
        assert empty_registry.get("reasoning").description == "Reasoning v2"

    def test_duplicate_name_lower_confidence_ignored(self, empty_registry):
        cap1 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v1",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.9,
        )
        cap2 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v2",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.5,
        )
        empty_registry.register(cap1)
        empty_registry.register(cap2)
        assert len(empty_registry._capabilities) == 1
        assert empty_registry.get("reasoning").native_confidence == 0.9
        assert empty_registry.get("reasoning").description == "Reasoning v1"

    def test_duplicate_name_same_confidence_keeps_first(self, empty_registry):
        cap1 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v1",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
        )
        cap2 = IntelligenceCapability(
            name="reasoning",
            description="Reasoning v2",
            capability_type=CapabilityType.NATIVE,
            native_confidence=0.7,
        )
        empty_registry.register(cap1)
        empty_registry.register(cap2)
        assert len(empty_registry._capabilities) == 1
        assert empty_registry.get("reasoning").description == "Reasoning v1"


class TestCapabilityListing:
    """Test capability listing and filtering."""

    def test_list_capabilities_all(self, seeded_registry):
        caps = seeded_registry.list_capabilities()
        assert len(caps) > 0
        assert caps == sorted(caps, key=lambda c: c.native_confidence, reverse=True)

    def test_list_capabilities_by_type(self, seeded_registry):
        native_caps = seeded_registry.list_capabilities(capability_type=CapabilityType.NATIVE)
        assert all(c.capability_type == CapabilityType.NATIVE for c in native_caps)

        external_caps = seeded_registry.list_capabilities(capability_type=CapabilityType.EXTERNAL_MODEL)
        assert all(c.capability_type == CapabilityType.EXTERNAL_MODEL for c in external_caps)

        unavailable_caps = seeded_registry.list_capabilities(capability_type=CapabilityType.UNAVAILABLE)
        assert all(c.capability_type == CapabilityType.UNAVAILABLE for c in unavailable_caps)

    def test_list_capabilities_bounded(self, seeded_registry):
        caps = seeded_registry.list_capabilities(limit=3)
        assert len(caps) == 3

    def test_get_native_capabilities(self, seeded_registry):
        native = seeded_registry.get_native_capabilities()
        assert len(native) > 0
        assert all(c.capability_type == CapabilityType.NATIVE for c in native)
        assert all(c.native_confidence > 0.0 for c in native)

    def test_get_model_enhanced_capabilities(self, seeded_registry):
        enhanced = seeded_registry.get_model_enhanced_capabilities()
        assert len(enhanced) > 0
        assert all(
            c.capability_type in (CapabilityType.LOCAL_MODEL, CapabilityType.EXTERNAL_MODEL)
            for c in enhanced
        )

    def test_get_unavailable_capabilities(self, seeded_registry):
        unavailable = seeded_registry.get_unavailable_capabilities()
        assert len(unavailable) > 0
        assert all(c.capability_type == CapabilityType.UNAVAILABLE for c in unavailable)


class TestSeededCapabilities:
    """Test seeded default capabilities."""

    def test_seeded_native_capabilities(self, seeded_registry):
        native_names = [c.name for c in seeded_registry.get_native_capabilities()]
        assert "simple reasoning" in native_names
        assert "known-pattern planning" in native_names
        assert "knowledge retrieval" in native_names
        assert "tool suggestion" in native_names
        assert "known-fact inference" in native_names

    def test_seeded_external_capabilities(self, seeded_registry):
        external_names = [c.name for c in seeded_registry.get_model_enhanced_capabilities()]
        assert "complex reasoning" in external_names
        assert "novel planning" in external_names
        assert "advanced code generation" in external_names

    def test_seeded_unavailable_capabilities(self, seeded_registry):
        unavailable_names = [c.name for c in seeded_registry.get_unavailable_capabilities()]
        assert "autonomous self-modification" in unavailable_names
        assert "unrestricted computer control" in unavailable_names

    def test_seeded_web_capabilities_exist(self, seeded_registry):
        web_search = seeded_registry.get("web search")
        assert web_search is not None
        assert web_search.capability_type == CapabilityType.LOCAL_MODEL
        assert web_search.requires_approval is True

        web_browse = seeded_registry.get("web browse")
        assert web_browse is not None
        assert web_browse.capability_type == CapabilityType.LOCAL_MODEL
        assert web_browse.requires_approval is True

        web_research = seeded_registry.get("web research")
        assert web_research is not None
        assert web_research.capability_type == CapabilityType.LOCAL_MODEL
        assert web_research.requires_approval is True


class TestSummary:
    """Test registry summary."""

    def test_empty_summary(self, empty_registry):
        summary = empty_registry.summary()
        assert summary["total_capabilities"] == 0
        assert summary["by_type"] == {}

    def test_seeded_summary(self, seeded_registry):
        summary = seeded_registry.summary()
        assert summary["total_capabilities"] > 0
        assert summary["native_count"] > 0
        assert summary["model_enhanced_count"] > 0
        assert summary["unavailable_count"] > 0
        assert "native" in summary["by_type"]
        assert "external" in summary["by_type"]
        assert "unavailable" in summary["by_type"]


class TestMalformedInput:
    """Test safe handling of malformed input."""

    def test_register_non_capability_raises(self, empty_registry):
        with pytest.raises(TypeError):
            empty_registry.register("not a capability")

    def test_register_capability_with_empty_name_raises(self, empty_registry):
        with pytest.raises(ValueError):
            empty_registry.register(
                IntelligenceCapability(
                    name="",
                    description="test",
                    capability_type=CapabilityType.NATIVE,
                )
            )

    def test_register_capability_with_whitespace_name_raises(self, empty_registry):
        with pytest.raises(ValueError):
            empty_registry.register(
                IntelligenceCapability(
                    name="   ",
                    description="test",
                    capability_type=CapabilityType.NATIVE,
                )
            )

    def test_register_capability_with_empty_description_raises(self, empty_registry):
        with pytest.raises(ValueError):
            empty_registry.register(
                IntelligenceCapability(
                    name="test",
                    description="",
                    capability_type=CapabilityType.NATIVE,
                )
            )

    def test_register_capability_with_invalid_type_raises(self, empty_registry):
        with pytest.raises(ValueError):
            empty_registry.register(
                IntelligenceCapability(
                    name="test",
                    description="test",
                    capability_type="invalid",
                )
            )

    def test_register_capability_with_invalid_confidence_clamps(self, empty_registry):
        cap = IntelligenceCapability(
            name="test_high",
            description="test",
            capability_type=CapabilityType.NATIVE,
            native_confidence=1.5,
        )
        empty_registry.register(cap)
        assert empty_registry.get("test_high").native_confidence == 1.0

        cap2 = IntelligenceCapability(
            name="test_low",
            description="test",
            capability_type=CapabilityType.NATIVE,
            native_confidence=-0.5,
        )
        empty_registry.register(cap2)
        assert empty_registry.get("test_low").native_confidence == 0.0

    def test_get_missing_returns_none(self, empty_registry):
        assert empty_registry.get("nonexistent") is None

    def test_can_handle_missing_returns_false(self, empty_registry):
        assert empty_registry.can_handle("nonexistent") is False

    def test_requires_model_missing_returns_false(self, empty_registry):
        assert empty_registry.requires_model("nonexistent") is False


class TestBoundedResults:
    """Test bounded capability listing."""

    def test_max_capabilities_bounded(self):
        registry = CapabilityRegistry()
        registry._max_capabilities = 5
        for i in range(10):
            registry.register(
                IntelligenceCapability(
                    name=f"capability_{i}",
                    description=f"Capability {i}",
                    capability_type=CapabilityType.NATIVE,
                    native_confidence=float(i) / 10.0,
                )
            )
        assert len(registry._capabilities) <= 5

    def test_list_capabilities_bounded(self, seeded_registry):
        caps = seeded_registry.list_capabilities(limit=2)
        assert len(caps) == 2


class TestOfflineOperation:
    """Test offline operation."""

    def test_no_model_manager_dependency(self, empty_registry):
        """CapabilityRegistry has no ModelManager dependency."""
        assert not hasattr(empty_registry, "model_manager")
        assert not hasattr(empty_registry, "_model_manager")

    def test_no_network_dependency(self):
        """CapabilityRegistry does not use network libraries."""
        import evora.brain.intelligence.capabilities as cap_mod
        source = cap_mod.__file__
        with open(source, "r") as f:
            lines = f.readlines()
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            code_lines.append(line)
        code = "".join(code_lines)
        forbidden = ["requests", "aiohttp", "httpx", "urllib"]
        for term in forbidden:
            assert term not in code.lower(), f"CapabilityRegistry must not use {term}"

    def test_no_authority_grant(self, empty_registry):
        """CapabilityRegistry does not grant authority."""
        cap = IntelligenceCapability(
            name="test",
            description="test",
            capability_type=CapabilityType.NATIVE,
            requires_approval=True,
        )
        empty_registry.register(cap)
        retrieved = empty_registry.get("test")
        assert retrieved.requires_approval is True
        assert not hasattr(retrieved, "granted")
        assert not hasattr(retrieved, "authorized")


class TestWebCapabilities:
    """Test web capability metadata without implementing network access."""

    def test_web_capabilities_metadata_only(self, seeded_registry):
        """Web capabilities exist as metadata only, not implemented."""
        web_search = seeded_registry.get("web search")
        assert web_search is not None
        assert web_search.capability_type == CapabilityType.LOCAL_MODEL
        assert web_search.requires_approval is True
        assert web_search.requires_model is False
        assert web_search.native_confidence == 0.0

    def test_web_capabilities_no_network_implementation(self):
        """Web capabilities in capabilities.py must not implement network access."""
        import evora.brain.intelligence.capabilities as cap_mod
        source = cap_mod.__file__
        with open(source, "r") as f:
            lines = f.readlines()
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            code_lines.append(line)
        code = "".join(code_lines)
        forbidden = ["requests", "aiohttp", "httpx", "urllib"]
        for term in forbidden:
            assert term not in code.lower(), f"Web capabilities must not use {term}"

    def test_web_capabilities_require_approval(self, seeded_registry):
        """Web capabilities must require approval."""
        for name in ["web search", "web browse", "web research"]:
            cap = seeded_registry.get(name)
            assert cap is not None, f"{name} capability should be seeded"
            assert cap.requires_approval is True, f"{name} should require approval"


class TestSecurityBoundaries:
    """Test that CapabilityRegistry respects security boundaries."""

    def test_registry_does_not_call_model_manager(self):
        """CapabilityRegistry source must not reference ModelManager."""
        import evora.brain.intelligence.capabilities as cap_mod
        source = cap_mod.__file__
        with open(source, "r") as f:
            lines = f.readlines()
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            code_lines.append(line)
        code = "".join(code_lines)
        assert "ModelManager" not in code
        assert "model_manager" not in code

    def test_registry_does_not_call_external_apis(self):
        """CapabilityRegistry source must not call external APIs."""
        import evora.brain.intelligence.capabilities as cap_mod
        source = cap_mod.__file__
        with open(source, "r") as f:
            lines = f.readlines()
        code_lines = []
        in_docstring = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = not in_docstring
                continue
            if in_docstring:
                continue
            code_lines.append(line)
        code = "".join(code_lines)
        forbidden = ["openai", "anthropic", "ollama", "requests", "aiohttp", "httpx"]
        for term in forbidden:
            assert term not in code.lower()

    def test_requires_approval_is_metadata_only(self, seeded_registry):
        """requires_approval is metadata, not enforcement."""
        cap = seeded_registry.get("autonomous self-modification")
        assert cap is not None
        assert cap.requires_approval is True
        assert cap.capability_type == CapabilityType.UNAVAILABLE
        assert cap.native_confidence == 0.0
