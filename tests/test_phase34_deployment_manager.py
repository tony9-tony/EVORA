"""
Phase 34 — Native Deployment Manager tests.

Verifies:
1. Deployment has correct structure
2. Release has correct structure
3. DeploymentStatus enum exists
4. ReleaseStatus enum exists
5. NativeDeploymentManager initializes
6. NativeDeploymentManager creates release
7. NativeDeploymentManager stages release
8. NativeDeploymentManager deploys release
9. NativeDeploymentManager rolls back deployment
10. NativeDeploymentManager gets deployment
11. NativeDeploymentManager gets release
12. NativeDeploymentManager returns metrics
13. No ModelManager dependency
14. No external dependencies
15. Works offline
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.brain.intelligence.deployment_manager import (
    Deployment,
    DeploymentStatus,
    NativeDeploymentManager,
    Release,
    ReleaseStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deployment_manager():
    return NativeDeploymentManager(logger=MagicMock())


@pytest.fixture
def manager_with_release():
    dm = NativeDeploymentManager(logger=MagicMock())
    release = dm.create_release("1.0.0", "Initial release", ["artifact1"])
    return dm, release


# ---------------------------------------------------------------------------
# TestDeployment
# ---------------------------------------------------------------------------

class TestDeployment:
    """Test Deployment."""

    def test_default_deployment(self):
        deployment = Deployment()
        assert deployment.deployment_id != ""
        assert deployment.status == DeploymentStatus.PENDING

    def test_deployment_to_dict(self):
        deployment = Deployment(status=DeploymentStatus.DEPLOYED, version="1.0.0")
        data = deployment.to_dict()
        assert data["status"] == "deployed"
        assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# TestRelease
# ---------------------------------------------------------------------------

class TestRelease:
    """Test Release."""

    def test_default_release(self):
        release = Release()
        assert release.release_id != ""
        assert release.status == ReleaseStatus.DRAFT

    def test_release_to_dict(self):
        release = Release(version="1.0.0", status=ReleaseStatus.STAGED)
        data = release.to_dict()
        assert data["version"] == "1.0.0"
        assert data["status"] == "staged"


# ---------------------------------------------------------------------------
# TestNativeDeploymentManager
# ---------------------------------------------------------------------------

class TestNativeDeploymentManager:
    """Test NativeDeploymentManager."""

    def test_deployment_manager_initializes(self, deployment_manager):
        assert deployment_manager is not None

    def test_create_release(self, deployment_manager):
        release = deployment_manager.create_release("1.0.0", "Initial release")
        assert release.release_id != ""
        assert release.version == "1.0.0"

    def test_create_release_with_artifacts(self, deployment_manager):
        release = deployment_manager.create_release("1.0.0", artifacts=["a1", "a2"])
        assert len(release.artifacts) == 2

    def test_stage_release(self, manager_with_release):
        dm, release = manager_with_release
        result = dm.stage_release(release.release_id)
        assert result is True
        assert release.status == ReleaseStatus.STAGED

    def test_stage_release_missing(self, deployment_manager):
        result = deployment_manager.stage_release("nonexistent")
        assert result is False

    def test_release(self, manager_with_release):
        dm, release = manager_with_release
        dm.stage_release(release.release_id)
        deployment = dm.release(release.release_id, environment="staging")
        assert deployment.status == DeploymentStatus.DEPLOYED
        assert deployment.environment == "staging"
        assert release.status == ReleaseStatus.RELEASED

    def test_release_missing(self, deployment_manager):
        deployment = deployment_manager.release("nonexistent")
        assert deployment.status == DeploymentStatus.FAILED

    def test_rollback(self, manager_with_release):
        dm, release = manager_with_release
        deployment = dm.release(release.release_id)
        result = dm.rollback(deployment.deployment_id)
        assert result is True
        assert deployment.status == DeploymentStatus.ROLLED_BACK

    def test_rollback_missing(self, deployment_manager):
        result = deployment_manager.rollback("nonexistent")
        assert result is False

    def test_get_deployment(self, manager_with_release):
        dm, release = manager_with_release
        deployment = dm.release(release.release_id)
        retrieved = dm.get_deployment(deployment.deployment_id)
        assert retrieved is not None
        assert retrieved.version == "1.0.0"

    def test_get_deployment_missing(self, deployment_manager):
        retrieved = deployment_manager.get_deployment("nonexistent")
        assert retrieved is None

    def test_get_release(self, manager_with_release):
        dm, release = manager_with_release
        retrieved = dm.get_release(release.release_id)
        assert retrieved is not None
        assert retrieved.version == "1.0.0"

    def test_get_release_missing(self, deployment_manager):
        retrieved = deployment_manager.get_release("nonexistent")
        assert retrieved is None

    def test_get_deployment_metrics(self, manager_with_release):
        dm, release = manager_with_release
        dm.release(release.release_id)
        metrics = dm.get_deployment_metrics()
        assert metrics["total_deployments"] == 1
        assert metrics["total_releases"] == 1


# ---------------------------------------------------------------------------
# TestSecurityBoundaries
# ---------------------------------------------------------------------------

class TestSecurityBoundaries:
    """Test Phase 34 security boundaries."""

    def test_no_model_manager_in_deployment(self):
        import evora.brain.intelligence.deployment_manager as dep_mod
        source = Path(dep_mod.__file__).read_text(encoding="utf-8")
        assert "from evora.model import ModelManager" not in source
        assert "from evora.model import" not in source

    def test_no_external_dependencies(self):
        import evora.brain.intelligence.deployment_manager as dep_mod
        source = Path(dep_mod.__file__).read_text(encoding="utf-8")
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
    """Test Phase 34 works offline."""

    def test_deployment_manager_works_offline(self, deployment_manager):
        release = deployment_manager.create_release("1.0.0")
        assert release is not None

    def test_metrics_offline(self, deployment_manager):
        deployment_manager.create_release("1.0.0")
        metrics = deployment_manager.get_deployment_metrics()
        assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# TestArchitectureReadiness
# ---------------------------------------------------------------------------

class TestArchitectureReadiness:
    """Test Phase 34 architecture readiness."""

    def test_native_deployment_manager_exists(self):
        from evora.brain.intelligence.deployment_manager import NativeDeploymentManager
        assert NativeDeploymentManager is not None

    def test_deployment_exists(self):
        from evora.brain.intelligence.deployment_manager import Deployment
        assert Deployment is not None

    def test_release_exists(self):
        from evora.brain.intelligence.deployment_manager import Release
        assert Release is not None

    def test_deployment_status_enum_exists(self):
        from evora.brain.intelligence.deployment_manager import DeploymentStatus
        assert DeploymentStatus.PENDING is not None
        assert DeploymentStatus.DEPLOYED is not None

    def test_release_status_enum_exists(self):
        from evora.brain.intelligence.deployment_manager import ReleaseStatus
        assert ReleaseStatus.DRAFT is not None
        assert ReleaseStatus.RELEASED is not None
