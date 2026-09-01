"""
Tests for EVORA Phase 3 identity and authority system.

These tests verify:
- AuthorityLevel enum and hierarchy
- Identity creation and serialization
- IdentityStore save/load/persistence
- Authorization checks (creator > admin > user > guest)
- Creator identity is protected (not via username comparison)
- Unauthorized administrative actions are rejected
- Restart persistence of identity configs
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from evora.identity import (
    AuthorityLevel,
    Identity,
    IdentityStore,
    IdentityService,
    AUTHORITY_RULES,
)


class TestAuthorityLevel:

    def test_hierarchy_ordering(self):
        """CREATOR is highest, GUEST is lowest."""
        hierarchy = AuthorityLevel.hierarchy()
        assert hierarchy[0] == AuthorityLevel.GUEST
        assert hierarchy[-1] == AuthorityLevel.CREATOR

    def test_guest_cannot_access_creator(self):
        assert not AuthorityLevel.GUEST.can_access(AuthorityLevel.CREATOR)

    def test_user_can_access_user(self):
        assert AuthorityLevel.USER.can_access(AuthorityLevel.USER)

    def test_admin_can_access_user(self):
        assert AuthorityLevel.ADMIN.can_access(AuthorityLevel.USER)

    def test_creator_can_access_all(self):
        for level in AuthorityLevel:
            assert AuthorityLevel.CREATOR.can_access(level)

    def test_guest_can_access_guest(self):
        assert AuthorityLevel.GUEST.can_access(AuthorityLevel.GUEST)

    def test_guest_cannot_access_admin(self):
        assert not AuthorityLevel.GUEST.can_access(AuthorityLevel.ADMIN)

    def test_authority_rules_exist(self):
        """Key actions should have authority rules defined."""
        assert "remember" in AUTHORITY_RULES
        assert "forget" in AUTHORITY_RULES
        assert "clear_project_memory" in AUTHORITY_RULES
        assert "set_creator" in AUTHORITY_RULES
        assert "modify_global_config" in AUTHORITY_RULES


class TestIdentity:

    def test_identity_creation(self):
        ident = Identity.create("Alice", AuthorityLevel.USER)
        assert ident.id is not None
        assert ident.name == "Alice"
        assert ident.authority == AuthorityLevel.USER
        assert ident.created_at is not None

    def test_creator_identity_creation(self):
        """Creator is designated by authority level, not username."""
        ident = Identity.create_creator("any_name")
        assert ident.authority == AuthorityLevel.CREATOR
        assert ident.is_creator is True

    def test_non_creator_is_not_creator(self):
        ident = Identity.create("Bob", AuthorityLevel.ADMIN)
        assert ident.is_creator is False

    def test_identity_to_dict(self):
        ident = Identity.create("Carol", AuthorityLevel.ADMIN)
        d = ident.to_dict()
        assert d["name"] == "Carol"
        assert d["authority"] == "admin"
        assert "id" in d

    def test_identity_from_dict(self):
        data = {
            "id": "test-id-123",
            "name": "Dave",
            "authority": "creator",
            "created_at": "2024-01-01T00:00:00",
        }
        ident = Identity.from_dict(data)
        assert ident.id == "test-id-123"
        assert ident.name == "Dave"
        assert ident.authority == AuthorityLevel.CREATOR


class TestIdentityStore:

    def test_save_and_load_identity(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create("TestUser", AuthorityLevel.USER)
        store.save_identity(ident)

        loaded = store.load_identity(ident.id)
        assert loaded is not None
        assert loaded.name == "TestUser"
        assert loaded.authority == AuthorityLevel.USER

    def test_load_nonexistent_identity(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        result = store.load_identity("nonexistent-id")
        assert result is None

    def test_list_identities(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        ident1 = Identity.create("User1", AuthorityLevel.USER)
        ident2 = Identity.create("User2", AuthorityLevel.ADMIN)
        store.save_identity(ident1)
        store.save_identity(ident2)

        identities = store.list_identities()
        assert len(identities) == 2

    def test_get_current_default_guest(self, tmp_path):
        """No identity configured → defaults to GUEST."""
        store = IdentityStore(str(tmp_path / "identity"))
        current = store.get_current()
        assert current.authority == AuthorityLevel.GUEST
        assert current.is_creator is False

    def test_set_current_and_get(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create("ActiveUser", AuthorityLevel.ADMIN)

        store.set_current(ident)
        current = store.get_current()
        assert current.id == ident.id
        assert current.name == "ActiveUser"

    def test_set_creator(self, tmp_path):
        """Creator identity is stored in a separate protected file."""
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create_creator("TheCreator")
        store.set_creator(ident)

        creator = store.get_creator()
        assert creator is not None
        assert creator.is_creator is True
        assert creator.name == "TheCreator"

    def test_get_creator_none_if_not_set(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        assert store.get_creator() is None

    def test_delete_identity(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create("ToDelete", AuthorityLevel.USER)
        store.save_identity(ident)

        assert store.delete_identity(ident.id) is True
        assert store.load_identity(ident.id) is None

    def test_delete_nonexistent_identity(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        assert store.delete_identity("does-not-exist") is False

    def test_identity_restart_persistence(self, tmp_path):
        """Identities must survive process restart (file persistence)."""
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create("PersistentUser", AuthorityLevel.ADMIN)
        store.save_identity(ident)
        store.set_current(ident)

        # Simulate restart: new store instance pointing to same dir
        store2 = IdentityStore(str(tmp_path / "identity"))
        loaded = store2.get_current()
        assert loaded is not None
        assert loaded.name == "PersistentUser"
        assert loaded.authority == AuthorityLevel.ADMIN

    def test_creator_restart_persistence(self, tmp_path):
        """Creator identity must survive restart."""
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create_creator("PersistentCreator")
        store.set_creator(ident)

        store2 = IdentityStore(str(tmp_path / "identity"))
        creator = store2.get_creator()
        assert creator is not None
        assert creator.is_creator is True

    def test_bootstrap_creator_first_time(self, tmp_path):
        """First-time creator bootstrap should work without authorization."""
        store = IdentityStore(str(tmp_path / "identity"))
        ident = store.bootstrap_creator("FirstCreator")
        assert ident.is_creator is True
        assert ident.name == "FirstCreator"

        # Creator is now set
        creator = store.get_creator()
        assert creator is not None
        assert creator.id == ident.id
        assert store.get_current().id == ident.id

    def test_bootstrap_creator_after_existing_fails(self, tmp_path):
        """Bootstrapping after creator exists should raise PermissionError."""
        store = IdentityStore(str(tmp_path / "identity"))
        store.bootstrap_creator("FirstCreator")

        with pytest.raises(PermissionError):
            store.bootstrap_creator("SecondCreator")

    def test_no_username_hardcoding(self, tmp_path):
        """Creator identity is NOT determined by username comparison."""
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("Tony", AuthorityLevel.GUEST))
        # Even though username is "Tony", authority is GUEST, not CREATOR
        current = store.get_current()
        assert current.is_creator is False
        assert current.authority == AuthorityLevel.GUEST


class TestIdentityService:

    def test_current_identity_returns_stored(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        ident = Identity.create("ServiceUser", AuthorityLevel.ADMIN)
        store.set_current(ident)

        service = IdentityService(store=store)
        current = service.current_identity()
        assert current.name == "ServiceUser"

    def test_check_authority_admin_action_for_user(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("RegularUser", AuthorityLevel.USER))

        service = IdentityService(store=store)
        assert not service.check_authority("clear_project_memory")

    def test_check_authority_admin_action_for_admin(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("AdminUser", AuthorityLevel.ADMIN))

        service = IdentityService(store=store)
        assert service.check_authority("clear_project_memory")

    def test_check_authority_creator_only_action_for_admin(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("AdminUser", AuthorityLevel.ADMIN))

        service = IdentityService(store=store)
        assert not service.check_authority("modify_global_config")

    def test_check_authority_creator_only_action_for_creator(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_creator(Identity.create_creator("TheCreator"))
        store.set_current(store.get_creator())

        service = IdentityService(store=store)
        assert service.check_authority("modify_global_config")

    def test_require_authority_raises_on_unauthorized(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("Guest", AuthorityLevel.GUEST))

        service = IdentityService(store=store)
        with pytest.raises(PermissionError):
            service.require_authority("clear_project_memory")

    def test_is_creator_true(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        creator = Identity.create_creator("Creator")
        store.set_current(creator)

        service = IdentityService(store=store)
        assert service.is_creator() is True

    def test_is_creator_false(self, tmp_path):
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("User", AuthorityLevel.USER))

        service = IdentityService(store=store)
        assert service.is_creator() is False

    def test_guest_cannot_clear_memory(self, tmp_path):
        """Guest should not be able to clear project memory."""
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create("Guest", AuthorityLevel.GUEST))

        service = IdentityService(store=store)
        assert not service.check_authority("clear_project_memory")
        with pytest.raises(PermissionError):
            service.require_authority("clear_project_memory")

    def test_creator_can_set_creator(self, tmp_path):
        """Only CREATOR authority can set creator identity."""
        store = IdentityStore(str(tmp_path / "identity"))
        store.set_current(Identity.create_creator("Creator"))

        service = IdentityService(store=store)
        assert service.check_authority("set_creator")
