"""
Identity and authority system for EVORA Phase 3.

Provides a protected creator identity that is NOT derived from conversational
text or username comparison. Identity and authority are stored in a separate
configuration file, designed to be replaceable with proper authentication
later.

Authority levels:
    CREATOR  — Full system control (global config, memory management,
               permission policies, approval rules, self-improvement approval,
               administrative settings)
    ADMIN    — Project-scoped admin (memory management, permission policies
               within project, but not global config)
    USER     — Standard user — can request tasks, limited memory operations
    GUEST    — Read-only — view memory, no modifications

The creator identity is stored in a protected JSON config file, not in
chat messages or hardcoded usernames.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from evora.logger import Logger


class AuthorityLevel(str, Enum):
    """Authority levels for EVORA identity system."""

    CREATOR = "creator"
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

    @classmethod
    def hierarchy(cls) -> list["AuthorityLevel"]:
        """Return authority levels in ascending order of power."""
        return [cls.GUEST, cls.USER, cls.ADMIN, cls.CREATOR]

    def can_access(self, required: "AuthorityLevel") -> bool:
        """Check if this level meets or exceeds the required level."""
        hierarchy = self.hierarchy()
        return hierarchy.index(self) >= hierarchy.index(required)


# Actions that require specific authority levels.
# These are checked during administrative operations.
AUTHORITY_RULES: dict[str, AuthorityLevel] = {
    # Memory management
    "clear_task_memory": AuthorityLevel.ADMIN,
    "clear_project_memory": AuthorityLevel.ADMIN,
    "delete_long_term_memory": AuthorityLevel.ADMIN,
    "set_creator": AuthorityLevel.CREATOR,
    # Identity management
    "change_identity": AuthorityLevel.CREATOR,
    "create_identity": AuthorityLevel.CREATOR,
    "delete_identity": AuthorityLevel.CREATOR,
    # Configuration
    "modify_global_config": AuthorityLevel.CREATOR,
    "modify_permission_policy": AuthorityLevel.ADMIN,
    "set_approval_rules": AuthorityLevel.ADMIN,
    # Self-improvement / safety
    "enable_self_modification": AuthorityLevel.CREATOR,
    "prohibit_self_modification": AuthorityLevel.CREATOR,
    # Long-term memory
    "remember": AuthorityLevel.USER,
    "forget": AuthorityLevel.ADMIN,
    "list_memories": AuthorityLevel.GUEST,
    "retrieve_memories": AuthorityLevel.GUEST,
}


@dataclass
class Identity:
    """A user identity with an authority level.

    The creator identity is protected — it is loaded from a separate
    configuration file, not from conversational text or username comparison.
    """

    id: str
    name: str
    authority: AuthorityLevel
    created_at: str
    config_path: Optional[str] = None

    @classmethod
    def create(cls, name: str, authority: AuthorityLevel = AuthorityLevel.USER) -> "Identity":
        """Create a new identity with a unique ID."""
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            authority=authority,
            created_at=datetime.now().isoformat(),
        )

    @classmethod
    def create_creator(cls, name: str = "creator") -> "Identity":
        """Create a creator identity.

        This does NOT hardcode a username. The creator designation comes
        from the authority level, which is set via the protected config file.
        """
        return cls(
            id=str(uuid.uuid4()),
            name=name,
            authority=AuthorityLevel.CREATOR,
            created_at=datetime.now().isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "authority": self.authority.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Identity":
        return cls(
            id=data["id"],
            name=data["name"],
            authority=AuthorityLevel(data.get("authority", AuthorityLevel.USER.value)),
            created_at=data.get("created_at", datetime.now().isoformat()),
            config_path=data.get("config_path"),
        )

    @property
    def is_creator(self) -> bool:
        return self.authority == AuthorityLevel.CREATOR


class IdentityStore:
    """Protected storage for identity configurations.

    Uses JSON files behind an abstraction. The 'current identity' is stored
    in a protected config file (separate from conversational memory).

    The creator identity is identified by its authority level, NOT by
    username comparison — there is no: if username == "Tony": creator = True
    """

    def __init__(self, identity_dir: str):
        self.identity_dir = Path(identity_dir)
        self.identity_dir.mkdir(parents=True, exist_ok=True)
        self._identities_dir = self.identity_dir / "identities"
        self._identities_dir.mkdir(parents=True, exist_ok=True)
        self._current_path = self.identity_dir / "current.json"
        self._creator_path = self.identity_dir / "creator.json"

    def save_identity(self, identity: Identity) -> str:
        """Persist an identity to the store."""
        path = self._identities_dir / f"{self._safe_name(identity.id)}.json"
        with open(path, "w") as f:
            json.dump(identity.to_dict(), f, indent=2)
        return str(path)

    def load_identity(self, identity_id: str) -> Optional[Identity]:
        """Load an identity by ID."""
        path = self._identities_dir / f"{self._safe_name(identity_id)}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return Identity.from_dict(data)

    def list_identities(self) -> list[Identity]:
        """List all stored identities."""
        identities = []
        for path in sorted(self._identities_dir.glob("*.json")):
            with open(path, "r") as f:
                data = json.load(f)
            identities.append(Identity.from_dict(data))
        return identities

    def get_current(self) -> Identity:
        """Get the current identity.

        Falls back to a default GUEST identity if no current identity
        is configured. The creator must be set explicitly via the
        protected creator config file.
        """
        if self._current_path.exists():
            with open(self._current_path, "r") as f:
                data = json.load(f)
            return Identity.from_dict(data)

        # Check for a creator config
        if self._creator_path.exists():
            with open(self._creator_path, "r") as f:
                data = json.load(f)
            return Identity.from_dict(data)

        return Identity.create(
            name="guest", authority=AuthorityLevel.GUEST
        )

    def set_current(self, identity: Identity) -> None:
        """Set the current identity.

        Requires the calling identity to have CREATOR authority.
        """
        with open(self._current_path, "w") as f:
            json.dump(identity.to_dict(), f, indent=2)

    def get_creator(self) -> Optional[Identity]:
        """Get the creator identity from the protected config.

        Returns None if no creator is configured.
        """
        if self._creator_path.exists():
            with open(self._creator_path, "r") as f:
                data = json.load(f)
            return Identity.from_dict(data)
        return None

    def set_creator(self, identity: Identity) -> None:
        """Set the creator identity.

        The creator identity is stored in a separate protected file.
        This does NOT compare usernames — it sets authority explicitly.
        """
        if not identity.is_creator:
            identity = Identity(
                id=identity.id,
                name=identity.name,
                authority=AuthorityLevel.CREATOR,
                created_at=identity.created_at,
            )
        identity.config_path = str(self._creator_path)
        with open(self._creator_path, "w") as f:
            json.dump(identity.to_dict(), f, indent=2)

    def bootstrap_creator(self, name: str) -> Identity:
        """First-time setup: create and store the initial creator identity.

        This does NOT compare usernames. It creates an identity with
        CREATOR authority and stores it in the protected creator file.

        Can only be called when no creator identity is configured yet.
        Subsequent creator changes require CREATOR authority.
        """
        if self._creator_path.exists():
            with open(self._creator_path, "r") as f:
                data = json.load(f)
            existing = Identity.from_dict(data)
            if existing.is_creator:
                raise PermissionError(
                    "Creator identity already configured. "
                    "Only the current creator can change it."
                )

        identity = Identity.create_creator(name)
        self.set_creator(identity)
        self.set_current(identity)
        return identity

    def delete_identity(self, identity_id: str) -> bool:
        """Delete an identity by ID. Returns True if deleted."""
        path = self._identities_dir / f"{self._safe_name(identity_id)}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def is_authorized(self, identity: Identity, action: str) -> bool:
        """Check if an identity is authorized to perform an action."""
        required_level = AUTHORITY_RULES.get(action)
        if required_level is None:
            return True
        return identity.authority.can_access(required_level)

    @staticmethod
    def _safe_name(name: str) -> str:
        """Sanitize a string for use as a filename."""
        safe = "".join(c for c in name if c.isalnum() or c in "-_")
        return safe or "unnamed"


class IdentityService:
    """Runtime identity and authority checks for the agent loop.

    Wraps IdentityStore with permission enforcement. All authority checks
    go through this service — callers never compare usernames.
    """

    def __init__(
        self,
        store: Optional[IdentityStore] = None,
        logger: Optional[Logger] = None,
        identity_dir: Optional[str] = None,
    ):
        if store is not None:
            self.store = store
        else:
            from evora.config import load_config

            config = load_config()
            dir_path = identity_dir or config.identity_dir
            self.store = IdentityStore(dir_path)
        self.logger = logger

    def current_identity(self) -> Identity:
        """Get the current acting identity."""
        identity = self.store.get_current()
        if self.logger:
            self.logger.info(f"Current identity: {identity.name} ({identity.authority.value})")
        return identity

    def check_authority(self, action: str, level: Optional[AuthorityLevel] = None) -> bool:
        """Check if the current identity can perform the given action.

        If `level` is provided, checks against that AuthorityLevel directly.
        If only `action` is provided, looks up the required level from
        AUTHORITY_RULES.
        """
        identity = self.current_identity()

        if level is not None:
            return identity.authority.can_access(level)

        return self.store.is_authorized(identity, action)

    def require_authority(self, action: str, level: Optional[AuthorityLevel] = None) -> Identity:
        """Require the current identity to have a specific authority.

        Raises PermissionError if unauthorized.
        Returns the current identity if authorized.
        """
        if self.check_authority(action, level):
            return self.current_identity()

        identity = self.current_identity()
        raise PermissionError(
            f"Identity '{identity.name}' (authority={identity.authority.value}) "
            f"is not authorized to perform '{action}'. "
            f"Required: {level or AUTHORITY_RULES.get(action, 'unknown').value}"
        )

    def is_creator(self) -> bool:
        """Check if the current identity has CREATOR authority."""
        return self.current_identity().is_creator

    def get_creator(self) -> Optional[Identity]:
        """Get the creator identity if configured."""
        return self.store.get_creator()

    def bootstrap_creator(self, name: str) -> Identity:
        """First-time setup for the initial creator identity.

        This is the ONLY way to set the first creator — it bypasses
        authority checks since no creator exists yet. Subsequent
        creator changes require existing CREATOR authority.
        """
        return self.store.bootstrap_creator(name)
