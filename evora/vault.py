"""
Encrypted vault for the owner's secrets (tokens, app passwords).

The owner unlocks it with a passphrase in the web console; the key lives only in memory and is
forgotten when EVORA stops. Tools read secrets through `vault.get()` at the moment they act, so the
model never sees raw values (and the audit log redacts them).
"""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ITERATIONS = 200_000
CHECK_TEXT = b"evora-vault-v1"


class VaultLocked(RuntimeError):
    pass


class Vault:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._fernet: Optional[Fernet] = None
        self._lock = threading.Lock()

    def _derive(self, passphrase: str, salt: bytes) -> Fernet:
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=ITERATIONS)
        return Fernet(base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8"))))

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(self.path)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    @property
    def unlocked(self) -> bool:
        return self._fernet is not None

    def unlock(self, passphrase: str) -> bool:
        """Create the vault on first use, otherwise check the passphrase."""
        if not passphrase or len(passphrase) < 8:
            raise ValueError("Passphrase must be at least 8 characters.")
        with self._lock:
            if not self.exists:
                salt = os.urandom(16)
                f = self._derive(passphrase, salt)
                self._write({"salt": base64.b64encode(salt).decode(), "check": f.encrypt(CHECK_TEXT).decode(),
                             "items": {}})
                self._fernet = f
                return True
            data = self._read()
            f = self._derive(passphrase, base64.b64decode(data["salt"]))
            try:
                if f.decrypt(data["check"].encode()) != CHECK_TEXT:
                    return False
            except InvalidToken:
                return False
            self._fernet = f
            return True

    def lock(self) -> None:
        with self._lock:
            self._fernet = None

    def _need(self) -> Fernet:
        if self._fernet is None:
            raise VaultLocked("The vault is locked. Unlock it in the EVORA console (Keys tab).")
        return self._fernet

    def set(self, name: str, value: str) -> None:
        with self._lock:
            f = self._need()
            data = self._read()
            data["items"][name] = f.encrypt(value.encode("utf-8")).decode()
            self._write(data)

    def get(self, name: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            f = self._need()
            token = self._read()["items"].get(name)
            if token is None:
                return default
            return f.decrypt(token.encode()).decode("utf-8")

    def delete(self, name: str) -> bool:
        with self._lock:
            self._need()
            data = self._read()
            existed = data["items"].pop(name, None) is not None
            self._write(data)
            return existed

    def names(self) -> list[str]:
        """Names only, never values."""
        if not self.exists:
            return []
        return sorted(self._read()["items"].keys())

    def status(self) -> dict:
        return {"exists": self.exists, "unlocked": self.unlocked, "names": self.names()}
