"""
Tests for the EVORA config module.
"""

import os
from pathlib import Path

import pytest

from evora.config import load_config, Config, ProviderConfig, PermissionConfig


class TestConfig:

    def test_default_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = load_config()
        assert config.model == "gpt-4o"
        assert config.workspace_dir is not None
        assert config.log_level == "INFO"

    def test_config_with_env_override(self, tmp_path, monkeypatch):
        evora_dir = tmp_path / ".evora"
        evora_dir.mkdir()
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("EVORA_MODEL", "gpt-3.5-turbo")
        monkeypatch.setenv("EVORA_API_KEY", "test-key-123")
        config = load_config()
        assert config.model == "gpt-3.5-turbo"
        assert config.api_key == "test-key-123"

    def test_permission_config_defaults(self):
        pc = PermissionConfig()
        assert pc.allow_file_write is True
        assert pc.allow_cmd_exec is True
        assert pc.allowed_cmds == []

    def test_provider_config_defaults(self):
        pc = ProviderConfig(name="openai", model="gpt-4o")
        assert pc.name == "openai"
        assert pc.model == "gpt-4o"
