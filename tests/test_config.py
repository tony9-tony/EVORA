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

    def test_provider_field_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = load_config()
        assert config.provider == ""

    def test_provider_field_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("EVORA_PROVIDER", "ollama")
        config = load_config()
        assert config.provider == "ollama"

    def test_ollama_provider_config_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = load_config()
        assert "ollama" in config.providers
        ollama = config.providers["ollama"]
        assert ollama.model == "qwen2.5-coder:latest"
        assert "127.0.0.1:11434" in ollama.base_url

    def test_ollama_provider_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("EVORA_OLLAMA_MODEL", "custom-model")
        monkeypatch.setenv("EVORA_OLLAMA_BASE_URL", "http://localhost:11434/v1")
        config = load_config()
        assert config.providers["ollama"].model == "custom-model"
        assert config.providers["ollama"].base_url == "http://localhost:11434/v1"

    def test_default_config_includes_ollama(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        config = load_config()
        # Auto-select should prefer ollama when no openai key
        assert "ollama" in config.providers
