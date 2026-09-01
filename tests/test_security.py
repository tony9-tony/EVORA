"""
Tests for the EVORA security and permission system.
"""

import pytest

from evora.security import PermissionManager, PermissionLevel


class TestPermissionManager:

    def test_safe_command(self):
        assert PermissionManager.check_command_safety("ls") == PermissionLevel.SAFE
        assert PermissionManager.check_command_safety("pwd") == PermissionLevel.SAFE
        assert PermissionManager.check_command_safety("echo hello") == PermissionLevel.SAFE

    def test_dangerous_command(self):
        assert PermissionManager.check_command_safety("rm -rf /") == PermissionLevel.DANGEROUS
        assert PermissionManager.check_command_safety("rm -rf ~") == PermissionLevel.DANGEROUS
        assert PermissionManager.check_command_safety("dd if=/dev/zero of=/dev/sda") == PermissionLevel.DANGEROUS
        assert PermissionManager.check_command_safety("format C:") == PermissionLevel.DANGEROUS
        assert PermissionManager.check_command_safety("drop database mydb") == PermissionLevel.DANGEROUS

    def test_ask_command(self):
        assert PermissionManager.check_command_safety("pip install requests") == PermissionLevel.ASK
        assert PermissionManager.check_command_safety("apt install vim") == PermissionLevel.ASK
        assert PermissionManager.check_command_safety("docker run nginx") == PermissionLevel.ASK
        assert PermissionManager.check_command_safety("sudo make install") == PermissionLevel.ASK

    def test_workspace_path_check(self, tmp_path):
        pm = PermissionManager(workspace_dir=str(tmp_path))
        safe_path = str(tmp_path / "test.txt")
        assert pm.check_workspace_path(safe_path) is not None

    def test_workspace_path_outside(self, tmp_path):
        pm = PermissionManager(workspace_dir=str(tmp_path))
        with pytest.raises(PermissionError):
            pm.check_workspace_path("/etc/passwd")

    def test_command_timeout_short(self):
        timeout = PermissionManager.check_command_timeout("echo hello")
        assert timeout > 0

    def test_command_timeout_long(self):
        timeout = PermissionManager.check_command_timeout("pip install requests")
        assert timeout > 60
