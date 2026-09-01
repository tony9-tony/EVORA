"""
Tests for the EVORA logger module.
"""

import logging
import sys

import pytest

from evora.logger import Logger, Stage, STAGE_COLORS, STAGE_EMOJIS


class TestLogger:

    def test_logger_creation(self):
        logger = Logger("test_logger", "debug")
        assert logger is not None

    def test_logger_levels(self):
        logger = Logger("test_levels", "debug")
        logger.debug("debug message")
        logger.info("info message")
        logger.warn("warn message")
        logger.error("error message")
        logger.success("success message")

    def test_stage_enum(self):
        assert Stage.PLAN == "PLAN"
        assert Stage.ASK == "ASK"
        assert Stage.CODE == "CODE"
        assert Stage.TEST == "TEST"
        assert Stage.FIX == "FIX"
        assert Stage.SUCCESS == "SUCCESS"
        assert Stage.ERROR == "ERROR"

    def test_stage_colors_exist(self):
        assert Stage.PLAN in STAGE_COLORS
        assert Stage.ASK in STAGE_COLORS
        assert Stage.CODE in STAGE_COLORS
        assert Stage.TEST in STAGE_COLORS
        assert Stage.FIX in STAGE_COLORS
        assert Stage.SUCCESS in STAGE_COLORS
        assert Stage.ERROR in STAGE_COLORS

    def test_stage_emojis_exist(self):
        assert Stage.PLAN in STAGE_EMOJIS
        assert Stage.ASK in STAGE_EMOJIS
        assert Stage.CODE in STAGE_EMOJIS

    def test_logger_has_get_logger(self):
        logger = Logger("test_get", "info")
        assert logger.get_logger() is not None
