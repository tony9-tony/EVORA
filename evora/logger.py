"""
Structured logging for EVORA.

Provides colored console output and optional file logging with stage-based formatting.

Example:
    [PLAN] Analyzing request...
    [CODE] Creating file /path/to/file.py
    [TEST] Running tests...
    [SUCCESS] All tests passed.
"""

import logging
import sys
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from enum import Enum
from pathlib import Path
from typing import Optional

from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)


class Stage(str, Enum):
    PLAN = "PLAN"
    ASK = "ASK"
    CODE = "CODE"
    TEST = "TEST"
    FIX = "FIX"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    INFO = "INFO"
    WARN = "WARN"


STAGE_COLORS = {
    Stage.PLAN: Fore.CYAN,
    Stage.ASK: Fore.YELLOW,
    Stage.CODE: Fore.GREEN,
    Stage.TEST: Fore.BLUE,
    Stage.FIX: Fore.MAGENTA,
    Stage.SUCCESS: Fore.GREEN,
    Stage.ERROR: Fore.RED,
    Stage.INFO: Fore.WHITE,
    Stage.WARN: Fore.YELLOW,
}

STAGE_EMOJIS = {
    Stage.PLAN: "[📋]",
    Stage.ASK: "[❓]",
    Stage.CODE: "[⚙️]",
    Stage.TEST: "[🧪]",
    Stage.FIX: "[🔧]",
    Stage.SUCCESS: "[✅]",
    Stage.ERROR: "[❌]",
    Stage.INFO: "[ℹ️]",
    Stage.WARN: "[⚠️]",
}


class StageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", "INFO")
        color = STAGE_COLORS.get(stage, Fore.WHITE)
        emoji = STAGE_EMOJIS.get(stage, "[ℹ️]")
        record.msg = f"{color}{emoji} {stage:<8} {Style.RESET_ALL}{record.getMessage()}"
        return record.msg


class Logger:
    def __init__(self, name: str = "evora", level: str = "INFO", log_file: Optional[str] = None):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers = []

        formatter = StageFormatter()

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(stage)s] %(levelname)s: %(message)s"
            ))
            self._logger.addHandler(file_handler)

    def _log(self, stage: Stage, msg: str, **kwargs):
        self._logger.info(msg, extra={"stage": stage.value, **kwargs})

    def plan(self, msg: str):
        self._log(Stage.PLAN, msg)

    def ask(self, msg: str):
        self._log(Stage.ASK, msg)

    def code(self, msg: str):
        self._log(Stage.CODE, msg)

    def test(self, msg: str):
        self._log(Stage.TEST, msg)

    def fix(self, msg: str):
        self._log(Stage.FIX, msg)

    def success(self, msg: str):
        self._log(Stage.SUCCESS, msg)

    def error(self, msg: str):
        self._log(Stage.ERROR, msg)

    def info(self, msg: str):
        self._log(Stage.INFO, msg)

    def warn(self, msg: str):
        self._log(Stage.WARN, msg)

    def debug(self, msg: str):
        self._logger.debug(msg)

    def get_logger(self) -> logging.Logger:
        return self._logger
