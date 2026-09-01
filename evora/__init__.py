"""
EVORA - Standalone AI Software Development Engine

EVORA is an AI coding engine that accepts user requests, creates structured plans,
asks for approval, writes code, tests it, attempts fixes, and saves task memory.

Core workflow:
    USER -> PLAN -> ASK/APPROVAL -> CODE -> TEST -> FIX -> REPORT -> MEMORY
"""

__version__ = "0.1.0"
__all__ = ["core", "models", "tools", "security", "memory", "planner", "analyzer"]
