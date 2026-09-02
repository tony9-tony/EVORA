"""
Reasoning engine for EVORA Phase 7.

Provides a model-agnostic reasoning abstraction that allows EVORA to
reason through development problems. Produces concise reasoning summaries
and decision metadata without persisting hidden chain-of-thought.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from evora.logger import Logger


@dataclass
class ReasoningContext:
    """Context for a reasoning session."""

    objective: str
    observations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    candidate_approaches: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningResult:
    """Result of a reasoning session."""

    summary: str
    selected_approach: str
    next_action: str
    confidence: float
    risks: list[str] = field(default_factory=list)
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "selected_approach": self.selected_approach,
            "confidence": self.confidence,
            "risks": self.risks,
            "next_action": self.next_action,
            "metadata": self.metadata,
        }


class ReasoningEngine:
    """Model-agnostic reasoning abstraction for EVORA.

    Uses the existing ModelManager to query the LLM for reasoning.
    Does not persist chain-of-thought — only concise summaries and
    decision metadata.
    """

    REASONING_PROMPT = """
You are EVORA's reasoning engine. Given an objective and context,
produce a structured reasoning summary.

Respond with JSON only:
{{
  "summary": "Brief reasoning summary (1-3 sentences)",
  "selected_approach": "The chosen approach",
  "confidence": 0.8,
  "risks": ["risk1", "risk2"],
  "next_action": "The immediate next action to take"
}}

Constraints:
{constraints}
Assumptions:
{assumptions}
Candidate approaches:
{candidates}
"""

    def __init__(self, model_manager: Any, logger: Optional[Logger] = None):
        self.model_manager = model_manager
        self.logger = logger

    async def reason(self, context: ReasoningContext) -> ReasoningResult:
        """Reason about a development problem and return a structured result."""
        constraints_text = "\n".join(f"- {c}" for c in context.constraints) or "- None"
        assumptions_text = "\n".join(f"- {a}" for a in context.assumptions) or "- None"
        candidates_text = "\n".join(f"- {c}" for c in context.candidate_approaches) or "- None"

        prompt = self.REASONING_PROMPT.format(
            constraints=constraints_text,
            assumptions=assumptions_text,
            candidates=candidates_text,
        )

        if self.logger:
            self.logger.reason(f"Reasoning about: {context.objective[:100]}")

        raw_response = ""
        try:
            from evora.model import ChatRequest, Message, Role
            request = ChatRequest(
                messages=[
                    Message(role=Role.SYSTEM, content=prompt),
                    Message(role=Role.USER, content=f"Objective: {context.objective}\n\nObservations:\n" + "\n".join(f"- {o}" for o in context.observations)),
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            response = await self.model_manager.chat(request)
            raw_response = response.content or ""
        except Exception as e:
            if self.logger:
                self.logger.warn(f"Reasoning model call failed: {e}")
            raw_response = ""

        return self._parse_response(raw_response, context)

    def _parse_response(self, raw: str, context: ReasoningContext) -> ReasoningResult:
        """Parse model response into a ReasoningResult.

        Malformed or missing responses are treated as hard failures, not silent fallbacks.
        """
        if not raw:
            return ReasoningResult(
                summary=f"Reasoning aborted: no model response for {context.objective}",
                selected_approach="abort",
                confidence=0.0,
                risks=["No model response available"],
                next_action="abort",
                raw_response=raw,
                metadata=context.metadata,
            )

        try:
            json_match = raw.find("{")
            if json_match < 0:
                raise ValueError("No JSON object found in response")

            data = json.loads(raw[json_match:])
            required_fields = ["summary", "selected_approach", "confidence", "next_action"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                raise ValueError(f"Missing required reasoning fields: {missing}")

            confidence = float(data.get("confidence", 0.5))
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError(f"Confidence out of bounds: {confidence}")

            return ReasoningResult(
                summary=data["summary"],
                selected_approach=data["selected_approach"],
                confidence=confidence,
                risks=data.get("risks", []),
                next_action=data["next_action"],
                raw_response=raw,
                metadata=context.metadata,
            )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Reasoning response parsing failed: {e}")
            return ReasoningResult(
                summary=f"Reasoning aborted: {e}",
                selected_approach="abort",
                confidence=0.0,
                risks=[f"Malformed reasoning response: {e}"],
                next_action="abort",
                raw_response=raw,
                metadata=context.metadata,
            )
