from __future__ import annotations

import json
from typing import List, Sequence

from semantic_opinion_dynamics.contracts import OpinionState


def build_system_prompt(*, persona: str, behavior_rules: Sequence[str]) -> str:
    rules = "\n".join(f"- {r}" for r in behavior_rules) if behavior_rules else ""
    return f"""You are an employee in a company internal discussion.

Persona:
{persona}

Behavior rules:
{rules}
""".strip()


def build_user_prompt(
    *,
    policy_question: str,
    self_state: OpinionState,
    neighbors: List[dict],
    json_schema: dict,
) -> str:
    context = {
        "policy_question": policy_question,
        "self_state": self_state.model_dump(),
        "neighbors": neighbors,
    }
    context_str = json.dumps(context, ensure_ascii=False, indent=2)

    schema_str = json.dumps(json_schema, ensure_ascii=False, indent=2)

    return f"""Context JSON (read-only):
{context_str}

Task:
Update your opinion after reading neighbors. You may keep your stance or change it.

Output format:
Return ONLY a JSON object that matches this JSON Schema (no markdown, no extra keys):
{schema_str}
""".strip()
