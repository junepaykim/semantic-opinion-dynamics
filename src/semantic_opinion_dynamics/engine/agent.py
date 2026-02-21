from __future__ import annotations

import asyncio
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import ValidationError

from semantic_opinion_dynamics.contracts import AgentLog, OpinionState, OpinionUpdate, PersonaSpec, PromptLog
from semantic_opinion_dynamics.llm.base import LLMClient
from semantic_opinion_dynamics.llm.prompts import build_system_prompt, build_user_prompt


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


@dataclass
class Agent:
    agent_id: str
    persona: PersonaSpec
    state: OpinionState

    def display_name(self) -> str:
        return self.persona.display_name

    def persona_fingerprint(self) -> str:
        return _short_hash(self.persona.persona)

    def is_frozen(self) -> bool:
        extra = getattr(self.persona, "model_extra", None) or {}
        return bool(extra.get("frozen", False))

    async def step(
        self,
        *,
        policy_question: str,
        neighbors: List[Tuple[str, str, OpinionState, float]],
        llm: LLMClient,
        behavior_rules: Sequence[str],
        include_neighbor_opinion_text: bool,
        max_neighbors_in_prompt: int,
        store_prompts: bool,
        max_retries: int,
    ) -> Tuple[OpinionState, AgentLog]:
        neighbors_sorted = sorted(neighbors, key=lambda x: x[3], reverse=True)
        if max_neighbors_in_prompt >= 0:
            neighbors_sorted = neighbors_sorted[:max_neighbors_in_prompt]

        neighbor_ids: List[str] = [nb_id for nb_id, _, _, _ in neighbors_sorted]

        if self.is_frozen():
            log = AgentLog(
                agent_id=self.agent_id,
                display_name=self.persona.display_name,
                state=self.state,
                neighbors=neighbor_ids,
                llm_update=None,
                prompt=None,
                error=None,
            )
            return self.state, log

        neighbors_payload: List[Dict] = []
        for nb_id, nb_name, nb_state, w in neighbors_sorted:
            st = nb_state.model_dump(mode="json")
            if not include_neighbor_opinion_text:
                st.pop("opinion_text", None)
            neighbors_payload.append(
                {
                    "neighbor_id": nb_id,
                    "display_name": nb_name,
                    "weight": w,
                    "state": st,
                }
            )

        json_schema = OpinionUpdate.model_json_schema()

        system_prompt = build_system_prompt(persona=self.persona.persona, behavior_rules=behavior_rules)
        user_prompt = build_user_prompt(
            policy_question=policy_question,
            self_state=self.state,
            neighbors=neighbors_payload,
            json_schema=json_schema,
        )

        error: Optional[str] = None
        update: Optional[OpinionUpdate] = None
        for attempt in range(max_retries + 1):
            try:
                raw = await llm.generate_json(system_prompt=system_prompt, user_prompt=user_prompt, json_schema=json_schema)
                update = OpinionUpdate.model_validate(raw)
                error = None
                break
            except (ValidationError, Exception) as e:
                error = f"{type(e).__name__}: {e}"
                update = None
                if attempt < max_retries:
                    backoff = min(2.0, 0.2 * (2**attempt))
                    jitter = random.random() * 0.1
                    await asyncio.sleep(backoff + jitter)

        if update is None:
            next_state = self.state
        else:
            next_state = OpinionState(
                stance=update.stance,
                confidence=update.confidence,
                tone=update.tone,
                key_arguments=update.key_arguments,
                opinion_text=update.updated_opinion,
            )

        log = AgentLog(
            agent_id=self.agent_id,
            display_name=self.persona.display_name,
            state=next_state,
            neighbors=neighbor_ids,
            llm_update=update,
            prompt=PromptLog(system=system_prompt, user=user_prompt, json_schema=json_schema) if store_prompts else None,
            error=error,
        )

        return next_state, log
