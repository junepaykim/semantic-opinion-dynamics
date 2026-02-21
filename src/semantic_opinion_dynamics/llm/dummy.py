from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from semantic_opinion_dynamics.contracts import Stance, Tone
from semantic_opinion_dynamics.llm.base import LLMClient, LLMError


def _extract_context(user_prompt: str) -> Dict[str, Any]:
    marker = "Context JSON (read-only):"
    if marker not in user_prompt:
        raise LLMError("dummy client: missing context block")
    after = user_prompt.split(marker, 1)[1].lstrip()
    if "\n\nTask:" not in after:
        raise LLMError("dummy client: context delimiter not found")
    context_str = after.split("\n\nTask:", 1)[0].strip()
    return json.loads(context_str)


def _persona_bias(system_prompt: str) -> Dict[Stance, float]:
    txt = system_prompt.lower()
    bias: Dict[Stance, float] = {s: 0.0 for s in Stance}

    anti_kw = ["deep work", "remote", "autonomy", "commut", "async"]
    pro_kw = ["in-person", "in person", "collaboration", "culture", "whiteboard", "onsite", "on-site"]
    hr_kw = ["fairness", "equitable", "legal", "dei", "accessibility", "policy clarity"]

    if any(k in txt for k in anti_kw):
        bias[Stance.anti_rto] += 0.25
    if any(k in txt for k in pro_kw):
        bias[Stance.pro_rto] += 0.25
    if any(k in txt for k in hr_kw):
        bias[Stance.neutral] += 0.20
        bias[Stance.hybrid] += 0.10

    if "leadership" in txt and "bot" in txt:
        bias[Stance.pro_rto] += 0.80

    return bias


def _collect_scores(self_state: Dict[str, Any], neighbors: List[Dict[str, Any]], bias: Dict[Stance, float]) -> Dict[Stance, float]:
    scores: Dict[Stance, float] = {s: 0.0 for s in Stance}

    self_stance = Stance(self_state["stance"])
    self_conf = float(self_state["confidence"])
    scores[self_stance] += 0.6 * self_conf

    for nb in neighbors:
        st = Stance(nb["state"]["stance"])
        conf = float(nb["state"]["confidence"])
        w = float(nb.get("weight", 1.0))
        scores[st] += w * conf

    for st, b in bias.items():
        scores[st] += b

    return scores


def _pick_stance(scores: Dict[Stance, float]) -> Stance:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, second = ranked[0], ranked[1]
    if top[1] - second[1] < 0.25 and top[0] != Stance.neutral:
        return Stance.hybrid
    return top[0]


def _tone_from_change(prev: Stance, nxt: Stance, neighbor_tones: List[str]) -> Tone:
    if "aggressive" in neighbor_tones:
        return Tone.conciliatory
    if prev != nxt:
        return Tone.analytical
    if prev in (Stance.neutral, Stance.hybrid):
        return Tone.conciliatory
    return Tone.analytical


def _arguments(self_args: List[str], neighbors: List[Dict[str, Any]], stance: Stance, rng: random.Random) -> List[str]:
    pool: List[str] = []
    for nb in neighbors:
        if Stance(nb["state"]["stance"]) == stance:
            pool.extend(nb["state"].get("key_arguments", []))
    pool.extend(self_args or [])
    uniq = []
    for a in pool:
        if a and a not in uniq:
            uniq.append(a)
    if not uniq:
        uniq = ["Collaboration trade-offs", "Productivity considerations"]
    rng.shuffle(uniq)
    return uniq[: min(6, len(uniq))]


def _write_opinion(policy: str, stance: Stance, tone: Tone, args: List[str]) -> str:
    head = {
        Stance.pro_rto: "I'm supportive of moving toward more in-office time.",
        Stance.anti_rto: "I'm concerned a mandatory 4-day RTO will backfire.",
        Stance.hybrid: "I think we should land on a structured hybrid approach.",
        Stance.neutral: "I see pros and cons and want more clarity before deciding.",
    }[stance]

    tone_hint = {
        Tone.analytical: "From a practical standpoint",
        Tone.conciliatory: "Trying to balance perspectives",
        Tone.aggressive: "Frankly",
        Tone.empathetic: "I empathize with different needs",
        Tone.neutral: "Overall",
    }[tone]

    bullet_args = "; ".join(args[:3])
    body = f"{tone_hint}, on '{policy}', key points are: {bullet_args}." if bullet_args else ""
    return f"{head} {body}".strip()


class DummyLLMClient(LLMClient):
    def __init__(self, *, model: str, temperature: float = 0.7, seed: Optional[int] = None) -> None:
        super().__init__(model=model, temperature=temperature, seed=seed)
        self._rng = random.Random(seed)

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        ctx = _extract_context(user_prompt)
        self_state = ctx["self_state"]
        neighbors: List[Dict[str, Any]] = ctx.get("neighbors", [])

        bias = _persona_bias(system_prompt)
        scores = _collect_scores(self_state, neighbors, bias)
        prev = Stance(self_state["stance"])
        nxt = _pick_stance(scores)

        neighbor_tones = [str(nb["state"].get("tone", "")) for nb in neighbors]
        tone = _tone_from_change(prev, nxt, neighbor_tones)

        args = _arguments(self_state.get("key_arguments", []), neighbors, nxt, self._rng)

        base_conf = max(0.2, min(0.98, (scores[nxt] / (sum(scores.values()) + 1e-6))))
        jitter = (self._rng.random() - 0.5) * 0.15 * max(0.2, self.temperature)
        conf = float(max(0.0, min(1.0, base_conf + jitter)))

        updated = _write_opinion(ctx.get("policy_question", ""), nxt, tone, args)

        return {
            "stance": nxt.value,
            "confidence": conf,
            "tone": tone.value,
            "key_arguments": args[:6],
            "updated_opinion": updated,
            "reply_to_neighbors": True,
        }
