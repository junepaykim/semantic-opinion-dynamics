"""LLM API: persona and opinion generation. OpenAI first, Ollama fallback. Retries on failure."""
import json
import os
import random
import requests
import time
from pathlib import Path

MAX_RETRIES = 5
TOPIC = "Remote Work v.s. Return-to-Office"  # work-from-home vs work-from-office
OLLAMA_MODEL = "qwen3:4b"
OPENAI_MODEL = "gpt-3.5-turbo"

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_API_KEY_FILE = _PROJECT_ROOT / "api_key.txt"


_API_KEY_CREATED = False


def _load_api_key() -> str | None:
    """Load API key from api_key.txt or OPENAI_API_KEY env."""
    global _API_KEY_CREATED
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    if not _API_KEY_FILE.exists():
        if not _API_KEY_CREATED:
            _API_KEY_FILE.write_text(
                "# Put your OpenAI API key on the next line (e.g. sk-xxx...). Lines with # are ignored.\n"
                "# This file is in .gitignore and will not be uploaded to GitHub.\n\n",
                encoding="utf-8",
            )
            _API_KEY_CREATED = True
            print(f"[modelCall] api_key.txt not found, created at project root. Add your OpenAI API key to {_API_KEY_FILE}")
        return None
    text = _API_KEY_FILE.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    return None


def _call_openai(prompt: str) -> str:
    """Call OpenAI API. Raises on failure."""
    api_key = _load_api_key()
    if not api_key:
        raise ValueError("OpenAI API key not found. Put it in api_key.txt or set OPENAI_API_KEY.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    response = requests.post(url, json=payload, headers=headers, timeout=60)
    if response.status_code != 200:
        raise Exception(f"OpenAI HTTP {response.status_code}: {response.text[:200]}")
    result = response.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content")
    if not content or not str(content).strip():
        raise ValueError("OpenAI returned empty content")
    return str(content).strip()


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama local API."""
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(f"Ollama HTTP {response.status_code}: {response.text[:200]}")
    result = response.json()
    content = result.get("message", {}).get("content")
    if content is None:
        raise ValueError("Ollama response missing content")
    return str(content).strip()


_FALLBACK_PRINTED = False


def _call_with_retry(call_fn, prompt: str, name: str) -> str:
    """Retry call_fn(prompt) up to MAX_RETRIES times."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return call_fn(prompt)
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
    print(f"[modelCall] {name} failed after {MAX_RETRIES} retries: {last_error}")
    raise last_error


def _call_llm(prompt: str) -> str:
    """Call LLM: OpenAI first, Ollama fallback. Both retry on failure."""
    global _FALLBACK_PRINTED
    api_key = _load_api_key()
    if api_key:
        try:
            return _call_with_retry(_call_openai, prompt, "OpenAI")
        except Exception as e:
            if not _FALLBACK_PRINTED:
                print(f"[modelCall] OpenAI unavailable, falling back to Ollama: {e}")
                _FALLBACK_PRINTED = True
            return _call_with_retry(_call_ollama, prompt, "Ollama")
    return _call_with_retry(_call_ollama, prompt, "Ollama")


def generatePersona(opinionScore: float | None = None) -> str:
    """Generate comma-separated persona tags from opinion score [0,1]."""
    if opinionScore is None:
        opinionScore = random.random()
        num = random.randint(3, 5)
    else:
        num = random.randint(6, 10)
    prompt = (
        f"On a personality scale 0 (gentle) to 1 (firm), score={opinionScore}. "
        f"Generate {num} adjectives/phrases. Output comma-separated list only."
    )
    return _call_llm(prompt)


def generateOpinionPrompt(opinionScore: float, persona: str, topic: str = TOPIC) -> str:
    """Generate first-person opinion paragraph (≤50 words) from score and persona."""
    prompt = (
        f"Scale 0=remote work, 1=office. Score={opinionScore}. Persona: {persona}. "
        f"Write a short first-person paragraph (≤50 words). Output paragraph only."
    )
    return _call_llm(prompt)


def updateNodeOpinion(
    persona: str,
    current_score: float,
    current_prompt: str,
    neighbor_info: list[tuple[str, float]],
    topic: str = TOPIC,
) -> tuple[float, str]:
    """Update opinion via LLM from persona, current state, and neighbor opinions. Returns (score, prompt)."""
    prompt = (
        f"Topic: {topic}. Scale 0 = strongly prefer remote work, 1 = strongly prefer office.\n\n"
        f"Your personality: {persona}\n\n"
        f"Your current opinion (score {current_score:.3f}): \"{current_prompt}\"\n\n"
    )
    if neighbor_info:
        prompt += "Your neighbors' opinions (weighted by connection strength):\n"
        for i, (nprompt, w) in enumerate(neighbor_info, 1):
            prompt += f"  - (weight {w:.3f}): \"{nprompt}\"\n"
        prompt += (
            "\nConsider these opinions and update your own. Output a JSON object with exactly two keys:\n"
            '  "opinionScore": float between 0 and 1\n'
            '  "prompt": string, first-person, at most 50 words, describing your updated opinion\n'
            "Only output the JSON, no other text."
        )
    else:
        prompt += (
            "You have no neighbors. Reaffirm your opinion. Output a JSON object with exactly two keys:\n"
            '  "opinionScore": float between 0 and 1 (same as current)\n'
            '  "prompt": string, first-person, at most 50 words\n'
            "Only output the JSON, no other text."
        )

    raw = _call_llm(prompt)
    score, promptText = _parseUpdateResponse(raw, current_score, current_prompt)
    return (score, promptText)


def _parseUpdateResponse(raw: str, fallbackScore: float, fallbackPrompt: str) -> tuple[float, str]:
    """Parse JSON from LLM response; return fallbacks on parse failure."""
    raw = raw.strip()
    for s in (raw, raw.split("```")[0].strip(), raw.split("\n")[-1]):
        if not s:
            continue
        try:
            obj = json.loads(s)
            score = float(obj.get("opinionScore", fallbackScore))
            promptText = str(obj.get("prompt", fallbackPrompt)).strip() or fallbackPrompt
            score = max(0.0, min(1.0, score))
            return (score, promptText)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return (fallbackScore, fallbackPrompt)

