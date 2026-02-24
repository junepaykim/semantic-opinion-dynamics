"""
LLM API calls: generate opinion descriptions from node opinion scores.
Prefer Ollama local model for implementation.
"""
import random
import requests
import json

TOPIC = "Remote Work v.s. Return-to-Office" # work-from-home vs work-from-office
MODEL = "qwen3:4b"

def generatePersona(opinionScore: float | None = None) -> str:
    """
    Call API to randomly generate persona description tags.

    Args:
        opinionScore: Optional, float [0, 1], node opinion score. If provided, generates more fitting persona.

    Returns:
        String of persona words, comma-separated, at most 10 words. E.g. "gentle, introverted, prefers solitude, rational, cautious"
    """
    # Get prompt
    if opinionScore is None:
        opinionScore = random.random()
        num = random.randint(3, 5)
    else:
        num = random.randint(6, 10)
    prompt = f'On a personality scale from 0 (extremely gentle) to 1 (extremely firm), a person scores {opinionScore}. Generate {num} adjectives (or short descriptive phrases) that reflect this person\'s traits. Return them as a comma-separated list, for example: "trait1, trait2, trait3". Only output the list, with no additional text.'
    
    prompt = (
        f'On a personality scale from 0 (extremely gentle) to 1 (extremely firm), '
        f'a person scores {opinionScore}. '
        f'Generate {num} adjectives (or short descriptive phrases) that reflect this person\'s traits. '
        f'Return them as a comma-separated list, for example: "trait1, trait2, trait3". '
        f'Only output the list, with no additional text.'
    )

    # Get persona
    persona = call_ollama(prompt=prompt, model=MODEL)

    return persona


def generateOpinionPrompt(opinionScore: float, persona: str, topic: str = TOPIC) -> str:
    """
    Generate initial opinion prompt from opinion score and persona description.

    Args:
        opinionScore: float [0, 1], 0=prefer remote work, 1=prefer return-to-office, 0.5=both with preference
        persona: Persona tag string, e.g. "gentle, introverted, prefers solitude"
        topic: Topic, default "remote work vs return to office"

    Returns:
        Updated initial opinion content (first-person description, at most 50 words), e.g. "I prefer cats because..."
    """
    # Get prompt
    prompt = (
        f'On a scale from 0 (strongly prefers working from home) to 1 (strongly prefers working from office), '
        f'a person scores {opinionScore}. This person\'s personality traits are: {persona}. '
        f'Write a short first-person paragraph (under 50 words, at most 3 sentences) expressing this person\'s opinion and the reason behind it. '
        f'For example: "I prefer cats because it\'s cute." Only output the paragraph, with no additional text.'
    )
   
    # Get opinion
    opinion = call_ollama(prompt=prompt, model=MODEL)

    return opinion

def call_ollama(prompt: str, model: str) -> str:
    """
    Use prompt to generate results from Ollama

    Args:
        prompt: a paragragh telling Ollama what to generate
        model: ollama model used for generating results

    Returns:
        String of words or paragrah from Ollama
    """
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        return result["message"]["content"]
    else:
        raise Exception(f"Failed: {response.text}")

