"""
LLM API calls: generate opinion descriptions from node opinion scores.
Prefer Ollama local model for implementation.
"""

TOPIC = "Like cats vs dogs"


def generatePersona(opinionScore: float | None = None) -> str:
    """
    Call API to randomly generate persona description tags.

    Args:
        opinionScore: Optional, float [0, 1], node opinion score. If provided, generates more fitting persona.

    Returns:
        String of persona words, comma-separated, at most 10 words. E.g. "gentle, introverted, prefers solitude, rational, cautious"
    """
    # TODO: implement
    raise NotImplementedError


def generateOpinionPrompt(opinionScore: float, persona: str, topic: str = TOPIC) -> str:
    """
    Generate initial opinion prompt from opinion score and persona description.

    Args:
        opinionScore: float [0, 1], 0=prefer cats, 1=prefer dogs, 0.5=both with preference
        persona: Persona tag string, e.g. "gentle, introverted, prefers solitude"
        topic: Topic, default "cats vs dogs"

    Returns:
        Updated initial opinion content (first-person description, at most 50 words), e.g. "I prefer cats because..."
    """
    # TODO: implement
    raise NotImplementedError

