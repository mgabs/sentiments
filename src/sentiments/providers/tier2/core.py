import json
import re
from typing import Any
from sentiments.core.models import SentimentLabel
from sentiments.providers.base import ClassifierProviderError


def build_sentiment_prompt(text: str) -> str:
    """Construct a deterministic prompt enforcing strict JSON schema output."""
    return (
        "You are an expert sentiment classification engine. "
        "Analyze the customer review below and classify its sentiment as exactly one of: "
        "POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
        "Output ONLY a valid JSON object with exact keys:\n"
        "{\n"
        '  "label": "POSITIVE" | "NEGATIVE" | "NEUTRAL",\n'
        '  "confidence": <float between 0.0 and 1.0>,\n'
        '  "explanation": "<concise reason>"\n'
        "}\n\n"
        f"Review:\n\"\"\"{text}\"\"\"\n\n"
        "JSON:"
    )


def parse_sentiment_output(raw_output: str) -> tuple[SentimentLabel, float, dict[str, Any]]:
    """Extract and validate sentiment label, confidence, and metadata from raw model output."""
    clean = raw_output.strip()

    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean, flags=re.IGNORECASE)
        clean = re.sub(r"\s*```$", "", clean)
        clean = clean.strip()

    match = re.search(r"\{.*?\}", clean, re.DOTALL)
    if not match:
        raise ClassifierProviderError(f"Malformed output: no JSON object found in: {raw_output[:100]}")

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ClassifierProviderError(f"Malformed JSON in response: {exc}") from exc

    raw_label = str(data.get("label", "")).strip().upper()
    try:
        label = SentimentLabel(raw_label)
    except ValueError as exc:
        raise ClassifierProviderError(f"Invalid sentiment label returned: '{raw_label}'") from exc

    raw_conf = data.get("confidence", 1.0)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    metadata: dict[str, Any] = {}
    if "explanation" in data and isinstance(data["explanation"], str):
        metadata["explanation"] = data["explanation"]

    return label, confidence, metadata
