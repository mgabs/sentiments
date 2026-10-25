import pytest
from sentiments.core.models import SentimentLabel
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier2.core import build_sentiment_prompt, parse_sentiment_output


def test_build_sentiment_prompt_contains_text_and_schema():
    prompt = build_sentiment_prompt("The battery died quickly.")
    assert "The battery died quickly." in prompt
    assert "POSITIVE" in prompt
    assert "NEGATIVE" in prompt
    assert "NEUTRAL" in prompt
    assert '"label"' in prompt
    assert '"confidence"' in prompt


def test_parse_sentiment_output_clean_json():
    raw = '{"label": "POSITIVE", "confidence": 0.94, "explanation": "Great service"}'
    label, confidence, metadata = parse_sentiment_output(raw)
    assert label == SentimentLabel.POSITIVE
    assert confidence == 0.94
    assert metadata == {"explanation": "Great service"}


def test_parse_sentiment_output_markdown_fences():
    raw = '```json\n{"label": "NEGATIVE", "confidence": 0.85}\n```'
    label, confidence, metadata = parse_sentiment_output(raw)
    assert label == SentimentLabel.NEGATIVE
    assert confidence == 0.85
    assert metadata == {}


def test_parse_sentiment_output_embedded_json_in_commentary():
    raw = 'Here is your analysis:\n{"label": "NEUTRAL", "confidence": 0.70}\nHope this helps!'
    label, confidence, metadata = parse_sentiment_output(raw)
    assert label == SentimentLabel.NEUTRAL
    assert confidence == 0.70


def test_parse_sentiment_output_missing_confidence_defaults_to_one():
    raw = '{"label": "POSITIVE"}'
    label, confidence, metadata = parse_sentiment_output(raw)
    assert label == SentimentLabel.POSITIVE
    assert confidence == 1.0


def test_parse_sentiment_output_clamps_confidence_bounds():
    raw_high = '{"label": "POSITIVE", "confidence": 1.5}'
    _, conf_high, _ = parse_sentiment_output(raw_high)
    assert conf_high == 1.0

    raw_low = '{"label": "NEGATIVE", "confidence": -0.5}'
    _, conf_low, _ = parse_sentiment_output(raw_low)
    assert conf_low == 0.0


def test_parse_sentiment_output_invalid_confidence_type():
    raw = '{"label": "POSITIVE", "confidence": "unknown"}'
    _, conf, _ = parse_sentiment_output(raw)
    assert conf == 1.0


def test_parse_sentiment_output_invalid_label_raises():
    raw = '{"label": "UNKNOWN_SENTIMENT", "confidence": 0.5}'
    with pytest.raises(ClassifierProviderError, match="Invalid sentiment label"):
        parse_sentiment_output(raw)


def test_parse_sentiment_output_no_json_object_raises():
    raw = "There is no json object here at all."
    with pytest.raises(ClassifierProviderError, match="no JSON object found"):
        parse_sentiment_output(raw)


def test_parse_sentiment_output_malformed_json_syntax_raises():
    raw = '{"label": "POSITIVE", bad_json: invalid}'
    with pytest.raises(ClassifierProviderError, match="Malformed JSON in response"):
        parse_sentiment_output(raw)
