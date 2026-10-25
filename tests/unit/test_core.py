import pytest
from sentiments.core import SentimentLabel, Tier, SentimentResult, SentimentClassifier


def test_sentiment_label_values():
    assert SentimentLabel.POSITIVE.value == "POSITIVE"
    assert SentimentLabel.NEGATIVE.value == "NEGATIVE"
    assert SentimentLabel.NEUTRAL.value == "NEUTRAL"
    with pytest.raises(ValueError):
        SentimentLabel("INVALID")


def test_tier_enum_values():
    assert Tier.TIER_0.value == "TIER_0"
    assert Tier.TIER_1.value == "TIER_1"
    assert Tier.TIER_2.value == "TIER_2"
    assert Tier.TIER_3.value == "TIER_3"
    assert Tier.TIER_4.value == "TIER_4"
    with pytest.raises(ValueError):
        Tier("TIER_99")


def test_sentiment_result_valid_instantiation():
    res = SentimentResult(
        label=SentimentLabel.POSITIVE,
        confidence=0.9,
        latency_ms=12.5,
        tier=Tier.TIER_0,
        model_name="lexicon_baseline",
    )
    assert res.label == SentimentLabel.POSITIVE
    assert res.confidence == 0.9
    assert res.latency_ms == 12.5
    assert res.tier == Tier.TIER_0
    assert res.model_name == "lexicon_baseline"
    assert res.metadata == {}


def test_sentiment_result_with_metadata():
    res = SentimentResult(
        label=SentimentLabel.NEUTRAL,
        confidence=0.5,
        latency_ms=0.0,
        tier=Tier.TIER_1,
        model_name="gpt-4o",
        metadata={"tokens": 42},
    )
    assert res.metadata == {"tokens": 42}


@pytest.mark.parametrize("invalid_conf", [-0.0001, -1.0, 1.0001, 2.5])
def test_sentiment_result_invalid_confidence(invalid_conf):
    with pytest.raises(ValueError, match=r"^confidence must be between 0\.0 and 1\.0"):
        SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=invalid_conf,
            latency_ms=1.0,
            tier=Tier.TIER_0,
            model_name="m",
        )


@pytest.mark.parametrize("invalid_latency", [-0.0001, -10.0])
def test_sentiment_result_invalid_latency(invalid_latency):
    with pytest.raises(ValueError, match=r"^latency_ms must be non-negative"):
        SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=0.5,
            latency_ms=invalid_latency,
            tier=Tier.TIER_0,
            model_name="m",
        )


def test_sentiment_result_empty_model_name():
    with pytest.raises(ValueError, match=r"^model_name cannot be empty$"):
        SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=0.5,
            latency_ms=1.0,
            tier=Tier.TIER_0,
            model_name="",
        )


def test_sentiment_result_immutability():
    res = SentimentResult(
        label=SentimentLabel.POSITIVE,
        confidence=0.5,
        latency_ms=1.0,
        tier=Tier.TIER_0,
        model_name="m",
    )
    with pytest.raises(Exception):
        res.confidence = 0.8


def test_sentiment_classifier_protocol_conformance():
    class GoodClassifier:
        @property
        def tier(self) -> Tier:
            return Tier.TIER_0

        @property
        def model_name(self) -> str:
            return "good"

        def classify(self, text: str) -> SentimentResult:
            return SentimentResult(
                label=SentimentLabel.POSITIVE,
                confidence=1.0,
                latency_ms=0.1,
                tier=self.tier,
                model_name=self.model_name,
            )

        def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
            return [self.classify(t) for t in texts]

    good = GoodClassifier()
    assert isinstance(good, SentimentClassifier)


def test_sentiment_classifier_protocol_non_conformance():
    class IncompleteClassifier:
        def classify(self, text: str) -> str:
            return "pos"

    bad = IncompleteClassifier()
    assert not isinstance(bad, SentimentClassifier)
