import pytest
from pytest_bdd import scenarios, given, then, parsers

from sentiments.core import SentimentLabel, Tier, SentimentResult, SentimentClassifier

scenarios("../features/core_contracts.feature")


@given("the sentiment taxonomy")
def sentiment_taxonomy():
    return SentimentLabel


@then(parsers.parse('"{label}" is a valid SentimentLabel'))
def valid_sentiment_label(label):
    assert SentimentLabel(label).value == label


@then(parsers.parse('"{label}" is rejected as an invalid SentimentLabel'))
def invalid_sentiment_label(label):
    with pytest.raises(ValueError):
        SentimentLabel(label)


@given(
    parsers.parse(
        'a sentiment classification with label "{label}", confidence {confidence:f}, latency {latency:f} ms, tier "{tier}", and model "{model}"'
    ),
    target_fixture="sentiment_result",
)
def create_valid_sentiment_result(label, confidence, latency, tier, model):
    return SentimentResult(
        label=SentimentLabel(label),
        confidence=confidence,
        latency_ms=latency,
        tier=Tier(tier),
        model_name=model,
    )


@then(parsers.parse('the result has label "{label}"'))
def check_result_label(sentiment_result, label):
    assert sentiment_result.label == SentimentLabel(label)


@then(parsers.parse("the result has confidence {confidence:f}"))
def check_result_confidence(sentiment_result, confidence):
    assert sentiment_result.confidence == pytest.approx(confidence)


@then(parsers.parse("the result has latency {latency:f} ms"))
def check_result_latency(sentiment_result, latency):
    assert sentiment_result.latency_ms == pytest.approx(latency)


@then(parsers.parse('the result has tier "{tier}"'))
def check_result_tier(sentiment_result, tier):
    assert sentiment_result.tier == Tier(tier)


@then(parsers.parse('the result has model name "{model}"'))
def check_result_model(sentiment_result, model):
    assert sentiment_result.model_name == model


@then("the result is immutable")
def check_result_immutable(sentiment_result):
    with pytest.raises(Exception):
        sentiment_result.confidence = 0.5


@given(
    parsers.parse("a sentiment classification attempt with confidence {confidence:f}"),
    target_fixture="attempt_conf",
)
def attempt_confidence(confidence):
    return confidence


@then(parsers.parse("the result creation should {outcome}"))
def check_confidence_outcome(attempt_conf, outcome):
    if outcome == "succeed":
        res = SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=attempt_conf,
            latency_ms=1.0,
            tier=Tier.TIER_0,
            model_name="test_model",
        )
        assert res.confidence == attempt_conf
    else:
        with pytest.raises(ValueError):
            SentimentResult(
                label=SentimentLabel.POSITIVE,
                confidence=attempt_conf,
                latency_ms=1.0,
                tier=Tier.TIER_0,
                model_name="test_model",
            )


@given(
    parsers.parse("a sentiment classification attempt with latency {latency:f} ms"),
    target_fixture="attempt_lat",
)
def attempt_latency(latency):
    return latency


@then("the latency validation should fail with a value error")
def check_latency_outcome(attempt_lat):
    with pytest.raises(ValueError):
        SentimentResult(
            label=SentimentLabel.POSITIVE,
            confidence=0.5,
            latency_ms=attempt_lat,
            tier=Tier.TIER_0,
            model_name="test_model",
        )


@given("a compliant mock sentiment classifier", target_fixture="mock_classifier_instance")
def mock_classifier():
    class MockClassifier:
        @property
        def tier(self) -> Tier:
            return Tier.TIER_0

        @property
        def model_name(self) -> str:
            return "mock"

        def classify(self, text: str) -> SentimentResult:
            return SentimentResult(
                label=SentimentLabel.POSITIVE,
                confidence=0.9,
                latency_ms=0.1,
                tier=Tier.TIER_0,
                model_name=self.model_name,
            )

        def classify_batch(self, texts: list[str]) -> list[SentimentResult]:
            return [self.classify(t) for t in texts]

    return MockClassifier()


@then("it conforms to the SentimentClassifier protocol")
def check_protocol_conformance(mock_classifier_instance):
    assert isinstance(mock_classifier_instance, SentimentClassifier)


@then("it exposes tier and model_name attributes")
def check_attributes(mock_classifier_instance):
    assert mock_classifier_instance.tier == Tier.TIER_0
    assert mock_classifier_instance.model_name == "mock"


@then("it supports single text classification returning a SentimentResult")
def check_single_classify(mock_classifier_instance):
    res = mock_classifier_instance.classify("good")
    assert isinstance(res, SentimentResult)
    assert res.label == SentimentLabel.POSITIVE


@then("it supports batch classification returning a list of SentimentResults")
def check_batch_classify(mock_classifier_instance):
    results = mock_classifier_instance.classify_batch(["good", "great"])
    assert len(results) == 2
    assert all(isinstance(r, SentimentResult) for r in results)
