import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core import SentimentLabel, Tier, SentimentResult
from sentiments.dataset.models import DatasetRecord
from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier

scenarios("../features/tier0_baseline.feature")


@given(
    parsers.parse("a training dataset with {pos:d} positive and {neg:d} negative records"),
    target_fixture="training_records",
)
def create_training_records(pos, neg):
    records = []
    for i in range(pos):
        records.append(DatasetRecord(f"pos_{i}", f"Positive {i}", SentimentLabel.POSITIVE, "src", {"product": "P"}))
    for i in range(neg):
        records.append(DatasetRecord(f"neg_{i}", f"Negative {i}", SentimentLabel.NEGATIVE, "src", {"product": "P"}))
    return records


@when("a MajorityClassClassifier is fitted on the training dataset", target_fixture="fitted_majority_clf")
def fit_majority_classifier(training_records):
    clf = MajorityClassClassifier()
    clf.fit(training_records)
    return clf


@then(parsers.parse('its majority tier is "{tier}"'))
def verify_majority_tier(fitted_majority_clf, tier):
    assert fitted_majority_clf.tier == Tier(tier)


@then(parsers.parse('its majority model name is "{model_name}"'))
def verify_majority_model_name(fitted_majority_clf, model_name):
    assert fitted_majority_clf.model_name == model_name


@then(parsers.parse('classifying any input text "{text}" yields label "{label}"'))
def verify_classification_label(fitted_majority_clf, text, label):
    res = fitted_majority_clf.classify(text)
    assert res.label == SentimentLabel(label)


@then(parsers.parse("the prediction confidence is {confidence:f}"))
def verify_prediction_confidence(fitted_majority_clf, confidence):
    res = fitted_majority_clf.classify("sample")
    assert res.confidence == pytest.approx(confidence)


@then("the majority inference latency is under 1.0 ms")
def verify_majority_latency(fitted_majority_clf):
    res = fitted_majority_clf.classify("sample")
    assert res.latency_ms < 1.0


@then("classifying text produces a deterministic label")
def verify_tie_break_label(fitted_majority_clf):
    res = fitted_majority_clf.classify("sample")
    assert res.label in [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE]


@given("an initialized LexiconRuleClassifier", target_fixture="lexicon_clf")
def init_lexicon_classifier():
    return LexiconRuleClassifier()


@when(parsers.re(r'the classifier evaluates "(?P<text>.*)"'), target_fixture="eval_result")
def eval_lexicon(lexicon_clf, text):
    return lexicon_clf.classify(text)


@then(parsers.parse('the resulting sentiment label is "{expected_label}"'))
def verify_lexicon_label(eval_result, expected_label):
    assert eval_result.label == SentimentLabel(expected_label)


@then(parsers.parse('the lexicon tier is "{tier}"'))
def verify_lexicon_tier(eval_result, tier):
    assert eval_result.tier == Tier(tier)


@then(parsers.parse('the lexicon model name is "{model_name}"'))
def verify_lexicon_model(eval_result, model_name):
    assert eval_result.model_name == model_name


@then("the lexicon inference latency is under 1.0 ms")
def verify_lexicon_latency(eval_result):
    assert eval_result.latency_ms < 1.0


@when(
    parsers.parse("classifying a batch of {count:d} customer feedback reviews"),
    target_fixture="batch_results",
)
def classify_batch(lexicon_clf, count):
    samples = ["Great!", "Awful!", "Normal."]
    return lexicon_clf.classify_batch(samples[:count])


@then(parsers.parse("it returns a list of {count:d} SentimentResults"))
def verify_batch_count(batch_results, count):
    assert len(batch_results) == count
    assert all(isinstance(r, SentimentResult) for r in batch_results)


@then(parsers.parse('each result has tier "{tier}"'))
def verify_batch_tier(batch_results, tier):
    assert all(r.tier == Tier(tier) for r in batch_results)
