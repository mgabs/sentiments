import pytest
from sentiments.core import SentimentLabel, Tier, SentimentClassifier
from sentiments.dataset.models import DatasetRecord
from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier


def test_majority_classifier_protocol_conformance():
    clf = MajorityClassClassifier()
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_0
    assert clf.model_name == "majority_class_baseline"
    assert clf.majority_label is None
    assert clf.confidence is None


def test_majority_classifier_unfitted_raises():
    clf = MajorityClassClassifier()
    with pytest.raises(RuntimeError, match=r"^classifier must be fitted before inference$"):
        clf.classify("test")


def test_majority_classifier_fit_empty_raises():
    clf = MajorityClassClassifier()
    with pytest.raises(ValueError, match=r"^cannot fit on empty dataset$"):
        clf.fit([])


def test_majority_classifier_predict_and_latency():
    records = [
        DatasetRecord(f"pos_{i}", f"Good text {i}", SentimentLabel.POSITIVE, "src", {"product": "P"})
        for i in range(8)
    ] + [
        DatasetRecord(f"neg_{i}", f"Bad text {i}", SentimentLabel.NEGATIVE, "src", {"product": "P"})
        for i in range(2)
    ]

    clf = MajorityClassClassifier()
    clf.fit(records)
    assert clf.majority_label == SentimentLabel.POSITIVE
    assert clf.confidence == pytest.approx(0.80)

    res = clf.classify("Any arbitrary text review")
    assert res.label == SentimentLabel.POSITIVE
    assert res.confidence == pytest.approx(0.80)
    assert res.tier == Tier.TIER_0
    assert res.model_name == "majority_class_baseline"
    assert res.latency_ms < 1.0


def test_majority_classifier_tie_breaking():
    records = [
        DatasetRecord("1", "Good text", SentimentLabel.POSITIVE, "src", {"product": "P"}),
        DatasetRecord("2", "Bad text", SentimentLabel.NEGATIVE, "src", {"product": "P"}),
    ]
    clf = MajorityClassClassifier()
    clf.fit(records)
    # Alphabetical order: NEGATIVE comes before POSITIVE
    assert clf.majority_label == SentimentLabel.NEGATIVE
    assert clf.confidence == pytest.approx(0.50)


def test_majority_classifier_batch():
    records = [DatasetRecord("1", "Good", SentimentLabel.POSITIVE, "src", {"product": "P"})]
    clf = MajorityClassClassifier()
    clf.fit(records)
    batch = clf.classify_batch(["text1", "text2"])
    assert len(batch) == 2
    assert all(r.label == SentimentLabel.POSITIVE for r in batch)


def test_lexicon_classifier_protocol_conformance():
    clf = LexiconRuleClassifier()
    assert isinstance(clf, SentimentClassifier)
    assert clf.tier == Tier.TIER_0
    assert clf.model_name == "lexicon_rule_baseline"


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("This is fantastic and amazing and wonderful", SentimentLabel.POSITIVE),
        ("This is horrible and terrible and awful", SentimentLabel.NEGATIVE),
        ("The product is standard and average", SentimentLabel.NEUTRAL),
        ("", SentimentLabel.NEUTRAL),
        ("   \t\n", SentimentLabel.NEUTRAL),
        ("good bad", SentimentLabel.NEUTRAL),
    ],
)
def test_lexicon_classifier_polarity(text, expected_label):
    clf = LexiconRuleClassifier()
    res = clf.classify(text)
    assert res.label == expected_label
    assert res.tier == Tier.TIER_0
    assert res.model_name == "lexicon_rule_baseline"
    assert 0.0 <= res.confidence <= 1.0
    assert res.latency_ms < 1.0


def test_lexicon_classifier_confidence_scaling():
    clf = LexiconRuleClassifier()
    # Strong positive
    strong_pos = clf.classify("excellent amazing wonderful fantastic")
    assert strong_pos.confidence > 0.8
    assert strong_pos.label == SentimentLabel.POSITIVE

    # Strong negative
    strong_neg = clf.classify("terrible awful horrible disgusting")
    assert strong_neg.confidence > 0.8
    assert strong_neg.label == SentimentLabel.NEGATIVE

    # Neutral
    neutral = clf.classify("average normal")
    assert neutral.confidence == pytest.approx(0.50)
    assert neutral.label == SentimentLabel.NEUTRAL


def test_lexicon_classifier_batch():
    clf = LexiconRuleClassifier()
    batch = clf.classify_batch(["Great!", "Awful!", "Normal."])
    assert len(batch) == 3
    assert batch[0].label == SentimentLabel.POSITIVE
    assert batch[1].label == SentimentLabel.NEGATIVE
    assert batch[2].label == SentimentLabel.NEUTRAL


def test_majority_classifier_latency_millisecond_scaling(monkeypatch):
    records = [DatasetRecord("1", "Good text", SentimentLabel.POSITIVE, "src", {"product": "P"})]
    clf = MajorityClassClassifier()
    clf.fit(records)

    times = [0.0, 0.005]  # 5ms
    monkeypatch.setattr("time.perf_counter", lambda: times.pop(0))
    res = clf.classify("sample")
    assert res.latency_ms == pytest.approx(5.0)


def test_lexicon_classifier_latency_millisecond_scaling(monkeypatch):
    clf = LexiconRuleClassifier()
    times = [0.0, 0.005]  # 5ms
    monkeypatch.setattr("time.perf_counter", lambda: times.pop(0))
    res = clf.classify("sample")
    assert res.latency_ms == pytest.approx(5.0)


def test_majority_classifier_invalid_input_type():
    records = [DatasetRecord("1", "Good text", SentimentLabel.POSITIVE, "src", {"product": "P"})]
    clf = MajorityClassClassifier()
    clf.fit(records)
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(None)


def test_lexicon_classifier_invalid_input_type():
    clf = LexiconRuleClassifier()
    with pytest.raises(ValueError, match=r"^text must be a string$"):
        clf.classify(None)


def test_lexicon_classifier_mixed_polarity_exact_confidence():
    clf = LexiconRuleClassifier()
    # 2 positive, 1 negative -> diff=1, total=3, conf = 0.5 + 0.5 * (1/3) = 0.666667
    mixed_pos = clf.classify("good great bad")
    assert mixed_pos.label == SentimentLabel.POSITIVE
    assert mixed_pos.confidence == pytest.approx(0.5 + 0.5 * (1.0 / 3.0))

    # 2 negative, 1 positive -> diff=1, total=3, conf = 0.5 + 0.5 * (1/3) = 0.666667
    mixed_neg = clf.classify("bad terrible good")
    assert mixed_neg.label == SentimentLabel.NEGATIVE
    assert mixed_neg.confidence == pytest.approx(0.5 + 0.5 * (1.0 / 3.0))
