import pytest
from pathlib import Path

from sentiments.core import SentimentLabel
from sentiments.dataset.models import DatasetRecord, DatasetSplit
from sentiments.dataset.normalizer import LabelNormalizer, FilenameMetadataParser, TextSanitizer
from sentiments.dataset.loader import DatasetLoader
from sentiments.dataset.splitter import StratifiedSplitter


def test_label_normalizer_positive():
    for raw in ["Positive", "positive", "POS", "pos", "Positive Review"]:
        assert LabelNormalizer.normalize(raw) == SentimentLabel.POSITIVE


def test_label_normalizer_negative():
    for raw in ["Negative", "negative", "NEG", "neg", "Negative Review"]:
        assert LabelNormalizer.normalize(raw) == SentimentLabel.NEGATIVE


def test_label_normalizer_neutral():
    for raw in ["Neutral", "neutral", "N/A", "neut"]:
        assert LabelNormalizer.normalize(raw) == SentimentLabel.NEUTRAL


def test_label_normalizer_unmapped():
    assert LabelNormalizer.normalize("Random Category") is None
    assert LabelNormalizer.normalize("") is None
    assert LabelNormalizer.normalize(None) is None


def test_filename_metadata_parser():
    meta = FilenameMetadataParser.parse("Green Lake (Jewelry) - Trustpilot Product Reviews - English.csv")
    assert meta["product"] == "Green Lake"
    assert meta["category"] == "Jewelry"
    assert meta["platform"] == "Trustpilot"
    assert meta["language"] == "English"

    meta2 = FilenameMetadataParser.parse("Back to The Future (Movie) - Training Set.csv")
    assert meta2["product"] == "Back to The Future"
    assert meta2["category"] == "Movie"
    assert meta2["platform"] == "Training Set"

    meta3 = FilenameMetadataParser.parse("SimpleReviews.csv")
    assert meta3["product"] == "SimpleReviews"


def test_text_sanitizer():
    assert TextSanitizer.clean("  <b>Hello World!</b> \n") == "Hello World!"
    assert TextSanitizer.clean("<p>Paragraph &amp; text</p>") == "Paragraph & text"
    assert TextSanitizer.clean("   \t\n  ") == ""
    assert TextSanitizer.clean(None) == ""


def test_dataset_record_validation():
    rec = DatasetRecord(
        id="rec_1",
        text="Great shoes!",
        label=SentimentLabel.POSITIVE,
        source="test.csv",
        metadata={"product": "Shoes"},
    )
    assert rec.id == "rec_1"
    assert rec.text == "Great shoes!"
    assert rec.label == SentimentLabel.POSITIVE
    assert rec.metadata["product"] == "Shoes"

    with pytest.raises(ValueError, match="text cannot be empty"):
        DatasetRecord(
            id="rec_2",
            text="   ",
            label=SentimentLabel.POSITIVE,
            source="test.csv",
            metadata={"product": "Shoes"},
        )


def test_dataset_loader_delim_detection(tmp_path: Path):
    # comma separated file
    csv1 = tmp_path / "Product (Cat) - Comma Reviews.csv"
    csv1.write_text("Content,Tag\nI love it!,Positive\nTerrible,Negative\n", encoding="utf-8")

    # semicolon separated file
    csv2 = tmp_path / "Service (Cat) - Semi Reviews.csv"
    csv2.write_text("Content;Consumer Sentiment Classifier\nGreat service!;Positive\nAwful;Negative\n", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 4
    products = {r.metadata["product"] for r in records}
    assert "Product" in products
    assert "Service" in products


def test_dataset_loader_filters_empty_and_unlabelled(tmp_path: Path):
    csv = tmp_path / "Test (Cat) - Reviews.csv"
    csv.write_text("Content,Tag\n   ,Positive\nValid review,UnknownTag\nReal review,Negative\n", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Real review"
    assert records[0].label == SentimentLabel.NEGATIVE


def test_stratified_splitter():
    records = []
    for i in range(60):
        records.append(DatasetRecord(f"p_{i}", f"Positive text {i}", SentimentLabel.POSITIVE, "src", {"product": "P"}))
    for i in range(30):
        records.append(DatasetRecord(f"n_{i}", f"Negative text {i}", SentimentLabel.NEGATIVE, "src", {"product": "P"}))
    for i in range(10):
        records.append(DatasetRecord(f"u_{i}", f"Neutral text {i}", SentimentLabel.NEUTRAL, "src", {"product": "P"}))

    splitter = StratifiedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)
    split: DatasetSplit = splitter.split(records)

    assert len(split.train) == 70
    assert len(split.val) == 15
    assert len(split.test) == 15

    # check proportions
    train_pos = sum(1 for r in split.train if r.label == SentimentLabel.POSITIVE)
    assert train_pos == 42  # 60 * 0.7 = 42

    # verify determinism
    split2 = splitter.split(records)
    assert [r.id for r in split.train] == [r.id for r in split2.train]


def test_stratified_splitter_invalid_ratios():
    with pytest.raises(ValueError, match="split ratios must sum to 1.0"):
        StratifiedSplitter(train_ratio=0.8, val_ratio=0.15, test_ratio=0.15)


def test_dataset_record_empty_id():
    with pytest.raises(ValueError, match="id cannot be empty"):
        DatasetRecord(
            id="",
            text="Valid text",
            label=SentimentLabel.POSITIVE,
            source="src",
            metadata={"product": "P"},
        )


def test_dataset_loader_nonexistent_dir(tmp_path: Path):
    loader = DatasetLoader(root_dir=tmp_path / "does_not_exist")
    assert loader.load_all() == []


def test_dataset_loader_handles_short_rows(tmp_path: Path):
    csv = tmp_path / "Short (Cat) - Reviews.csv"
    csv.write_text("Content,Tag\nOnlyContent\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 0


def test_stratified_splitter_negative_ratio():
    with pytest.raises(ValueError, match="split ratios must be non-negative"):
        StratifiedSplitter(train_ratio=1.2, val_ratio=-0.1, test_ratio=-0.1)


def test_stratified_splitter_rounding_boundary():
    # Test rounding boundary condition with 2 items and 0.6 / 0.5 split
    records = [
        DatasetRecord("1", "Text 1", SentimentLabel.POSITIVE, "s", {"product": "P"}),
        DatasetRecord("2", "Text 2", SentimentLabel.POSITIVE, "s", {"product": "P"}),
    ]
    splitter = StratifiedSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    split = splitter.split(records)
    assert len(split.train) + len(split.val) + len(split.test) == 2


def test_filename_metadata_parser_training_variants():
    m = FilenameMetadataParser.parse("TestProduct - Training Set.csv")
    assert m["product"] == "TestProduct"
    assert m["platform"] == "Training Set"

    m2 = FilenameMetadataParser.parse("SoloProduct - Amazon Reviews.csv")
    assert m2["product"] == "SoloProduct"
    assert m2["platform"] == "Amazon"
