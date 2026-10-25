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

    meta4 = FilenameMetadataParser.parse("Simple.Product.Name.csv")
    assert meta4["product"] == "Simple.Product.Name"


def test_text_sanitizer():
    assert TextSanitizer.clean("  <b>Hello World!</b> \n") == "Hello World!"
    assert TextSanitizer.clean("<p>Paragraph &amp; text</p>") == "Paragraph & text"
    assert TextSanitizer.clean("   \t\n  ") == ""
    assert TextSanitizer.clean(None) == ""
    assert TextSanitizer.clean("word\\rword\\tword\\nword") == "word word word word"


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
    assert rec.source == "test.csv"
    assert rec.metadata["product"] == "Shoes"

    with pytest.raises(ValueError, match=r"^text cannot be empty$"):
        DatasetRecord(
            id="rec_2",
            text="   ",
            label=SentimentLabel.POSITIVE,
            source="test.csv",
            metadata={"product": "Shoes"},
        )

    with pytest.raises(ValueError, match=r"^id cannot be empty$"):
        DatasetRecord(
            id="",
            text="Valid text",
            label=SentimentLabel.POSITIVE,
            source="src",
            metadata={"product": "P"},
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
    assert all(r.source in ["Product (Cat) - Comma Reviews.csv", "Service (Cat) - Semi Reviews.csv"] for r in records)


def test_dataset_loader_bom_and_equal_delimiter(tmp_path: Path):
    csv_bom = tmp_path / "Bom (Cat) - Reviews.csv"
    # BOM in header and equal commas/semicolons
    csv_bom.write_text("\ufeffContent,Tag\nNice,Positive\n", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Nice"
    assert records[0].label == SentimentLabel.POSITIVE


def test_dataset_loader_delim_counts_commas(tmp_path: Path):
    # Has more commas than semicolons: should pick comma
    csv_multi = tmp_path / "Multi (Cat) - Reviews.csv"
    csv_multi.write_text("Content,Tag,Notes\nHello;World,Positive,Extra\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Hello;World"


def test_dataset_loader_handles_invalid_utf8(tmp_path: Path):
    csv_bad = tmp_path / "Bad (Cat) - Reviews.csv"
    csv_bad.write_bytes(b"Content,Tag\nValid \xff byte,Positive\n")
    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].label == SentimentLabel.POSITIVE


def test_dataset_loader_skips_git_and_worktrees(tmp_path: Path):
    git_dir = tmp_path / "0_git" / ".git" / "sub"
    git_dir.mkdir(parents=True)
    (git_dir / "Git (Cat) - Reviews.csv").write_text("Content,Tag\nGit text,Positive\n", encoding="utf-8")

    wt_dir = tmp_path / "0_wt" / ".worktrees" / "sub"
    wt_dir.mkdir(parents=True)
    (wt_dir / "Wt (Cat) - Reviews.csv").write_text("Content,Tag\nWt text,Positive\n", encoding="utf-8")

    real_dir = tmp_path / "1_real"
    real_dir.mkdir(parents=True)
    (real_dir / "Real (Cat) - Reviews.csv").write_text("Content,Tag\nReal text,Positive\n", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Real text"


def test_dataset_loader_filters_empty_and_unlabelled(tmp_path: Path):
    csv = tmp_path / "Test (Cat) - Reviews.csv"
    csv.write_text("Content,Tag\n   ,Positive\nValid review,UnknownTag\nReal review,Negative\n", encoding="utf-8")

    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Real review"
    assert records[0].label == SentimentLabel.NEGATIVE
    assert records[0].source == "Test (Cat) - Reviews.csv"


def test_dataset_loader_nonexistent_dir(tmp_path: Path):
    loader = DatasetLoader(root_dir=tmp_path / "does_not_exist")
    assert loader.load_all() == []


def test_dataset_loader_handles_short_rows(tmp_path: Path):
    csv = tmp_path / "Short (Cat) - Reviews.csv"
    csv.write_text("Content,Tag\nOnlyContent\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    assert loader.load_all() == []


def test_stratified_splitter_defaults():
    splitter = StratifiedSplitter()
    assert splitter.train_ratio == 0.70
    assert splitter.val_ratio == 0.15
    assert splitter.test_ratio == 0.15
    assert splitter.seed == 42


def test_stratified_splitter_execution():
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
    assert train_pos == 42

    # verify determinism
    split2 = splitter.split(records)
    assert [r.id for r in split.train] == [r.id for r in split2.train]


def test_stratified_splitter_invalid_sum():
    with pytest.raises(ValueError, match=r"^split ratios must sum to 1\.0"):
        StratifiedSplitter(0.8, 0.15, 0.15)


@pytest.mark.parametrize(
    "ratios",
    [
        (-0.1, 0.6, 0.5),
        (0.6, -0.1, 0.5),
        (0.6, 0.5, -0.1),
    ],
)
def test_stratified_splitter_negative_ratios(ratios):
    with pytest.raises(ValueError, match=r"^split ratios must be non-negative$"):
        StratifiedSplitter(*ratios)


def test_filename_metadata_parser_training_variants():
    m = FilenameMetadataParser.parse("TestProduct - Training Set.csv")
    assert m["product"] == "TestProduct"
    assert m["platform"] == "Training Set"

    m2 = FilenameMetadataParser.parse("SoloProduct - Amazon Reviews.csv")
    assert m2["product"] == "SoloProduct"
    assert m2["platform"] == "Amazon"


def test_stratified_splitter_zero_ratios():
    s1 = StratifiedSplitter(train_ratio=1.0, val_ratio=0.0, test_ratio=0.0)
    assert s1.train_ratio == 1.0
    assert s1.val_ratio == 0.0
    assert s1.test_ratio == 0.0

    s2 = StratifiedSplitter(train_ratio=0.0, val_ratio=1.0, test_ratio=0.0)
    assert s2.train_ratio == 0.0

    s3 = StratifiedSplitter(train_ratio=0.5, val_ratio=0.5, test_ratio=0.0)
    assert s3.test_ratio == 0.0


def test_stratified_splitter_precision_rounding():
    with pytest.raises(ValueError, match=r"^split ratios must sum to 1\.0"):
        StratifiedSplitter(0.70001, 0.15, 0.15)


def test_dataset_loader_delimiter_tie_breaks_to_comma(tmp_path: Path):
    csv_file = tmp_path / "Tie (Cat) - Reviews.csv"
    # Exactly 1 semicolon and 1 comma in header
    csv_file.write_text("Content,Tag;\nGreat,Positive;\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    records = loader.load_all()
    assert len(records) == 1
    assert records[0].text == "Great"
    assert records[0].label == SentimentLabel.POSITIVE
