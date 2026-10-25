import pytest
from pathlib import Path
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core import SentimentLabel
from sentiments.dataset.models import DatasetRecord, DatasetSplit
from sentiments.dataset.normalizer import LabelNormalizer, FilenameMetadataParser, TextSanitizer
from sentiments.dataset.loader import DatasetLoader
from sentiments.dataset.splitter import StratifiedSplitter

scenarios("../features/dataset_pipeline.feature")


@given(parsers.parse('the root dataset path is "{path}"'), target_fixture="dataset_path")
def root_dataset_path(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        return p
    for parent in Path(__file__).resolve().parents:
        candidate = parent / path
        if candidate.exists() and any(candidate.iterdir()):
            return candidate
    return p


@when("the dataset loader scans for labeled files", target_fixture="discovered_records")
def scan_dataset(dataset_path):
    loader = DatasetLoader(root_dir=dataset_path)
    return loader.load_all()


@then("it discovers CSV files containing valid sentiment labels")
def verify_discovered_records(discovered_records):
    assert len(discovered_records) > 0
    assert all(isinstance(r, DatasetRecord) for r in discovered_records)


@then("files without recognized label columns are ignored")
def verify_unrecognized_ignored(discovered_records):
    labels = {r.label for r in discovered_records}
    assert labels.issubset({SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE, SentimentLabel.NEUTRAL})


@given(parsers.parse('a raw label string "{raw_label}"'), target_fixture="raw_label_val")
def given_raw_label(raw_label):
    return raw_label


@when("the label is normalized", target_fixture="normalized_label")
def normalize_label(raw_label_val):
    return LabelNormalizer.normalize(raw_label_val)


@then(parsers.parse('the resulting sentiment label is "{canonical_label}"'))
def check_canonical_label(normalized_label, canonical_label):
    assert normalized_label == SentimentLabel(canonical_label)


@then("normalization returns None")
def check_normalization_none(normalized_label):
    assert normalized_label is None


@given(parsers.parse('a dataset file named "{filename}"'), target_fixture="file_metadata")
def file_for_metadata(filename):
    return FilenameMetadataParser.parse(filename)


@when("the file metadata is parsed")
def parse_file_metadata(file_metadata):
    pass


@then(parsers.parse('the extracted product is "{product}"'))
def check_meta_product(file_metadata, product):
    assert file_metadata["product"] == product


@then(parsers.parse('the extracted category is "{category}"'))
def check_meta_category(file_metadata, category):
    assert file_metadata["category"] == category


@then(parsers.parse('the extracted platform is "{platform}"'))
def check_meta_platform(file_metadata, platform):
    assert file_metadata["platform"] == platform


@given(parsers.parse('a dataset file with product "{product}"'), target_fixture="sample_records")
def dataset_file_with_product(product, tmp_path):
    csv = tmp_path / f"{product} (Category) - Platform Product Reviews.csv"
    csv.write_text("Content,Tag\nGreat product!,Positive\nTerrible!,Negative\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    return loader.load_all()


@when("the dataset is ingested")
def ingest_dataset():
    pass


@then(parsers.parse('every record includes metadata with product "{product}"'))
def verify_records_product(sample_records, product):
    assert len(sample_records) == 2
    assert all(r.metadata.get("product") == product for r in sample_records)


@given(parsers.parse('a raw text containing "{raw_text}"'), target_fixture="raw_text_val")
def given_raw_text(raw_text):
    return raw_text


@when("the text is sanitized", target_fixture="cleaned_text")
def sanitize_text(raw_text_val):
    return TextSanitizer.clean(raw_text_val)


@then(parsers.parse('the cleaned text is "{expected}"'))
def check_cleaned_text(cleaned_text, expected):
    assert cleaned_text == expected


@given("a dataset containing a mix of valid and whitespace-only text rows", target_fixture="mixed_records")
def given_mixed_dataset(tmp_path):
    csv = tmp_path / "Mixed (Cat) - Reviews.csv"
    csv.write_text("Content,Tag\nValid 1,Positive\n   \n\t,Negative\nValid 2,Negative\n", encoding="utf-8")
    loader = DatasetLoader(root_dir=tmp_path)
    return loader.load_all()


@then("only records with non-empty cleaned text are retained")
def verify_non_empty(mixed_records):
    assert len(mixed_records) == 2
    assert all(r.text in ["Valid 1", "Valid 2"] for r in mixed_records)


@given(
    parsers.parse("an ingested dataset with {pos:d} positive, {neg:d} negative, and {neut:d} neutral records"),
    target_fixture="synthetic_dataset",
)
def synthetic_dataset(pos, neg, neut):
    records = []
    for i in range(pos):
        records.append(DatasetRecord(f"pos_{i}", f"Positive text {i}", SentimentLabel.POSITIVE, "src", {"product": "P"}))
    for i in range(neg):
        records.append(DatasetRecord(f"neg_{i}", f"Negative text {i}", SentimentLabel.NEGATIVE, "src", {"product": "P"}))
    for i in range(neut):
        records.append(DatasetRecord(f"neut_{i}", f"Neutral text {i}", SentimentLabel.NEUTRAL, "src", {"product": "P"}))
    return records


@when(
    parsers.parse("the dataset is split with ratios {train:d}% train, {val:d}% val, {test:d}% test and seed {seed:d}"),
    target_fixture="split_result",
)
def split_synthetic(synthetic_dataset, train, val, test, seed):
    splitter = StratifiedSplitter(train_ratio=train / 100.0, val_ratio=val / 100.0, test_ratio=test / 100.0, seed=seed)
    return splitter.split(synthetic_dataset)


@then(parsers.parse("the train split contains exactly {count:d} records"))
def check_train_count(split_result, count):
    assert len(split_result.train) == count


@then(parsers.parse("the val split contains exactly {count:d} records"))
def check_val_count(split_result, count):
    assert len(split_result.val) == count


@then(parsers.parse("the test split contains exactly {count:d} records"))
def check_test_count(split_result, count):
    assert len(split_result.test) == count


@then("the class distribution ratio is preserved across all splits")
def check_distribution_preserved(split_result):
    train_pos = sum(1 for r in split_result.train if r.label == SentimentLabel.POSITIVE)
    val_pos = sum(1 for r in split_result.val if r.label == SentimentLabel.POSITIVE)
    test_pos = sum(1 for r in split_result.test if r.label == SentimentLabel.POSITIVE)
    assert train_pos == 42
    assert val_pos == 9
    assert test_pos == 9


@then(parsers.parse("splitting again with seed {seed:d} produces identical splits"))
def check_deterministic_split(synthetic_dataset, seed, split_result):
    splitter = StratifiedSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=seed)
    split2 = splitter.split(synthetic_dataset)
    assert [r.id for r in split_result.train] == [r.id for r in split2.train]
    assert [r.id for r in split_result.val] == [r.id for r in split2.val]
    assert [r.id for r in split_result.test] == [r.id for r in split2.test]
