Feature: Dataset Ingestion and Preprocessing Pipeline

  Scenario: Discovering labeled dataset files in nlp-datasets
    Given the root dataset path is "nlp-datasets"
    When the dataset loader scans for labeled files
    Then it discovers CSV files containing valid sentiment labels
    And files without recognized label columns are ignored

  Scenario Outline: Normalizing heterogeneous sentiment labels to canonical taxonomy
    Given a raw label string "<raw_label>"
    When the label is normalized
    Then the resulting sentiment label is "<canonical_label>"

    Examples:
      | raw_label | canonical_label |
      | Positive  | POSITIVE        |
      | positive  | POSITIVE        |
      | Pos       | POSITIVE        |
      | Negative  | NEGATIVE        |
      | negative  | NEGATIVE        |
      | Neg       | NEGATIVE        |
      | Neutral   | NEUTRAL         |
      | neutral   | NEUTRAL         |

  Scenario Outline: Extracting contextual metadata including product from dataset paths
    Given a dataset file named "<filename>"
    When the file metadata is parsed
    Then the extracted product is "<product>"
    And the extracted category is "<category>"
    And the extracted platform is "<platform>"

    Examples:
      | filename                                                                 | product            | category        | platform    |
      | Green Lake (Jewelry) - Trustpilot Product Reviews - English.csv          | Green Lake         | Jewelry         | Trustpilot  |
      | Apple AirTag (Technology) - Amazon Product Reviews.csv                  | Apple AirTag       | Technology      | Amazon      |
      | Back to The Future (Movie) - Training Set.csv                            | Back to The Future | Movie           | Training Set|
      | Robin&Jesper (Digital Marketing) - Udemy User Reviews - English.csv     | Robin&Jesper       | Digital Marketing| Udemy       |

  Scenario: Ingested records attach contextual metadata
    Given a dataset file with product "Green Lake"
    When the dataset is ingested
    Then every record includes metadata with product "Green Lake"

  Scenario: Discarding unmapped or ambiguous labels
    Given a raw label string "Unrelated Topic Tag"
    When the label is normalized
    Then normalization returns None

  Scenario: Sanitizing and validating review text
    Given a raw text containing "   <b>Great product!</b> \n  "
    When the text is sanitized
    Then the cleaned text is "Great product!"

  Scenario: Filtering out empty or whitespace-only records
    Given a dataset containing a mix of valid and whitespace-only text rows
    When the dataset is ingested
    Then only records with non-empty cleaned text are retained

  Scenario: Deterministic stratified dataset splitting
    Given an ingested dataset with 60 positive, 30 negative, and 10 neutral records
    When the dataset is split with ratios 70% train, 15% val, 15% test and seed 42
    Then the train split contains exactly 70 records
    And the val split contains exactly 15 records
    And the test split contains exactly 15 records
    And the class distribution ratio is preserved across all splits
    And splitting again with seed 42 produces identical splits
