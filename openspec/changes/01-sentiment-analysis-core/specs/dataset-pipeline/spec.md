# Specification: Dataset Ingestion & Preprocessing Pipeline

## 1. Overview
Defines specifications for ingesting, validating, normalizing, and splitting customer feedback datasets from `nlp-datasets/` into standardized evaluation corpora.

## 2. Requirements & Capabilities

### Capability: Dataset Ingestion
- **REQ-DATA-1**: The system MUST discover and ingest labeled CSV files from `nlp-datasets/` including:
  - `training-sets/*.csv` (columns: `Content`, `Tag`)
  - `trustpilot-reviews/*.csv`, `udemy-reviews/*.csv`, `etsy-reviews/*.csv` with `Consumer Sentiment Classifier` or `Automatic Classifier`.
- **REQ-DATA-2**: The system MUST normalize heterogeneous label strings to canonical three-class sentiment labels:
  - `POSITIVE`: "Positive", "positive", "Pos", or tags indicating positive sentiment.
  - `NEGATIVE`: "Negative", "negative", "Neg", or tags indicating negative sentiment.
  - `NEUTRAL`: "Neutral", "neutral", "N/A" (where applicable).
  - Unmapped or conflicting tags MUST be discarded or logged with warnings.

### Capability: Text Cleaning & Validation
- **REQ-DATA-3**: Records with empty, whitespace-only, or `null` text content MUST be filtered out.
- **REQ-DATA-4**: Text inputs MUST be stripped of extraneous HTML tags, trailing whitespace, and control characters.
- **REQ-DATA-5**: Each sanitized record MUST output a standardized structure:
  - `id`: Unique string identifier.
  - `text`: Non-empty cleaned review string.
  - `label`: Canonical sentiment enum (`POSITIVE`, `NEGATIVE`, `NEUTRAL`).
  - `source`: Dataset file origin identifier.

### Capability: Dataset Splitting
- **REQ-DATA-6**: The system MUST support deterministic stratified train/val/test splitting given a random seed and split ratios (e.g., 70/15/15 or 80/10/10).
- **REQ-DATA-7**: Class distribution proportions MUST be preserved across all splits.

## 3. Assumptions
- **ASM-DATA-1**: Corpus fits completely in local memory (<100MB RAM); no distributed streaming or Spark required.
- **ASM-DATA-2**: Primary target language is English. UTF-8 multi-lingual text is preserved without automated translation.
- **ASM-DATA-3**: Records with unmapped or ambiguous custom tags are discarded from ground-truth training/eval sets rather than guessed.
- **ASM-DATA-4**: CSV delimiters may vary (comma `,` or semicolon `;`) and must be detected dynamically.

## 4. Edge Cases & Error Handling
- **Empty Dataset**: Raising `DatasetEmptyError` if specified source folder contains 0 valid labeled records.
- **Malformed Rows**: Gracefully skipping malformed CSV rows while tracking skipped row count.
- **Single-Class Dataset**: Raising a descriptive validation error if dataset contains only one label class.
