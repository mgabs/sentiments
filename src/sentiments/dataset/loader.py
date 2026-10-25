import csv
import os
from pathlib import Path
from typing import Optional

from sentiments.dataset.models import DatasetRecord
from sentiments.dataset.normalizer import LabelNormalizer, FilenameMetadataParser, TextSanitizer


class DatasetLoader:
    _TEXT_COLUMNS = ["content", "comments", "reviews", "review"]
    _LABEL_COLUMNS = ["tag", "consumer sentiment classifier", "automatic classifier", "sentiment", "label"]

    def __init__(self, root_dir: Path | str) -> None:
        self.root_dir = Path(root_dir)

    def load_all(self) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        if not self.root_dir.exists():
            return records

        for root, _, files in os.walk(self.root_dir):
            if ".git" in root or ".worktrees" in root:
                continue
            for file in sorted(files):
                if file.endswith(".csv"):
                    csv_path = Path(root) / file
                    records.extend(self._load_file(csv_path))

        return records

    def _load_file(self, file_path: Path) -> list[DatasetRecord]:
        records: list[DatasetRecord] = []
        filename = file_path.name
        metadata = FilenameMetadataParser.parse(filename)

        with open(file_path, encoding="utf-8", errors="ignore") as f:
            sample = f.readline()
            if not sample:
                return records
            f.seek(0)

            # Detect delimiter
            delimiter = ";" if sample.count(";") > sample.count(",") else ","
            reader = csv.reader(f, delimiter=delimiter)

            try:
                headers = [h.strip(" \t\r\n;,").lower() for h in next(reader)]
            except Exception:
                return records

            text_col_idx = self._find_col(headers, self._TEXT_COLUMNS)
            label_col_idx = self._find_col(headers, self._LABEL_COLUMNS)

            if text_col_idx is None or label_col_idx is None:
                return records

            for row_idx, row in enumerate(reader):
                if len(row) <= max(text_col_idx, label_col_idx):
                    continue

                raw_text = row[text_col_idx]
                raw_label = row[label_col_idx].strip(" \t\r\n;,")

                cleaned_text = TextSanitizer.clean(raw_text)
                if not cleaned_text:
                    continue

                label = LabelNormalizer.normalize(raw_label)
                if label is None:
                    continue

                record = DatasetRecord(
                    id=f"{filename}_{row_idx}",
                    text=cleaned_text,
                    label=label,
                    source=filename,
                    metadata=dict(metadata),
                )
                records.append(record)

        return records

    def _find_col(self, headers: list[str], candidates: list[str]) -> Optional[int]:
        for idx, col in enumerate(headers):
            # Strip BOM if present
            clean_col = col[1:] if col.startswith("\ufeff") else col
            if clean_col in candidates:
                return idx
        return None
