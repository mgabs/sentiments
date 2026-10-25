import math
import random
from collections import defaultdict

from sentiments.core import SentimentLabel
from sentiments.dataset.models import DatasetRecord, DatasetSplit


class StratifiedSplitter:
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        total = train_ratio + val_ratio + test_ratio
        if not math.isclose(total, 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
            raise ValueError("split ratios must be non-negative")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, records: list[DatasetRecord]) -> DatasetSplit:
        by_class: dict[SentimentLabel, list[DatasetRecord]] = defaultdict(list)
        for r in records:
            by_class[r.label].append(r)

        train_records: list[DatasetRecord] = []
        val_records: list[DatasetRecord] = []
        test_records: list[DatasetRecord] = []

        rng = random.Random(self.seed)

        for _, class_records in sorted(by_class.items()):
            # Sort first for deterministic ordering before shuffle
            shuffled = sorted(class_records, key=lambda r: r.id)
            rng.shuffle(shuffled)

            n = len(shuffled)
            n_train = int(round(n * self.train_ratio))
            n_val = int(round(n * self.val_ratio))

            train_records.extend(shuffled[:n_train])
            val_records.extend(shuffled[n_train : n_train + n_val])
            test_records.extend(shuffled[n_train + n_val :])

        return DatasetSplit(
            train=train_records,
            val=val_records,
            test=test_records,
        )
