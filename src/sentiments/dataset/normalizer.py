import html
import re
from typing import Optional, Any

from sentiments.core import SentimentLabel


class LabelNormalizer:
    _MAPPINGS: dict[str, SentimentLabel] = {
        "positive": SentimentLabel.POSITIVE,
        "pos": SentimentLabel.POSITIVE,
        "positive review": SentimentLabel.POSITIVE,
        "negative": SentimentLabel.NEGATIVE,
        "neg": SentimentLabel.NEGATIVE,
        "negative review": SentimentLabel.NEGATIVE,
        "neutral": SentimentLabel.NEUTRAL,
        "n/a": SentimentLabel.NEUTRAL,
        "neut": SentimentLabel.NEUTRAL,
    }

    @classmethod
    def normalize(cls, raw: Optional[str]) -> Optional[SentimentLabel]:
        if not raw:
            return None
        cleaned = raw.strip().lower()
        return cls._MAPPINGS.get(cleaned)


class FilenameMetadataParser:
    # Pattern e.g. "Product (Category) - Platform Product Reviews - English.csv"
    _PATTERN = re.compile(
        r"^(?P<product>[^(]+?)(?:\s*\((?P<category>[^)]+)\))?\s*-\s*(?P<platform>[^-]+?)(?:Product|Customer|User)?\s*Reviews?(?:\s*-\s*(?P<language>[^.]+))?\.csv$",
        re.IGNORECASE,
    )
    _TRAINING_PATTERN = re.compile(
        r"^(?P<product>[^(]+?)(?:\s*\((?P<category>[^)]+)\))?\s*-\s*(?P<platform>Training Set)\.csv$",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, filename: str) -> dict[str, Any]:
        match = cls._PATTERN.match(filename)
        if match:
            gd = match.groupdict()
            meta: dict[str, Any] = {"product": gd["product"].strip()}
            if gd.get("category"):
                meta["category"] = gd["category"].strip()
            if gd.get("platform"):
                meta["platform"] = gd["platform"].strip()
            if gd.get("language"):
                meta["language"] = gd["language"].strip()
            return meta

        t_match = cls._TRAINING_PATTERN.match(filename)
        if t_match:
            gd = t_match.groupdict()
            meta = {"product": gd["product"].strip()}
            if gd.get("category"):
                meta["category"] = gd["category"].strip()
            if gd.get("platform"):
                meta["platform"] = gd["platform"].strip()
            return meta

        base = filename.rsplit(".", 1)[0]
        return {"product": base}


class TextSanitizer:
    _HTML_TAG_RE = re.compile(r"<[^>]+>")

    @classmethod
    def clean(cls, text: Optional[str]) -> str:
        if not text:
            return ""
        # Normalize escaped control characters
        text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
        # Remove HTML tags
        no_html = cls._HTML_TAG_RE.sub(" ", text)
        # Unescape HTML entities
        unescaped = html.unescape(no_html)
        # Normalize whitespace
        cleaned = " ".join(unescaped.split())
        return cleaned.strip()
