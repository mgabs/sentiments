"""Sentiment classifier providers."""

from sentiments.providers.tier0 import MajorityClassClassifier, LexiconRuleClassifier

__all__ = [
    "MajorityClassClassifier",
    "LexiconRuleClassifier",
]
