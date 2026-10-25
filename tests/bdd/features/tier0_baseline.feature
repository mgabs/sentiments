Feature: Tier 0 Dumb Baseline Classifiers

  Scenario: MajorityClassClassifier predicts mode of training distribution
    Given a training dataset with 80 positive and 20 negative records
    When a MajorityClassClassifier is fitted on the training dataset
    Then its tier is "TIER_0"
    And its model name is "majority_class_baseline"
    And classifying any input text "Any product review" yields label "POSITIVE"
    And the prediction confidence is 0.80
    And the inference latency is under 1.0 ms

  Scenario: MajorityClassClassifier tie breaking
    Given a training dataset with 50 positive and 50 negative records
    When a MajorityClassClassifier is fitted on the training dataset
    Then classifying text produces a deterministic label
    And the prediction confidence is 0.50

  Scenario Outline: LexiconRuleClassifier polarity determination
    Given an initialized LexiconRuleClassifier
    When the classifier evaluates "<text>"
    Then the resulting sentiment label is "<expected_label>"
    And the tier is "TIER_0"
    And the model name is "lexicon_rule_baseline"
    And the inference latency is under 1.0 ms

    Examples:
      | text                                | expected_label |
      | This is an amazing excellent product| POSITIVE       |
      | Absolutely terrible awful broke     | NEGATIVE       |
      | Standard normal ordinary device     | NEUTRAL        |
      |                                     | NEUTRAL        |

  Scenario: Tier 0 batch classification execution
    Given an initialized LexiconRuleClassifier
    When classifying a batch of 3 customer feedback reviews
    Then it returns a list of 3 SentimentResults
    And each result has tier "TIER_0"
