Feature: Sentiment Core Domain Contracts and Taxonomy

  Scenario: Validating canonical sentiment labels
    Given the sentiment taxonomy
    Then "POSITIVE" is a valid SentimentLabel
    And "NEGATIVE" is a valid SentimentLabel
    And "NEUTRAL" is a valid SentimentLabel
    And "UNKNOWN" is rejected as an invalid SentimentLabel

  Scenario: Creating a valid SentimentResult
    Given a sentiment classification with label "POSITIVE", confidence 0.95, latency 1.2 ms, tier "TIER_0", and model "majority_baseline"
    Then the result has label "POSITIVE"
    And the result has confidence 0.95
    And the result has latency 1.2 ms
    And the result has tier "TIER_0"
    And the result has model name "majority_baseline"
    And the result is immutable

  Scenario Outline: Validating confidence bounds
    Given a sentiment classification attempt with confidence <confidence>
    Then the result creation should <outcome>

    Examples:
      | confidence | outcome |
      | 0.0        | succeed |
      | 0.5        | succeed |
      | 1.0        | succeed |
      | -0.01      | fail    |
      | 1.01       | fail    |

  Scenario: Validating non-negative latency
    Given a sentiment classification attempt with latency -0.5 ms
    Then the latency validation should fail with a value error

  Scenario: Verifying classifier protocol conformance
    Given a compliant mock sentiment classifier
    Then it conforms to the SentimentClassifier protocol
    And it exposes tier and model_name attributes
    And it supports single text classification returning a SentimentResult
    And it supports batch classification returning a list of SentimentResults
