Feature: Multi-Tier Provider Contracts and Adapters

  Scenario: Tier 1 Frontier LLM Classifier handles responses and token accounting
    Given a mock FrontierLLM client returning label "POSITIVE", confidence 0.96, and 45 tokens
    When the FrontierLLMClassifier evaluates "Remarkable quality and customer support"
    Then the result has tier "TIER_1"
    And the result has label "POSITIVE"
    And the result confidence is 0.96
    And the metadata contains token count 45

  Scenario: Tier 1 Frontier LLM Classifier handles network timeouts
    Given a mock FrontierLLM client that simulates a timeout error
    When the FrontierLLMClassifier attempts classification
    Then it raises a ClassifierProviderError indicating timeout

  Scenario: Tier 2 Compact SLM Classifier execution
    Given a mock CompactSLM endpoint returning label "NEGATIVE" and confidence 0.88
    When the CompactSLMClassifier evaluates "Battery life degraded after one week"
    Then the result has tier "TIER_2"
    And the result has label "NEGATIVE"
    And the result confidence is 0.88

  Scenario: Tier 3 Fine-Tuned Encoder Classifier execution
    Given a mock FineTunedEncoder model returning label "NEUTRAL" and confidence 0.72
    When the FineTunedEncoderClassifier evaluates "The device arrived on Tuesday"
    Then the result has tier "TIER_3"
    And the result has label "NEUTRAL"
    And the result confidence is 0.72

  Scenario: Tier 4 Ultra-Lightweight ONNX Classifier execution
    Given a mock ONNX runtime model returning label "POSITIVE" and confidence 0.91
    When the ONNXClassifier evaluates "Super fast and lightweight"
    Then the result has tier "TIER_4"
    And the result has label "POSITIVE"
    And the result confidence is 0.91
