Feature: Tier 4 Quantized ONNX & SetFit CPU Runtime

  Scenario: Successful sentiment classification via ONNX backend
    Given an ONNX backend configured with model "minilm-int8" returning "POSITIVE" with confidence 0.94
    When the ONNXClassifier classifies "Outstanding customer support and rapid delivery!"
    Then the result has tier "TIER_4"
    And the result has model name "minilm-int8"
    And the result has label "POSITIVE"
    And the result confidence is 0.94
    And the latency in milliseconds is positive

  Scenario: Vectorized batch classification with order preservation
    Given an ONNX backend configured with model "minilm-int8"
    When the ONNXClassifier classifies a batch of 3 texts
    Then exactly 3 results are returned in matching order
    And each result has tier "TIER_4"
    And each result has valid sentiment labels and confidence scores

  Scenario: Rejection of empty or whitespace review text
    Given a configured ONNXClassifier
    When classifying an empty or whitespace string with ONNXClassifier
    Then it raises a ValueError with message containing "text must be a non-empty string"

  Scenario: Resilient fallback handling in batch mode when fail_fast is false
    Given an ONNX backend that fails on specific text
    When the ONNXClassifier classifies a batch with fail_fast set to false
    Then failed samples return fallback NEUTRAL results with error metadata
    And valid samples return successful classifications

  Scenario: Fail-fast error propagation in batch mode when fail_fast is true
    Given an ONNX backend that fails on specific text
    When the ONNXClassifier classifies a batch with fail_fast set to true
    Then it raises a ClassifierProviderError immediately

  Scenario: Zero external network calls during inference
    Given an initialized ONNXClassifier
    When classifying text while all network sockets are blocked
    Then classification succeeds without initiating any network connections
