Feature: Tier 2 Local SLM Classifier (Ollama & MLX)

  Scenario: Successful sentiment classification via Ollama backend
    Given an Ollama backend configured with model "llama3.2:3b" returning valid JSON "POSITIVE" with confidence 0.95
    When the Tier2SLMClassifier classifies "The product quality exceeded all my expectations!"
    Then the result has tier "TIER_2"
    And the result has model name "llama3.2:3b"
    And the result has label "POSITIVE"
    And the result confidence is 0.95
    And the latency in milliseconds is positive

  Scenario: Successful sentiment classification via MLX backend
    Given an MLX backend configured with model "qwen2.5:3b" returning valid JSON "NEGATIVE" with confidence 0.92
    When the Tier2SLMClassifier classifies "Completely defective item and rude customer support."
    Then the result has tier "TIER_2"
    And the result has model name "qwen2.5:3b"
    And the result has label "NEGATIVE"
    And the result confidence is 0.92

  Scenario: Ollama daemon connection refusal diagnostics
    Given an Ollama backend pointing to an unreachable port
    When the Tier2SLMClassifier attempts classification
    Then it raises a ClassifierProviderError diagnosing unreachable Ollama daemon

  Scenario: MLX backend platform incompatibility handling
    Given an MLX backend on an unsupported platform or missing mlx-lm library
    When checking backend availability or attempting generation
    Then it indicates backend is unavailable or raises ClassifierProviderError

  Scenario: Robust parsing of markdown code-fenced JSON responses
    Given a backend returning markdown fenced JSON with extra commentary
    When the Tier2SLMClassifier classifies "Package arrived on time as expected."
    Then the result has label "NEUTRAL"
    And the result confidence is clamped between 0.0 and 1.0

  Scenario: Rejection of empty or whitespace review text
    Given a configured Tier2SLMClassifier
    When classifying an empty or whitespace string
    Then it raises a ValueError before invoking the backend
