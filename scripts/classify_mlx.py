import sys
from sentiments.providers.tier2 import Tier2SLMClassifier, MLXBackend


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else "The battery life is incredible and screen looks stunning!"
    model_name = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Qwen2.5-3B-Instruct-4bit"

    print(f"Loading MLX model: {model_name}...")
    backend = MLXBackend(model_name=model_name)
    clf = Tier2SLMClassifier(backend=backend)

    print(f"\nClassifying:\n\"{text}\"\n")
    result = clf.classify(text)

    print("Result:")
    print(f"  Label:      {result.label.value}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Latency:    {result.latency_ms:.2f} ms")
    print(f"  Tier:       {result.tier.value}")
    print(f"  Model:      {result.model_name}")
    if result.metadata:
        print(f"  Metadata:   {result.metadata}")


if __name__ == "__main__":
    main()
