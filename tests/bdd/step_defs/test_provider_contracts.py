import pytest
from pytest_bdd import scenarios, given, when, then, parsers

from sentiments.core import SentimentLabel, Tier, SentimentResult
from sentiments.providers.base import ClassifierProviderError
from sentiments.providers.tier1 import FrontierLLMClassifier
from sentiments.providers.tier2 import CompactSLMClassifier
from sentiments.providers.tier3 import FineTunedEncoderClassifier
from sentiments.providers.tier4 import ONNXClassifier

scenarios("../features/provider_contracts.feature")


@pytest.fixture
def provider_state():
    return {
        "clf": None,
        "result": None,
        "error": None,
    }


@given(
    parsers.parse(
        'a mock FrontierLLM client returning label "{label}", confidence {confidence:f}, and {tokens:d} tokens'
    )
)
def mock_frontier_client(provider_state, label, confidence, tokens):
    def client(prompt: str) -> dict:
        return {"label": label, "confidence": confidence, "tokens": tokens}

    provider_state["clf"] = FrontierLLMClassifier(client_fn=client, model_name="claude-3-5-sonnet")


@when(parsers.parse('the FrontierLLMClassifier evaluates "{text}"'))
def eval_frontier(provider_state, text):
    provider_state["result"] = provider_state["clf"].classify(text)


@then(parsers.parse('the result has tier "{tier}"'))
def verify_result_tier(provider_state, tier):
    assert provider_state["result"].tier == Tier(tier)


@then(parsers.parse('the result has label "{label}"'))
def verify_result_label(provider_state, label):
    assert provider_state["result"].label == SentimentLabel(label)


@then(parsers.parse("the result confidence is {confidence:f}"))
def verify_result_confidence(provider_state, confidence):
    assert provider_state["result"].confidence == pytest.approx(confidence)


@then(parsers.parse("the metadata contains token count {tokens:d}"))
def verify_token_count(provider_state, tokens):
    assert provider_state["result"].metadata.get("tokens") == tokens


@given("a mock FrontierLLM client that simulates a timeout error")
def mock_timeout_client(provider_state):
    def client(prompt: str) -> dict:
        raise TimeoutError("Endpoint timed out")

    provider_state["clf"] = FrontierLLMClassifier(client_fn=client, model_name="gpt-4o")


@when("the FrontierLLMClassifier attempts classification")
def attempt_frontier_classification(provider_state):
    try:
        provider_state["clf"].classify("test text")
    except Exception as e:
        provider_state["error"] = e


@then("it raises a ClassifierProviderError indicating timeout")
def verify_timeout_error(provider_state):
    assert isinstance(provider_state["error"], ClassifierProviderError)
    assert "Endpoint timed out" in str(provider_state["error"])


@given(parsers.parse('a mock CompactSLM endpoint returning label "{label}" and confidence {confidence:f}'))
def mock_compact_slm_endpoint(provider_state, label, confidence):
    def client(prompt: str) -> dict:
        return {"label": label, "confidence": confidence}

    provider_state["clf"] = CompactSLMClassifier(client_fn=client, model_name="llama-3-8b")


@when(parsers.parse('the CompactSLMClassifier evaluates "{text}"'))
def eval_compact_slm(provider_state, text):
    provider_state["result"] = provider_state["clf"].classify(text)


@given(parsers.parse('a mock FineTunedEncoder model returning label "{label}" and confidence {confidence:f}'))
def mock_encoder_model(provider_state, label, confidence):
    def pipeline(text: str) -> dict:
        return {"label": label, "confidence": confidence}

    provider_state["clf"] = FineTunedEncoderClassifier(pipeline_fn=pipeline, model_name="roberta-sentiment")


@when(parsers.parse('the FineTunedEncoderClassifier evaluates "{text}"'))
def eval_encoder(provider_state, text):
    provider_state["result"] = provider_state["clf"].classify(text)


@given(parsers.parse('a mock ONNX runtime model returning label "{label}" and confidence {confidence:f}'))
def mock_onnx_model(provider_state, label, confidence):
    def inference(text: str) -> dict:
        return {"label": label, "confidence": confidence}

    provider_state["clf"] = ONNXClassifier(inference_fn=inference, model_name="distilbert-onnx")


@when(parsers.parse('the ONNXClassifier evaluates "{text}"'))
def eval_onnx(provider_state, text):
    provider_state["result"] = provider_state["clf"].classify(text)
