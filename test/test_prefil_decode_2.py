# tests/test_generate_prefill_decode.py
"""
Exercises exactly ONE prefill forward pass and ONE decode forward pass
against an already-running dummy-platform SGLang server.

How this maps to SGLang internals:
- The first forward pass for any request is always a prefill (processes
  the full input prompt, produces the 1st output token).
- Every forward pass after that is a decode step (processes exactly 1 new
  token, produces the next output token).

So max_new_tokens=2 is the minimal request shape that guarantees:
  forward call #1 -> prefill  (produces token 1)
  forward call #2 -> decode   (produces token 2)

max_new_tokens=1 would only ever trigger a prefill (no decode step
follows), which is why 2 is the minimum for this test's purpose.
"""
import pytest
import requests

SERVER_HOST = "localhost"
SERVER_PORT = 30000


@pytest.fixture(scope="session")
def base_url():
    return f"http://{SERVER_HOST}:{SERVER_PORT}"


@pytest.fixture(scope="session")
def http_session():
    with requests.Session() as session:
        yield session


@pytest.fixture
def generate_payload():
    return {
        "text": "The capital of France is",
        "sampling_params": {
            "max_new_tokens": 2,   # 1 prefill token + 1 decode token
            "temperature": 0,
            "ignore_eos": True,    # don't let random dummy-weight logits
                                   # short-circuit before the decode step
        },
        "return_logprob": True,    # surfaces completion_tokens in meta_info
    }


@pytest.mark.integration
class TestGeneratePrefillDecode:
    """Requires a live SGLang server (SGLANG_PLATFORM=dummy) on
    localhost:30000 -- these are integration tests, not unit tests."""

    def test_response_is_successful(self, http_session, base_url, generate_payload):
        resp = http_session.post(f"{base_url}/generate", json=generate_payload)
        assert resp.status_code == 200
        assert "text" in resp.json()

    def test_exactly_one_prefill_and_one_decode(self, http_session, base_url, generate_payload):
        resp = http_session.post(f"{base_url}/generate", json=generate_payload)
        resp.raise_for_status()
        data = resp.json()

        meta = data.get("meta_info", {})
        completion_tokens = meta.get("completion_tokens")

        # Exactly 2 completion tokens => exactly 1 prefill + 1 decode occurred.
        assert completion_tokens == 2, (
            f"Expected 2 completion tokens (1 prefill + 1 decode), "
            f"got {completion_tokens}. If this is 1, ignore_eos may not be "
            f"taking effect, or the server produced an early stop."
        )

    def test_prompt_tokens_reported(self, http_session, base_url, generate_payload):
        resp = http_session.post(f"{base_url}/generate", json=generate_payload)
        resp.raise_for_status()
        meta = resp.json().get("meta_info", {})

        assert meta.get("prompt_tokens", 0) > 0, (
            "Expected prompt_tokens > 0 in meta_info from the prefill pass"
        )