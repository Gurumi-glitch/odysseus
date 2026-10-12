"""Regressions for the spec #44 code-review findings."""
from src.chat_agent.loop import Candidate, _TurnUsage
from src.chat_agent.providers import Endpoint, RoundEnd
from src.model_context import estimate_text_tokens


def test_estimated_output_tokens_use_the_shared_unicode_aware_estimate():
    usage = _TurnUsage(0.0, Candidate(Endpoint("http://x", "m", None), {}))
    text = "你好世界" * 10
    usage.add_round(1, RoundEnd([{"type": "text", "text": text}], None), [])
    bucket = usage.buckets[0]
    assert bucket["usage_source"] == "estimated"
    assert bucket["output_tokens"] == estimate_text_tokens(text)
