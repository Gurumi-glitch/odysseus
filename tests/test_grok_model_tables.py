"""Pin xAI Grok context windows and prices in the three known-model tables.

grok-4.7 used to substring-match the generic 'grok-4' key and showed a 131k
window at $3/$15. Values come from the official xAI models page
(https://docs.x.ai/developers/models, checked 2026-10-12); prices are the
<200k-prompt tier.

The frontend table is driven through node against the real MODEL_INFO literal
and matchModelKey (chatRenderer.js pulls in browser-only modules, so it can't be
imported whole). Those cases skip when node isn't installed.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.model_context import _lookup_known
from src.model_pricing import estimate_cost_usd

_REPO = Path(__file__).resolve().parent.parent
_RENDERER = _REPO / "static" / "js" / "chatRenderer.js"
_MATCH_KEY = _REPO / "static" / "js" / "model" / "matchKey.js"
_XAI_URL = "https://api.x.ai/v1"

# model id -> (context window, $ input / 1M, $ output / 1M)
GROK = {
    "grok-4.7": (500_000, 2.00, 6.00),
    "grok-4.6": (500_000, 2.00, 6.00),
    "grok-4.5": (500_000, 2.00, 6.00),
    "grok-4.3": (1_000_000, 1.25, 2.50),
    "grok-4.20-0309-reasoning": (1_000_000, 1.25, 2.50),
    "grok-4.20-0309-non-reasoning": (1_000_000, 1.25, 2.50),
    "grok-4.20-multi-agent-0309": (1_000_000, 1.25, 2.50),
    "grok-build-0.1": (256_000, 1.00, 2.00),
}
# Provider-prefixed ids (OpenRouter) must resolve the same way.
_IDS = list(GROK) + ["x-ai/grok-4.7", "x-ai/grok-4.5"]


def _expected(model_id):
    return GROK[model_id.split("/")[-1]]


@pytest.mark.parametrize("model_id", _IDS)
def test_backend_context_window(model_id):
    assert _lookup_known(model_id) == _expected(model_id)[0]


@pytest.mark.parametrize("model_id", _IDS)
def test_backend_price(model_id):
    _, price_in, price_out = _expected(model_id)
    cost = estimate_cost_usd(model_id, 1_000_000, 1_000_000, _XAI_URL)
    assert cost == pytest.approx(price_in + price_out)


def _frontend_info(model_ids):
    src = _RENDERER.read_text(encoding="utf-8")
    table = re.search(r"const MODEL_INFO = \{.*?\n\};", src, re.DOTALL)
    assert table, "MODEL_INFO not found in chatRenderer.js"
    match_key = _MATCH_KEY.read_text(encoding="utf-8").replace("export function", "function")
    js = (
        match_key + "\n" + table.group(0) + "\n"
        + f"const ids = {json.dumps(model_ids)};\n"
        + "console.log(JSON.stringify(ids.map(id => MODEL_INFO[matchModelKey(id, Object.keys(MODEL_INFO))] || null)));"
    )
    proc = subprocess.run(
        ["node", "--input-type=module"],
        input=js, capture_output=True, text=True, cwd=str(_REPO), timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return dict(zip(model_ids, json.loads(proc.stdout)))


@pytest.mark.skipif(shutil.which("node") is None, reason="node binary not on PATH")
def test_frontend_model_info():
    infos = _frontend_info(_IDS)
    for model_id, info in infos.items():
        ctx, price_in, price_out = _expected(model_id)
        assert info == {"input": price_in, "output": price_out, "ctx": ctx}, model_id
