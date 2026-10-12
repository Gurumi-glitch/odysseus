"""Audit-all results carry the card state the Skills page shows."""

import asyncio

from routes import skills_routes as routes


def test_each_audit_all_result_carries_the_skills_card_state(monkeypatch):
    class Skills:
        def load(self, owner=None):
            return [{"name": "dup", "status": "binned"}]

    async def audit(manager, skill, *args, **kwargs):
        return {"skill": skill["name"], "result": "fail"}

    monkeypatch.setattr(routes, "_audit_one_skill", audit)
    monkeypatch.setattr(routes, "_finalize_audit_batch", lambda *args: None)
    job = {"status": "running", "log": [], "results": [], "done": 0}
    monkeypatch.setattr(routes, "_skill_audit_jobs", {("alice",): job})

    asyncio.run(routes._run_audit_all_job(
        ("alice",), Skills(), ["dup"], "url", "model", {}, None, "alice",
    ))

    [result] = job["results"]
    assert result["skill_state"]["state"] == "binned"
