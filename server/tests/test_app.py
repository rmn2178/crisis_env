"""Tests for the FastAPI application (server/app.py).

Uses httpx.AsyncClient with the ASGI transport for fast in-process testing.
"""

from __future__ import annotations

import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
from httpx import ASGITransport, AsyncClient

from server.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ===================================================================
# Health & system endpoints
# ===================================================================


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_body(self, client):
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "ok"
        assert body["version"] == "1.0.0"
        assert isinstance(body["tasks_available"], list)


class TestTasks:
    @pytest.mark.asyncio
    async def test_tasks_returns_list(self, client):
        resp = await client.get("/tasks")
        assert resp.status_code == 200
        tasks = resp.json()
        assert isinstance(tasks, list)
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_tasks_have_required_fields(self, client):
        resp = await client.get("/tasks")
        for task in resp.json():
            assert "id" in task
            assert "name" in task
            assert "difficulty" in task
            assert "description" in task


# ===================================================================
# Reset
# ===================================================================


class TestAppReset:
    @pytest.mark.asyncio
    async def test_reset_easy(self, client):
        resp = await client.post("/reset", json={"task_id": "easy"})
        assert resp.status_code == 200
        obs = resp.json()
        assert "threat_id" in obs
        assert obs["done"] is False

    @pytest.mark.asyncio
    async def test_reset_invalid_task(self, client):
        resp = await client.post("/reset", json={"task_id": "extreme"})
        assert resp.status_code == 400


# ===================================================================
# Step
# ===================================================================


class TestAppStep:
    @pytest.mark.asyncio
    async def test_step_valid(self, client):
        # Reset first
        reset_resp = await client.post("/reset", json={"task_id": "easy"})
        obs = reset_resp.json()

        step_resp = await client.post(
            "/step",
            json={
                "action_type": "classify",
                "threat_id": obs["threat_id"],
                "priority_level": "CRITICAL",
            },
        )
        assert step_resp.status_code == 200
        result = step_resp.json()
        assert "observation" in result
        assert "reward" in result
        assert "done" in result


# ===================================================================
# State
# ===================================================================


class TestAppState:
    @pytest.mark.asyncio
    async def test_state_returns_dict(self, client):
        resp = await client.get("/state")
        assert resp.status_code == 200
        state = resp.json()
        assert "episode_id" in state
        assert "step_count" in state
        assert "task_id" in state


# ===================================================================
# Root redirect
# ===================================================================


class TestRoot:
    @pytest.mark.asyncio
    async def test_root_redirects_to_docs(self, client):
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers.get("location", "")
