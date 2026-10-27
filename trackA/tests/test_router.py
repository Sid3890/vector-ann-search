from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from router import query_router as qr

VEC = [0.0] * 128


def make_client(monkeypatch, handler, timeout="0.3"):
    monkeypatch.setenv("SHARD_URLS", "0=http://s0,1=http://s1,2=http://s2")
    monkeypatch.setenv("SHARD_TIMEOUT_S", timeout)
    tc = TestClient(qr.app)
    tc.__enter__()
    qr.app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tc


def good(ids, dists):
    return httpx.Response(200, json={"ids": ids, "distances": dists})


def test_merge_sorted_across_shards(monkeypatch):
    data = {
        "s0": ([1, 2], [0.1, 0.5]),
        "s1": ([3, 4], [0.2, 0.3]),
        "s2": ([5, 6], [0.05, 0.9]),
    }

    async def handler(req):
        return good(*data[req.url.host])

    tc = make_client(monkeypatch, handler)
    r = tc.post("/search", json={"query_vector": VEC, "top_k": 4})
    assert r.status_code == 200
    b = r.json()
    assert b["ids"] == ["5", "1", "3", "4"]
    assert b["distances"] == sorted(b["distances"])
    assert b["partial"] is False


def test_timeout_and_5xx_are_partial(monkeypatch):
    async def handler(req):
        if req.url.host == "s0":
            await asyncio.sleep(2)
        if req.url.host == "s1":
            return httpx.Response(500)
        return good(["a"], [0.4])

    tc = make_client(monkeypatch, handler)
    r = tc.post("/search", json={"query_vector": VEC, "top_k": 3})
    assert r.status_code == 200
    assert r.json()["partial"] is True
    assert r.json()["ids"] == ["a"]
    assert r.headers["X-Failed-Shards"] == "0,1"
    assert r.json()["latency_ms"] < 1000


def test_all_fail_503(monkeypatch):
    async def handler(req):
        return httpx.Response(502)

    tc = make_client(monkeypatch, handler)
    r = tc.post("/search", json={"query_vector": VEC, "top_k": 3})
    assert r.status_code == 503


def test_bad_shard_payload_and_padding(monkeypatch):
    async def handler(req):
        if req.url.host == "s0":
            return httpx.Response(200, json={"ids": ["x"], "distances": []})
        if req.url.host == "s1":
            return good([-1, 7], [3.4e38, 0.2])
        return httpx.Response(200, text="not json")

    tc = make_client(monkeypatch, handler)
    r = tc.post("/search", json={"query_vector": VEC, "top_k": 5})
    assert r.json()["ids"] == ["7"]
    assert r.json()["partial"] is True


def test_validation(monkeypatch):
    async def handler(req):
        return good([], [])

    tc = make_client(monkeypatch, handler)
    assert tc.post("/search", json={"query_vector": [1.0], "top_k": 3}).status_code == 422
    assert tc.post("/search", json={"query_vector": VEC, "top_k": 0}).status_code == 422
