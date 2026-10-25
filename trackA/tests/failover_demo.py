"""Cross-platform failover demo (no bash/curl needed). Run from the project root:
    python tests/failover_demo.py
Shard 0: one replica (:8013). Shard 1: leader :8011, follower :8012. Router :8010.
"""
import json, os, subprocess, sys, time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
ROUTER = "http://127.0.0.1:8010"
Q = {"query_vector": [0.1, 0.2, 0.3, 0.4], "top_k": 4}
procs: list[subprocess.Popen] = []

def spawn(args: list[str], **env: str) -> subprocess.Popen:
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", *args, "--log-level", "warning"],
                         cwd=ROOT, env={**os.environ, **env})
    procs.append(p)
    return p

def mock(port: int, shard: int, label: str) -> subprocess.Popen:
    return spawn(["demo_mock_shard:app", "--app-dir", str(TESTS), "--port", str(port)],
                 SHARD_ID=str(shard), LABEL=label)

def wait_up(url: str, timeout: float = 15.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            if httpx.get(url, timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.3)
    raise RuntimeError(f"{url} did not come up")

def search() -> dict:
    r = httpx.post(f"{ROUTER}/search", json=Q, timeout=5)
    return {"status": r.status_code, **r.json()}

def main() -> int:
    ok = True
    def check(name: str, cond: bool) -> None:
        nonlocal ok
        ok &= cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    try:
        mock(8013, 0, "s0")
        leader = mock(8011, 1, "leader")
        follower = mock(8012, 1, "follower")
        spawn(["router.query_router:app", "--port", "8010"],
              SHARD_REPLICAS_JSON=json.dumps({"0": ["http://127.0.0.1:8013"],
                                             "1": ["http://127.0.0.1:8011", "http://127.0.0.1:8012"]}),
              VECTOR_DIM="4", HEALTH_INTERVAL_S="3")
        for port in (8013, 8011, 8012):
            wait_up(f"http://127.0.0.1:{port}/health")
        wait_up(f"{ROUTER}/healthz")

        print("1. all healthy")
        r = search(); print("  ", r)
        check("not partial, 4 ids, sorted", not r["partial"] and len(r["ids"]) == 4
              and r["distances"] == sorted(r["distances"]))

        print("2. kill shard-1 leader, search immediately (live-failure path)")
        leader.terminate(); leader.wait(); time.sleep(0.3)
        r = search(); print("  ", r)
        check("still complete (follower served shard 1)", not r["partial"] and any(i.startswith("s1-") for i in r["ids"]))

        print("3. wait for health poll; leader should be skipped")
        time.sleep(4)
        state = {s["url"]: s["healthy"] for s in httpx.get(f"{ROUTER}/replicas").json()["state"]}
        check("leader marked unhealthy", state["http://127.0.0.1:8011"] is False)
        r = search(); check("still complete", not r["partial"])

        print("4. failover log")
        ev = httpx.get(f"{ROUTER}/failovers").json()
        for e in ev[:3]: print("  ", {k: e[k] for k in ("shard_id", "failed_replica", "recovered_replica", "reason")})
        check("event logged leader->follower", any(e["failed_replica"].endswith(":8011")
              and (e["recovered_replica"] or "").endswith(":8012") for e in ev))

        print("5. kill follower too -> shard 1 gone, expect partial")
        follower.terminate(); follower.wait(); time.sleep(0.3)
        r = search(); print("  ", r)
        check("partial=true, shard 0 results returned", r["partial"] and r["ids"] and all(i.startswith("s0-") for i in r["ids"]))

        print("6. wrong dimension -> 422")
        code = httpx.post(f"{ROUTER}/search", json={"query_vector": [1], "top_k": 3}).status_code
        check("422", code == 422)
    finally:
        for p in procs:
            p.terminate()
    print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
