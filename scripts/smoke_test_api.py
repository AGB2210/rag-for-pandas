"""Start the real API server, check every endpoint once, and stop it.

An end-to-end check with the real models; the unit tests use fake ones. The
server is stopped by the process id this script started, together with any
child processes, so no other process is touched.

Usage:
  python scripts/smoke_test_api.py [--port 8765] [--no-answers]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import httpx

from docsearch.pipeline import DISABLED, GENERATOR_ENV

STARTUP_TIMEOUT_SECONDS = 600


def stop(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    if os.name == "nt":
        # On Windows a virtual-environment python.exe starts the real interpreter as a
        # child process; /T stops that whole tree, found from our own process id.
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(server.pid)], capture_output=True, check=False)
    else:
        server.terminate()
    server.wait(timeout=30)


def wait_until_up(server: subprocess.Popen, url: str) -> float:
    start = time.perf_counter()
    while time.perf_counter() - start < STARTUP_TIMEOUT_SECONDS:
        if server.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {server.returncode}")
        try:
            if httpx.get(f"{url}/health", timeout=5).status_code == 200:
                return time.perf_counter() - start
        except httpx.TransportError:
            pass
        time.sleep(2)
    raise RuntimeError(f"server did not start within {STARTUP_TIMEOUT_SECONDS} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-answers", action="store_true", help="start without the generator and expect /answer to return 503")
    args = parser.parse_args()

    url = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ)
    if args.no_answers:
        env[GENERATOR_ENV] = DISABLED

    results: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        results.append((name, passed, detail))
        print(f"{'PASS' if passed else 'FAIL'}  {name}  {detail}", flush=True)

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "docsearch.api:app", "--host", "127.0.0.1", "--port", str(args.port)], env=env
    )
    try:
        check("server starts", True, f"{wait_until_up(server, url):.0f} s")
        with httpx.Client(base_url=url, timeout=120) as client:
            health = client.get("/health")
            body = health.json()
            check("GET /health", health.status_code == 200 and body["documents"] > 0, str(body))

            search = client.post("/search", json={"query": "delete rows with missing values", "k": 3})
            ranks = [r["rank"] for r in search.json().get("results", [])]
            check("POST /search", search.status_code == 200 and ranks == [1, 2, 3],
                  str([(r["name"], r["score"]) for r in search.json().get("results", [])]))

            blank = client.post("/search", json={"query": "   "})
            check("POST /search rejects a blank query", blank.status_code == 422, f"status {blank.status_code}")

            start = time.perf_counter()
            answer = client.post("/answer", json={"question": "How do I delete rows that contain missing values?"})
            seconds = time.perf_counter() - start
            if args.no_answers:
                check("POST /answer disabled", answer.status_code == 503, f"status {answer.status_code}")
            else:
                reply = answer.json()
                check("POST /answer", answer.status_code == 200 and len(reply["sources"]) == 3,
                      f"{seconds:.1f} s, cited {[s['name'] for s in reply['sources'] if s['cited']]}, answer: {reply['answer'][:120]!r}")

            check("GET /docs", client.get("/docs").status_code == 200)
    except Exception as error:  # report any failure as a failed check rather than a traceback
        check("unexpected error", False, repr(error))
    finally:
        stop(server)

    failed = [name for name, passed, _ in results if not passed]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
