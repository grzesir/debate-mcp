#!/usr/bin/env python3
"""
Benchmark driver for the debate MCP server.

Spawns the server as a subprocess, sends MCP initialize + tools/call for a
fixture, reads stdout line-by-line, and stops as soon as the response arrives.
Measures real time-to-response (not sleep padding) and asserts expected markers
and budget. Writes a JSON result to benchmarks/results/.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, cast

BENCH_DIR = Path(__file__).resolve().parent
ROOT_DIR = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_fixture(fixture_path: Path) -> dict:
    with open(fixture_path) as f:
        fx = json.load(f)

    name = fx["name"]
    expect = fx["expect"]
    is_rejection = expect["status"] == "rejected"
    # Generous hard cap so runaway servers still get killed.
    hard_cap_sec = 20 if is_rejection else int(expect.get("max_duration_sec", 240) * 1.5 + 30)

    print(f"\u25b6 running fixture: {name}")

    messages = [
        {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "bench", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {
            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": fx["tool"], "arguments": fx["arguments"]},
        },
    ]
    stdin_payload = "\n".join(json.dumps(m) for m in messages) + "\n"

    start = time.time()
    proc = subprocess.Popen(
        ["bash", str(ROOT_DIR / "run.sh")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    stdin = cast(IO[str], proc.stdin)
    stdout = cast(IO[str], proc.stdout)

    try:
        stdin.write(stdin_payload)
        stdin.flush()
    except BrokenPipeError:
        pass

    response_text = None
    deadline = start + hard_cap_sec
    while time.time() < deadline:
        line = stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == 2 and "result" in msg:
            response_text = msg["result"]["content"][0]["text"]
            break

    duration_sec = round(time.time() - start, 2)

    try:
        stdin.close()
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    if response_text is None:
        result = {"fixture": name, "status": "fail", "failures": ["no response from server"], "duration_sec": duration_sec}
        print(f"  \u274c FAIL  no response in {duration_sec}s")
        _save(name, result)
        return result

    failures: list[str] = []

    if is_rejection:
        marker = expect["reject_marker"]
        if marker not in response_text:
            failures.append(f"expected rejection marker '{marker}' not found")
        if "Phase 0: Evidence" in response_text:
            failures.append("rejection fixture unexpectedly ran the full debate")
    else:
        for marker in expect.get("markers", []):
            if marker not in response_text:
                failures.append(f"expected marker '{marker}' not found")

    cost_usd = None
    cost_match = re.search(r"TOTAL:.*?\$([0-9.]+)", response_text)
    if cost_match:
        cost_usd = float(cost_match.group(1))

    max_cost = expect.get("max_cost_usd")
    if max_cost is not None and cost_usd is not None and cost_usd > max_cost:
        failures.append(f"cost ${cost_usd:.4f} exceeds budget ${max_cost}")

    max_duration = expect.get("max_duration_sec")
    if max_duration and duration_sec > max_duration:
        failures.append(f"duration {duration_sec:.1f}s exceeds budget {max_duration}s")

    optional_markers_present = [
        m for m in expect.get("optional_markers", []) if m in response_text
    ]

    result = {
        "fixture": name,
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "duration_sec": duration_sec,
        "cost_usd": cost_usd,
        "response_size_bytes": len(response_text),
        "optional_markers_present": optional_markers_present,
    }

    badge = "\u2705 PASS" if not failures else "\u274c FAIL"
    cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "$0.0000"
    print(f"  {badge}  duration={duration_sec:.1f}s  cost={cost_str}  size={len(response_text)}B")
    for f in failures:
        print(f"     - {f}")
    if optional_markers_present:
        print(f"     (optional markers present: {optional_markers_present})")

    _save(name, result)
    return result


def _save(name: str, result: dict) -> None:
    path = RESULTS_DIR / f"last_{name}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: run_fixture.py <fixture.json | --all | --rejections-only>", file=sys.stderr)
        return 2

    arg = sys.argv[1]
    if arg in ("--all", "--rejections-only"):
        fixtures = sorted((BENCH_DIR / "fixtures").glob("*.json"))
        if arg == "--rejections-only":
            fixtures = [f for f in fixtures if f.name.startswith("rejection-")]
        passed = 0
        for fx in fixtures:
            result = run_fixture(fx)
            if result["status"] == "pass":
                passed += 1
        print(f"\n=== {passed} / {len(fixtures)} fixtures passed ===")
        return 0 if passed == len(fixtures) else 1

    result = run_fixture(Path(arg))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
