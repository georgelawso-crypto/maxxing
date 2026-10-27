"""Tests for the Kronos MCP server (kronos_mcp.py), using tiny models so no download is needed."""
import asyncio
import json
import math
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

import kronos_mcp
from model import KronosPredictor

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DATA_PATH = Path(__file__).parent / "data" / "regression_input.csv"
MAX_CONTEXT = 64


def candles(n, start=0, volume=True):
    df = pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"]).iloc[start:start + n]
    rows = []
    for row in df.to_dict("records"):
        candle = {"timestamp": row["timestamps"].isoformat(), **{k: row[k] for k in ("open", "high", "low", "close")}}
        if volume:
            candle.update(volume=row["volume"], amount=row["amount"])
        rows.append(candle)
    return rows


def call(server, arguments):
    async def run():
        async with Client(server) as client:
            return await client.call_tool("forecast_candles", arguments)
    return asyncio.run(run())


@pytest.fixture
def tiny_server(tiny_models, monkeypatch):
    tokenizer, model = tiny_models
    monkeypatch.setattr(kronos_mcp, "_predictor", KronosPredictor(model, tokenizer, device="cpu", max_context=MAX_CONTEXT))
    return kronos_mcp.server


def test_tool_is_listed_with_schemas(tiny_server):
    async def run():
        async with Client(tiny_server) as client:
            return await client.list_tools()
    tools = {tool.name: tool for tool in asyncio.run(run()).tools}
    tool = tools["forecast_candles"]
    assert {"candles", "pred_len"} <= set(tool.input_schema["required"])
    assert "forecast" in tool.output_schema["properties"]
    assert tool.annotations.read_only_hint


def test_forecast_infers_timestamps_and_caps_context(tiny_server):
    result = call(tiny_server, {"candles": candles(100), "pred_len": 6, "sample_count": 2})
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["context_candles_used"] == MAX_CONTEXT
    assert out["timestamps_inferred"] is True
    assert len(out["forecast"]) == 6

    last = datetime.fromisoformat(candles(100)[-1]["timestamp"])
    assert [datetime.fromisoformat(c["timestamp"]) for c in out["forecast"]] == [last + timedelta(minutes=5 * (i + 1)) for i in range(6)]
    for c in out["forecast"]:
        assert all(math.isfinite(c[k]) for k in ("open", "high", "low", "close", "volume", "amount"))
    assert out["final_forecast_close"] == out["forecast"][-1]["close"]
    assert out["close_change_pct"] == pytest.approx((out["final_forecast_close"] / out["last_close"] - 1) * 100)


def test_forecast_without_volume_returns_no_volume(tiny_server):
    result = call(tiny_server, {"candles": candles(40, volume=False), "pred_len": 3})
    assert not result.is_error, result.content
    assert all(c["volume"] is None and c["amount"] is None for c in result.structured_content["forecast"])


def test_forecast_uses_given_future_timestamps_and_ignores_utc_offsets(tiny_server):
    # Offsets change across a DST switch; only the wall-clock time is used.
    history = [dict(c, timestamp=c["timestamp"] + ("-04:00" if i < 20 else "-05:00")) for i, c in enumerate(candles(40))]
    future = ["2030-01-02T09:30:00-05:00", "2030-01-02T09:35:00-05:00"]
    result = call(tiny_server, {"candles": history, "pred_len": 2, "future_timestamps": future})
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["timestamps_inferred"] is False
    assert [c["timestamp"] for c in out["forecast"]] == ["2030-01-02T09:30:00", "2030-01-02T09:35:00"]


@pytest.mark.parametrize("arguments, message", [
    ({"candles": candles(10)[::-1], "pred_len": 2}, "strictly increasing"),
    ({"candles": candles(9) + candles(1), "pred_len": 2}, "strictly increasing"),
    ({"candles": candles(10), "pred_len": 2, "future_timestamps": ["2030-01-01T00:00:00"]}, "expected pred_len=2"),
    ({"candles": candles(10), "pred_len": 1, "future_timestamps": ["2000-01-01T00:00:00"]}, "after the last candle"),
    ({"candles": candles(5) + candles(5, start=5, volume=False), "pred_len": 2}, "every candle or for none"),
    ({"candles": [{k: v for k, v in c.items() if k != "volume"} for c in candles(10)], "pred_len": 2}, "'amount' requires 'volume'"),
    ({"candles": candles(10), "pred_len": 0}, "pred_len"),
    ({"candles": candles(1), "pred_len": 2}, "candles"),
])
def test_invalid_input_is_a_tool_error(tiny_server, arguments, message):
    result = call(tiny_server, arguments)
    assert result.is_error
    assert message in result.content[0].text


def test_model_load_failure_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(kronos_mcp, "_predictor", None)
    monkeypatch.setattr(kronos_mcp, "TOKENIZER_ID", str(tmp_path / "missing"))
    result = call(kronos_mcp.server, {"candles": candles(10), "pred_len": 2})
    assert result.is_error
    assert "Could not load Kronos" in result.content[0].text


def server_env(tiny_model_dirs):
    tokenizer_dir, model_dir = tiny_model_dirs
    return {"KRONOS_TOKENIZER": str(tokenizer_dir), "KRONOS_MODEL": str(model_dir),
            "KRONOS_MAX_CONTEXT": str(MAX_CONTEXT), "KRONOS_DEVICE": "cpu", "HF_HUB_OFFLINE": "1",
            "PATH": os.environ.get("PATH", "")}


def test_stdio_end_to_end(tiny_model_dirs):
    _, model_dir = tiny_model_dirs
    params = StdioServerParameters(
        command=sys.executable,
        args=[str(REPO_ROOT / "kronos_mcp.py")],
        cwd=str(REPO_ROOT),
        env=server_env(tiny_model_dirs),
    )
    result = call(params, {"candles": candles(80), "pred_len": 4})
    assert not result.is_error, result.content
    out = result.structured_content
    assert out["model"] == str(model_dir)
    assert out["context_candles_used"] == MAX_CONTEXT
    assert len(out["forecast"]) == 4


def test_http_requires_bearer_token(tiny_model_dirs):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(server_env(tiny_model_dirs), KRONOS_MCP_TOKEN="s3cret", NO_PROXY="127.0.0.1")
    proc = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "kronos_mcp.py"), "--transport", "streamable-http", "--port", str(port)],
        cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    initialize = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}}).encode()

    def post(authorization=None):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if authorization:
            headers["Authorization"] = authorization
        request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", data=initialize, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=10) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code

    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                assert proc.poll() is None and time.monotonic() < deadline, "server did not start"
                time.sleep(0.2)
        assert post() == 401
        assert post("Bearer wrong") == 401
        assert post("Bearer s3cret") == 200
    finally:
        proc.terminate()
        proc.wait(timeout=10)
