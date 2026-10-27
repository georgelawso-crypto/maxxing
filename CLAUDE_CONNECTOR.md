# Kronos as a Claude connector

`kronos_mcp.py` is an [MCP](https://modelcontextprotocol.io) server with one tool,
`forecast_candles`: give it historical OHLCV candles and it returns the next
`pred_len` candles forecast by Kronos. Pair it with a market-data connector
(Financial Modeling Prep, Alpha Vantage, TradingView, ...) so Claude can fetch
the candles and then forecast them.

## Setup

Python 3.10 to 3.12.

```shell
python -m venv .venv
.venv/bin/pip install -r requirements-mcp.txt
```

The first forecast downloads `Kronos-small` and its tokenizer (about 100 MB)
from Hugging Face; later calls reuse the local cache.

## Claude Code

This repository ships a `.mcp.json`, so the server is available in any Claude
Code session started in the repo. Approve `kronos` when Claude Code asks (or in
`/mcp`). It runs `.venv/bin/python`, so create the venv first.

To use it from other projects as well:

```shell
claude mcp add -s user kronos -- /path/to/maxxing/.venv/bin/python /path/to/maxxing/kronos_mcp.py
```

## Claude Desktop

Settings > Developer > Edit Config, then add to `claude_desktop_config.json`
(absolute paths) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "kronos": {
      "command": "/path/to/maxxing/.venv/bin/python",
      "args": ["/path/to/maxxing/kronos_mcp.py"]
    }
  }
}
```

## claude.ai (custom connector)

claude.ai connects from Anthropic's cloud, not from your computer, so the server
has to be reachable on the public internet over HTTPS.

1. Generate a token: `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
2. On a host with at least 2 GB of RAM, run:

   ```shell
   KRONOS_MCP_TOKEN=<token> .venv/bin/python kronos_mcp.py --transport streamable-http --host 0.0.0.0 --port 8000
   ```

   Put HTTPS in front of it (most hosting platforms do this for you). The
   connector URL is `https://<your-host>/mcp`.
3. In claude.ai: Customize > Connectors > **+ Add** > **Add custom connector**.
   Name it Kronos, paste the URL, choose **No sign in**, and under
   **Request headers** add `Authorization` with the value `Bearer <token>`.
4. Turn the connector on for a chat from the tools menu in the message box.

Without `KRONOS_MCP_TOKEN` the HTTP server accepts anyone who can reach it.

## Configuration

| Variable | Default | |
|---|---|---|
| `KRONOS_MODEL` | `NeoQuasar/Kronos-small` | Hugging Face id or local directory |
| `KRONOS_TOKENIZER` | `NeoQuasar/Kronos-Tokenizer-base` | Must match the model (`Kronos-Tokenizer-2k` for `Kronos-mini`) |
| `KRONOS_MAX_CONTEXT` | `512` | Candles of history used; `2048` for `Kronos-mini` |
| `KRONOS_DEVICE` | auto | `cpu`, `cuda:0`, `mps`, ... |
| `KRONOS_MCP_TOKEN` | unset | HTTP only: required `Authorization: Bearer` token |

## The `forecast_candles` tool

| Input | |
|---|---|
| `candles` | List of `{timestamp, open, high, low, close, volume?, amount?}`, oldest first, evenly spaced. Only the last `KRONOS_MAX_CONTEXT` are used. Give `volume` for all candles or none. |
| `pred_len` | Candles to forecast (1 to 512). |
| `future_timestamps` | Optional timestamps for the forecast. Pass them when the market has gaps (nights, weekends); otherwise they continue the median input spacing. |
| `temperature`, `top_p` | Sampling settings (defaults 1.0 and 0.9). |
| `sample_count` | Sampled paths to average (1 to 20). |

It returns the forecast candles plus `last_close`, `final_forecast_close` and
`close_change_pct`. Volume is only forecast when the input has volume.
Timestamps are read as the market's local wall-clock time, and any UTC offset
is dropped, because that is what the model's time features use.

Forecasts are sampled from a probabilistic model, so repeated calls differ and
none of them is trading advice. The server runs one forecast at a time.

## Tests

```shell
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/test_mcp_server.py
```

These use tiny random models, so they need no download. They cover the tool
in-process, a real stdio session, and the HTTP bearer-token check.
