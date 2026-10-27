"""MCP server that lets Claude call Kronos to forecast OHLCV candles.

Local (Claude Code, Claude Desktop), over stdio:
    python kronos_mcp.py

Remote (claude.ai custom connector), over streamable HTTP at /mcp:
    python kronos_mcp.py --transport streamable-http --host 0.0.0.0 --port 8000

Environment variables:
    KRONOS_MODEL        Hugging Face id or local directory (default NeoQuasar/Kronos-small)
    KRONOS_TOKENIZER    Hugging Face id or local directory (default NeoQuasar/Kronos-Tokenizer-base)
    KRONOS_MAX_CONTEXT  Context length in candles (default 512; 2048 for Kronos-mini)
    KRONOS_DEVICE       cpu, cuda:0, mps, ... (auto-detected when unset)
    KRONOS_MCP_TOKEN    streamable-http only: require "Authorization: Bearer <token>" on every request
"""
import argparse
import hmac
import os
import sys
import threading
from datetime import datetime
from typing import Annotated

import pandas as pd
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from model import Kronos, KronosPredictor, KronosTokenizer

MODEL_ID = os.environ.get("KRONOS_MODEL", "NeoQuasar/Kronos-small")
TOKENIZER_ID = os.environ.get("KRONOS_TOKENIZER", "NeoQuasar/Kronos-Tokenizer-base")
MAX_CONTEXT = int(os.environ.get("KRONOS_MAX_CONTEXT", "512"))
DEVICE = os.environ.get("KRONOS_DEVICE") or None
MAX_PRED_LEN = 512
MAX_SAMPLE_COUNT = 20

PRICE_COLS = ["open", "high", "low", "close"]

server = MCPServer(
    name="kronos",
    title="Kronos",
    instructions=(
        "Kronos is a foundation model for financial candlesticks (OHLCV). Fetch historical candles "
        "from a market-data source, pass them to forecast_candles, and present the result as a "
        "probabilistic model forecast, not as a prediction to trade on."
    ),
)

_lock = threading.Lock()
_predictor = None


def to_timestamps(values, name):
    # Kronos's time features (minute, hour, weekday, day, month) are wall-clock values, so UTC
    # offsets are dropped rather than converted; this also tolerates offsets that change across DST.
    series = pd.Series(pd.to_datetime([v.replace(tzinfo=None) for v in values]))
    if not series.is_monotonic_increasing or series.duplicated().any():
        raise ToolError(f"{name}: timestamps must be strictly increasing (oldest first).")
    return series


def get_predictor():
    global _predictor
    with _lock:
        if _predictor is None:
            try:
                tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID).eval()
                model = Kronos.from_pretrained(MODEL_ID).eval()
            except Exception as exc:
                # Surface the cause (usually a failed download); the SDK hides unexpected errors.
                raise ToolError(f"Could not load Kronos ({MODEL_ID}, {TOKENIZER_ID}): {exc}") from exc
            _predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=MAX_CONTEXT)
        return _predictor


class Candle(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    timestamp: datetime = Field(description="Bar open time in the market's local time; any UTC offset is ignored.")
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    amount: float | None = Field(default=None, description="Traded value (turnover). Derived from volume when omitted.")


class Forecast(BaseModel):
    model: str
    context_candles_used: int = Field(description="Most recent input candles the model saw (capped at its context length).")
    timestamps_inferred: bool = Field(description="True when future timestamps were extrapolated from the median input spacing.")
    sample_count: int
    last_close: float
    final_forecast_close: float
    close_change_pct: float = Field(description="Change from last_close to final_forecast_close, in percent.")
    forecast: list[Candle]


@server.tool(annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
def forecast_candles(
    candles: Annotated[list[Candle], Field(min_length=2, description="Historical candles, oldest first, evenly spaced. Up to the model's context length are used (512 for Kronos-small); 100+ works best.")],
    pred_len: Annotated[int, Field(ge=1, le=MAX_PRED_LEN, description="Number of future candles to forecast.")],
    future_timestamps: Annotated[list[datetime] | None, Field(description="Timestamps for the forecast candles, length pred_len. Pass these when the market has gaps (overnight, weekends); otherwise they are extrapolated from the input spacing.")] = None,
    temperature: Annotated[float, Field(gt=0, le=2, description="Sampling temperature.")] = 1.0,
    top_p: Annotated[float, Field(gt=0, le=1, description="Nucleus sampling threshold.")] = 0.9,
    sample_count: Annotated[int, Field(ge=1, le=MAX_SAMPLE_COUNT, description="Sampled paths to average; more is smoother and slower.")] = 1,
) -> Forecast:
    """Forecast the next OHLCV candles of a price series with the Kronos foundation model.

    Works for any instrument and interval (stocks, crypto, FX; minutes to days). The output is the
    mean of `sample_count` sampled paths from a probabilistic model, so repeated calls differ.
    """
    df = pd.DataFrame([c.model_dump(exclude={"timestamp"}) for c in candles])
    timestamps = to_timestamps([c.timestamp for c in candles], "candles")

    for col in ("volume", "amount"):
        present = df[col].notna()
        if present.any() and not present.all():
            raise ToolError(f"'{col}' must be given for every candle or for none.")
        if not present.any():
            df = df.drop(columns=col)
    has_volume = "volume" in df.columns
    if "amount" in df.columns and not has_volume:
        raise ToolError("'amount' requires 'volume'.")

    if future_timestamps is not None:
        if len(future_timestamps) != pred_len:
            raise ToolError(f"future_timestamps has {len(future_timestamps)} entries; expected pred_len={pred_len}.")
        y_timestamp = to_timestamps(future_timestamps, "future_timestamps")
        if y_timestamp.iloc[0] <= timestamps.iloc[-1]:
            raise ToolError("future_timestamps must start after the last candle.")
    else:
        step = timestamps.diff().median()
        y_timestamp = pd.Series([timestamps.iloc[-1] + step * (i + 1) for i in range(pred_len)])

    predictor = get_predictor()
    df, timestamps = df.iloc[-predictor.max_context:], timestamps.iloc[-predictor.max_context:]
    with _lock:
        pred_df = predictor.predict(
            df=df.reset_index(drop=True),
            x_timestamp=timestamps.reset_index(drop=True),
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count,
            verbose=False,
        )

    forecast = [
        Candle(
            timestamp=ts,
            **{col: float(row[col]) for col in PRICE_COLS},
            # Without input volume the model only sees zeros, so its volume output means nothing.
            volume=float(row["volume"]) if has_volume else None,
            amount=float(row["amount"]) if has_volume else None,
        )
        for ts, row in zip(y_timestamp, pred_df.to_dict("records"))
    ]
    last_close = float(df["close"].iloc[-1])
    final_close = forecast[-1].close
    return Forecast(
        model=MODEL_ID,
        context_candles_used=len(df),
        timestamps_inferred=future_timestamps is None,
        sample_count=sample_count,
        last_close=last_close,
        final_forecast_close=final_close,
        close_change_pct=(final_close / last_close - 1) * 100,
        forecast=forecast,
    )


def require_bearer_token(app, token):
    expected = f"Bearer {token}".encode()

    async def guarded(scope, receive, send):
        if scope["type"] == "http" and not hmac.compare_digest(dict(scope["headers"]).get(b"authorization", b""), expected):
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain")]})
            await send({"type": "http.response.body", "body": b"Unauthorized"})
            return
        await app(scope, receive, send)

    return guarded


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", help="streamable-http only")
    parser.add_argument("--port", type=int, default=8000, help="streamable-http only")
    args = parser.parse_args()

    if args.transport == "stdio":
        server.run("stdio")
        return

    app = server.streamable_http_app(host=args.host)
    token = os.environ.get("KRONOS_MCP_TOKEN")
    if token:
        app = require_bearer_token(app, token)
    elif args.host not in ("127.0.0.1", "localhost", "::1"):
        print("Warning: KRONOS_MCP_TOKEN is not set, so anyone who can reach this server can use it.", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
