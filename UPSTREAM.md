# Upstream: Kronos

The code in this repository is a snapshot of
[shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos), a foundation model
for financial candlestick (OHLCV) forecasting, released under the MIT License
(see [`LICENSE`](LICENSE), copyright (c) 2025 ShiYu).

| | |
|---|---|
| Source | https://github.com/shiyu-coder/Kronos.git |
| Commit | `67b630e67f6a18c9e9be918d9b4337c960db1e9a` (2026-04-13, "Merge pull request #243 …") |
| Paper | [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) |
| Weights | [huggingface.co/NeoQuasar](https://huggingface.co/NeoQuasar) (downloaded on first use, not stored here) |

All upstream files are unchanged apart from a short note at the top of
`README.md`. Local additions:

- `UPSTREAM.md`: this file.
- `kronos_mcp.py`, `.mcp.json`, `requirements-mcp.txt`: an MCP server that
  lets Claude call Kronos as a connector. See [`CLAUDE_CONNECTOR.md`](CLAUDE_CONNECTOR.md).
- `tests/test_offline_smoke.py`: runs the full predict path (`predict`,
  `predict_batch`, no-volume input, save/load round-trip) on tiny randomly
  initialised models, so it needs no network access or downloaded weights.
- `tests/test_mcp_server.py`: tests for the MCP server, also offline.
- `tests/conftest.py`: the tiny models both test files use.

## Setup

Python 3.10 to 3.12. `requirements.txt` pins `pandas==2.2.2`, which has no
wheels for Python 3.13.

```shell
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest
```

## Tests

Run from the repository root with `python -m pytest`, so `model` is importable:

```shell
python -m pytest tests/test_offline_smoke.py   # offline, about 10 seconds on CPU
python -m pytest tests/test_kronos_regression.py   # downloads Kronos-small from Hugging Face
```

The regression test pins exact Hugging Face revisions of `Kronos-small` and
`Kronos-Tokenizer-base` and checks forecasts against stored outputs, so it is
the one that proves the pretrained model works.

## Pulling upstream changes

```shell
git clone https://github.com/shiyu-coder/Kronos.git /tmp/kronos
git -C /tmp/kronos diff --binary 67b630e67f6a18c9e9be918d9b4337c960db1e9a HEAD > /tmp/kronos.patch
git apply --3way /tmp/kronos.patch
```

Then update the commit above.
