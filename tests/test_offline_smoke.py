"""Offline smoke tests for the Kronos inference pipeline.

These build tiny, randomly initialised models instead of downloading the
pretrained weights from Hugging Face, so they run without network access.
They check that the code path (normalisation, tokenisation, autoregressive
decoding, de-normalisation, save/load) works end to end. They say nothing
about forecast quality; test_kronos_regression.py covers the real weights.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from model import Kronos, KronosPredictor, KronosTokenizer

INPUT_DATA_PATH = Path(__file__).parent / "data" / "regression_input.csv"
FEATURE_NAMES = ["open", "high", "low", "close", "volume", "amount"]
CONTEXT_LEN = 64
PRED_LEN = 8
SEED = 0

TOKENIZER_CONFIG = dict(
    d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
)
MODEL_CONFIG = dict(
    s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True,
)


@pytest.fixture(scope="module")
def tiny_predictor():
    torch.manual_seed(SEED)
    tokenizer = KronosTokenizer(**TOKENIZER_CONFIG).eval()
    model = Kronos(**MODEL_CONFIG).eval()
    # max_context == CONTEXT_LEN so decoding also exercises the rolling context-window branch.
    return KronosPredictor(model, tokenizer, device="cpu", max_context=CONTEXT_LEN)


@pytest.fixture(scope="module")
def kline_df():
    return pd.read_csv(INPUT_DATA_PATH, parse_dates=["timestamps"])


def _window(df, start):
    context = df.iloc[start:start + CONTEXT_LEN]
    future = df.iloc[start + CONTEXT_LEN:start + CONTEXT_LEN + PRED_LEN]
    return (
        context[FEATURE_NAMES].reset_index(drop=True),
        context["timestamps"].reset_index(drop=True),
        future["timestamps"].reset_index(drop=True),
    )


def _assert_valid_forecast(pred_df, y_timestamp):
    assert list(pred_df.columns) == FEATURE_NAMES
    assert pred_df.shape == (PRED_LEN, len(FEATURE_NAMES))
    assert (pred_df.index == y_timestamp).all()
    assert np.isfinite(pred_df.to_numpy()).all()


def test_predict(tiny_predictor, kline_df):
    x_df, x_ts, y_ts = _window(kline_df, 0)
    with torch.no_grad():
        pred_df = tiny_predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN, sample_count=2, verbose=False)
    _assert_valid_forecast(pred_df, y_ts)


def test_predict_without_volume(tiny_predictor, kline_df):
    x_df, x_ts, y_ts = _window(kline_df, 0)
    with torch.no_grad():
        pred_df = tiny_predictor.predict(x_df[["open", "high", "low", "close"]], x_ts, y_ts,
                                         pred_len=PRED_LEN, verbose=False)
    _assert_valid_forecast(pred_df, y_ts)


def test_predict_batch(tiny_predictor, kline_df):
    windows = [_window(kline_df, start) for start in (0, 500, 1000)]
    with torch.no_grad():
        preds = tiny_predictor.predict_batch(
            df_list=[w[0] for w in windows],
            x_timestamp_list=[w[1] for w in windows],
            y_timestamp_list=[w[2] for w in windows],
            pred_len=PRED_LEN,
            verbose=False,
        )
    assert len(preds) == len(windows)
    for pred_df, (_, _, y_ts) in zip(preds, windows):
        _assert_valid_forecast(pred_df, y_ts)


def test_greedy_decoding_is_deterministic(tiny_predictor, kline_df):
    x_df, x_ts, y_ts = _window(kline_df, 0)
    with torch.no_grad():
        first = tiny_predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN, top_k=1, top_p=1.0, verbose=False)
        second = tiny_predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN, top_k=1, top_p=1.0, verbose=False)
    pd.testing.assert_frame_equal(first, second)


def test_save_and_load_pretrained_roundtrip(tmp_path):
    # Same loading path as from_pretrained("NeoQuasar/..."), but from a local directory.
    torch.manual_seed(SEED)
    tokenizer = KronosTokenizer(**TOKENIZER_CONFIG).eval()
    model = Kronos(**MODEL_CONFIG).eval()
    tokenizer.save_pretrained(tmp_path / "tokenizer")
    model.save_pretrained(tmp_path / "model")

    loaded_tokenizer = KronosTokenizer.from_pretrained(tmp_path / "tokenizer")
    loaded_model = Kronos.from_pretrained(tmp_path / "model")

    for original, loaded in ((tokenizer, loaded_tokenizer), (model, loaded_model)):
        original_state, loaded_state = original.state_dict(), loaded.state_dict()
        assert original_state.keys() == loaded_state.keys()
        for key in original_state:
            torch.testing.assert_close(original_state[key], loaded_state[key])
