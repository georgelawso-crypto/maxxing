"""Tiny randomly initialised Kronos models for tests that must run without network access."""
import pytest
import torch

from model import Kronos, KronosTokenizer

TINY_TOKENIZER_CONFIG = dict(
    d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2, n_dec_layers=2,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
    s1_bits=4, s2_bits=4, beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4,
)
TINY_MODEL_CONFIG = dict(
    s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=2, ff_dim=64,
    ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True,
)


@pytest.fixture
def tiny_models():
    """A fresh (tokenizer, model) pair with fixed random weights."""
    torch.manual_seed(0)
    return KronosTokenizer(**TINY_TOKENIZER_CONFIG).eval(), Kronos(**TINY_MODEL_CONFIG).eval()


@pytest.fixture(scope="session")
def tiny_model_dirs(tmp_path_factory):
    """(tokenizer_dir, model_dir) saved with save_pretrained, loadable via from_pretrained."""
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("tiny_kronos")
    KronosTokenizer(**TINY_TOKENIZER_CONFIG).save_pretrained(root / "tokenizer")
    Kronos(**TINY_MODEL_CONFIG).save_pretrained(root / "model")
    return root / "tokenizer", root / "model"
