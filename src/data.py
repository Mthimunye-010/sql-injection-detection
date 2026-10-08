"""Encoding the corpus into padded tensors, with a disk cache.

Tokenising 60,000 functions takes ~30 s, which is fine once and annoying every
time the notebook is re-run, so the encoded arrays are cached as a single
`.npz`.  The cache key includes the tokeniser's configuration, so changing the
lexer invalidates it automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .code_tokenizer import (ID_TPL, MAX_IDS, PAD, SPECIALS, Vocab,
                             parse_rationale, rationale_mask, tokenise)
from .sqli_corpus import ALL_FAMILIES

FAMILY_TO_ID = {f: i for i, f in enumerate(ALL_FAMILIES)}


@dataclass
class EncodedSplit:
    x: np.ndarray            # (N, L) int32 token ids
    mask: np.ndarray         # (N, L) uint8 taint/evidence annotation
    valid: np.ndarray        # (N, L) bool  non-PAD positions
    y: np.ndarray            # (N,)   int8  label
    fam: np.ndarray          # (N,)   int16 family id
    hard_neg: np.ndarray     # (N,)   int8
    index: np.ndarray        # (N,)   int32 row index into the source dataframe

    def __len__(self) -> int:
        return len(self.y)


def truncate_head_tail(seq: List, max_len: int) -> List:
    """Keep the first and last halves of an over-long sequence.

    Plain right-truncation would cut off the `execute()` sink, which is the one
    token the decision depends on.  Head+tail truncation keeps both the
    function prologue (where the taint enters) and the epilogue (where it is
    used).  Only ~1% of samples exceed the configured length at all.
    """
    if len(seq) <= max_len:
        return seq
    head = max_len // 2
    tail = max_len - head
    return list(seq[:head]) + list(seq[-tail:])


def tokenise_frame(df: pd.DataFrame) -> Tuple[List[List[str]], List[List[int]]]:
    toks, masks = [], []
    for code, rat in zip(df["code"].values, df["rationale_lines"].values):
        t, lines = tokenise(code)
        m = rationale_mask(lines, parse_rationale(rat))
        toks.append(t)
        masks.append(m)
    return toks, masks


def encode(df: pd.DataFrame, vocab: Vocab, max_len: int) -> EncodedSplit:
    n = len(df)
    x = np.full((n, max_len), vocab.pad_id, dtype=np.int32)
    mk = np.zeros((n, max_len), dtype=np.uint8)
    va = np.zeros((n, max_len), dtype=bool)
    toks, masks = tokenise_frame(df)
    for i, (t, m) in enumerate(zip(toks, masks)):
        t = truncate_head_tail(t, max_len)
        m = truncate_head_tail(m, max_len)
        ids = vocab.encode(t)
        L = len(ids)
        x[i, :L] = ids
        mk[i, :L] = m
        va[i, :L] = True
    return EncodedSplit(
        x=x, mask=mk, valid=va,
        y=df["label"].values.astype(np.int8),
        fam=np.array([FAMILY_TO_ID[f] for f in df["family"].values], dtype=np.int16),
        hard_neg=df["hard_negative"].values.astype(np.int8),
        index=np.arange(n, dtype=np.int32),
    )


def token_strings(df: pd.DataFrame) -> List[str]:
    """Space-joined token stream, for the TF-IDF baseline.

    The classical baseline sees *exactly* the same preprocessing as the neural
    models, so the comparison isolates the model rather than the features.
    """
    return [" ".join(tokenise(c)[0]) for c in df["code"].values]


def _cache_key(df: pd.DataFrame, max_len: int, min_freq: int, max_vocab: int) -> str:
    h = hashlib.sha1()
    h.update(("|".join(str(u) for u in df["uid"].values[:2000])).encode())
    h.update(json.dumps({"n": len(df), "L": max_len, "mf": min_freq,
                         "mv": max_vocab, "specials": len(SPECIALS),
                         "max_ids": MAX_IDS, "v": 3}).encode())
    return h.hexdigest()[:16]


def prepare(df: pd.DataFrame, max_len: int = 192, min_freq: int = 5,
            max_vocab: int = 20000, cache_dir: str = "data/cache",
            verbose: bool = True) -> Tuple[Dict[str, EncodedSplit], Vocab,
                                           Dict[str, pd.DataFrame]]:
    """Build the vocabulary on `train` only, then encode every split."""
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(df, max_len, min_freq, max_vocab)
    path = os.path.join(cache_dir, "enc_%s.npz" % key)
    vpath = os.path.join(cache_dir, "vocab_%s.json" % key)

    frames = {sp: df[df["split"] == sp].reset_index(drop=True)
              for sp in ("train", "val", "test_id", "test_ood")}

    if os.path.exists(path) and os.path.exists(vpath):
        if verbose:
            print("loading encoded cache %s" % os.path.basename(path))
        with open(vpath, encoding="utf-8") as fh:
            vocab = Vocab.from_dict(json.load(fh))
        z = np.load(path)
        splits = {
            sp: EncodedSplit(
                x=z["%s_x" % sp], mask=z["%s_m" % sp], valid=z["%s_v" % sp],
                y=z["%s_y" % sp], fam=z["%s_f" % sp], hard_neg=z["%s_h" % sp],
                index=z["%s_i" % sp])
            for sp in frames}
        return splits, vocab, frames

    if verbose:
        print("tokenising %d training samples to build the vocabulary ..." % len(frames["train"]))
    train_toks, _ = tokenise_frame(frames["train"])
    vocab = Vocab.build(train_toks, min_freq=min_freq, max_size=max_vocab)
    if verbose:
        print("  vocabulary size: %d" % len(vocab))

    splits: Dict[str, EncodedSplit] = {}
    for sp, f in frames.items():
        if verbose:
            print("  encoding %-8s (%d samples)" % (sp, len(f)))
        splits[sp] = encode(f, vocab, max_len)

    np.savez_compressed(
        path,
        **{("%s_x" % sp): s.x for sp, s in splits.items()},
        **{("%s_m" % sp): s.mask for sp, s in splits.items()},
        **{("%s_v" % sp): s.valid for sp, s in splits.items()},
        **{("%s_y" % sp): s.y for sp, s in splits.items()},
        **{("%s_f" % sp): s.fam for sp, s in splits.items()},
        **{("%s_h" % sp): s.hard_neg for sp, s in splits.items()},
        **{("%s_i" % sp): s.index for sp, s in splits.items()},
    )
    with open(vpath, "w", encoding="utf-8") as fh:
        json.dump(vocab.to_dict(), fh)
    return splits, vocab, frames


def subsample(s: EncodedSplit, n: int, seed: int = 0,
              stratify_by: Optional[np.ndarray] = None) -> EncodedSplit:
    """Stratified subsample, used to keep the notebook's runtime manageable."""
    if n >= len(s):
        return s
    rng = np.random.default_rng(seed)
    if stratify_by is None:
        stratify_by = s.fam
    keep: List[int] = []
    groups = np.unique(stratify_by)
    per = max(1, n // len(groups))
    for g in groups:
        idx = np.flatnonzero(stratify_by == g)
        rng.shuffle(idx)
        keep.extend(idx[:per].tolist())
    keep_arr = np.array(sorted(keep)[:n], dtype=int)
    return EncodedSplit(
        x=s.x[keep_arr], mask=s.mask[keep_arr], valid=s.valid[keep_arr],
        y=s.y[keep_arr], fam=s.fam[keep_arr], hard_neg=s.hard_neg[keep_arr],
        index=s.index[keep_arr])
