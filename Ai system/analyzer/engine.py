"""Inference wrapper for the notebook's trained AGNCS checkpoint."""
import json
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from src.code_tokenizer import Vocab, tokenise
from src.models import AGNCS

BASE = Path(__file__).resolve().parent.parent
MAX_LEN = 160

@lru_cache(maxsize=1)
def load_engine():
    torch.set_num_threads(min(4, torch.get_num_threads()))
    vocab_data = json.loads((BASE / "artifacts/vocab.json").read_text(encoding="utf-8"))
    vocab = Vocab(vocab_data["itos"])
    model = AGNCS(vocab_size=len(vocab), d=96, max_len=MAX_LEN, n_heads=4,
                  dropout=0.1, pad_id=vocab.pad_id, pooling="attention", emb_dim=96,
                  dilations=(1, 2, 4))
    weights = torch.load(BASE / "artifacts/agncs_seed0.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    corpus = json.loads((BASE / "artifacts/search_corpus.json").read_text(encoding="utf-8"))
    return model, vocab, corpus

def encode(code, vocab):
    tokens, token_lines = tokenise(code)
    if len(tokens) > MAX_LEN:
        half = MAX_LEN // 2
        tokens = tokens[:half] + tokens[-(MAX_LEN-half):]
        token_lines = token_lines[:half] + token_lines[-(MAX_LEN-half):]
    ids = vocab.encode(tokens)
    x = torch.full((1, MAX_LEN), vocab.pad_id, dtype=torch.long)
    x[0, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x, tokens, token_lines

@lru_cache(maxsize=1)
def indexed_embeddings():
    model, vocab, corpus = load_engine()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(corpus), 32):
            batch = []
            for row in corpus[start:start+32]:
                x, _, _ = encode(row["code"], vocab)
                batch.append(x)
            pred = model(torch.cat(batch, dim=0))
            vectors.append(pred["embed"])
    return torch.cat(vectors, dim=0), corpus

def analyze_code(code):
    model, vocab, _ = load_engine()
    x, tokens, token_lines = encode(code, vocab)
    with torch.inference_mode():
        out = model(x)
    probability = float(torch.sigmoid(out["logit"])[0])
    alpha = out["alpha"][0, :len(tokens)]
    line_scores = {}
    for line, weight in zip(token_lines, alpha.tolist()):
        line_scores[line] = line_scores.get(line, 0.0) + weight
    denominator = max(sum(line_scores.values()), 1e-12)
    line_scores = {k: v / denominator for k, v in line_scores.items()}
    lines = code.splitlines()
    ranked_lines = [{"number": n, "text": lines[n-1], "score": round(score, 4)}
                    for n, score in sorted(line_scores.items(), key=lambda p: p[1], reverse=True)[:8]
                    if 1 <= n <= len(lines)]

    index, corpus = indexed_embeddings()
    similarities = index @ out["embed"][0]
    k = min(5, len(corpus))
    values, indices = torch.topk(similarities, k=k)
    hits = []
    for score, ix in zip(values.tolist(), indices.tolist()):
        row = corpus[ix]
        hits.append({"score": round(score, 4), "family": row["family"],
                     "label": "Vulnerable" if row["label"] else "Safe",
                     "hard_negative": row["hard_negative"], "code": row["code"]})
    return {"probability": probability, "verdict": "Likely vulnerable" if probability >= 0.5 else "Likely safe",
            "lines": ranked_lines, "hits": hits}
