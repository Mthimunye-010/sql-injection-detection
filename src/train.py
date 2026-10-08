"""Training and inference loop for the AGNCS variants.

Two details matter for the experiment's validity:

* **Family-balanced batches.**  The supervised-contrastive term needs several
  members of the same family inside a batch to have any positives at all.  A
  uniformly random batch of 128 drawn from 30 families would often give a
  family a single member, making `L_ret` degenerate.  `FamilyBatchSampler`
  instead draws `n_families_per_batch` families and `k` samples from each.

* **Model selection on validation only.**  The decision threshold and the early
  stopping epoch are both chosen on `val`; `test_id` and `test_ood` are touched
  exactly once, at the end.  Without this the FPR comparison would be
  optimistically biased.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .data import EncodedSplit
from .metrics import attention_metrics, detection_metrics
from .models import AGNCS, build_model, total_loss


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 128
    lr: float = 2.5e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.1
    families_per_batch: int = 16
    temperature: float = 0.1
    patience: int = 3
    seed: int = 0
    d_model: int = 128
    n_heads: int = 4
    dropout: float = 0.1
    emb_dim: int = 96
    label_smooth: float = 0.0
    dilations: Tuple[int, ...] = (1, 2, 4)
    verbose: bool = True


class FamilyBatchSampler:
    """Yields index batches containing several members of a few families."""

    def __init__(self, fam: np.ndarray, batch_size: int, families_per_batch: int,
                 seed: int = 0):
        self.batch_size = batch_size
        self.fpb = max(2, min(families_per_batch, len(np.unique(fam))))
        self.per = max(2, batch_size // self.fpb)
        self.rng = np.random.default_rng(seed)
        self.groups = {int(f): np.flatnonzero(fam == f) for f in np.unique(fam)}
        self.n = len(fam)

    def __len__(self) -> int:
        return max(1, self.n // self.batch_size)

    def __iter__(self):
        fams = list(self.groups)
        for _ in range(len(self)):
            self.rng.shuffle(fams)
            chosen = fams[:self.fpb]
            batch: List[int] = []
            for f in chosen:
                pool = self.groups[f]
                take = min(self.per, len(pool))
                batch.extend(self.rng.choice(pool, size=take, replace=False).tolist())
            if len(batch) > self.batch_size:
                batch = batch[:self.batch_size]
            yield np.array(batch, dtype=np.int64)


def _lr_at(step: int, total: int, base: float, warmup_frac: float) -> float:
    """Linear warm-up then cosine decay."""
    w = max(1, int(total * warmup_frac))
    if step < w:
        return base * step / w
    p = (step - w) / max(1, total - w)
    return base * 0.5 * (1.0 + np.cos(np.pi * min(1.0, p)))


@torch.no_grad()
def predict(model: AGNCS, split: EncodedSplit, batch_size: int = 512,
            want_alpha: bool = False, want_embed: bool = True,
            device: str = "cpu") -> Dict[str, np.ndarray]:
    model.eval()
    probs, embs, alphas = [], [], []
    for i in range(0, len(split), batch_size):
        xb = torch.from_numpy(split.x[i:i + batch_size].astype(np.int64)).to(device)
        out = model(xb)
        probs.append(torch.sigmoid(out["logit"]).cpu().numpy())
        if want_embed:
            embs.append(out["embed"].cpu().numpy())
        if want_alpha:
            alphas.append(out["alpha"].cpu().numpy().astype(np.float32))
    res = {"prob": np.concatenate(probs)}
    if want_embed:
        res["embed"] = np.concatenate(embs)
    if want_alpha:
        res["alpha"] = np.concatenate(alphas)
    return res


def train_variant(variant: str, train: EncodedSplit, val: EncodedSplit,
                  vocab_size: int, max_len: int, cfg: TrainConfig,
                  device: str = "cpu") -> Tuple[AGNCS, Dict[str, list], dict]:
    """Train one model variant; return (best model, history, info)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model, vcfg = build_model(variant, vocab_size=vocab_size, max_len=max_len,
                              d=cfg.d_model, n_heads=cfg.n_heads,
                              dropout=cfg.dropout, emb_dim=cfg.emb_dim,
                              dilations=cfg.dilations)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sampler = FamilyBatchSampler(train.fam, cfg.batch_size,
                                 cfg.families_per_batch, seed=cfg.seed)
    steps_per_epoch = len(sampler)
    total_steps = steps_per_epoch * cfg.epochs

    hist: Dict[str, list] = {"epoch": [], "train_loss": [], "cls": [], "att": [],
                             "att_kl": [], "ret": [], "val_f1": [], "val_fpr": [],
                             "val_auc": [], "val_att_lift": [], "sec": []}
    best = {"f1": -1.0, "state": None, "epoch": -1}
    bad = 0
    step = 0
    t_start = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"total": 0.0, "cls": 0.0, "att": 0.0, "att_kl": 0.0,
               "ret": 0.0, "n": 0}
        for idx in sampler:
            step += 1
            for g in opt.param_groups:
                g["lr"] = _lr_at(step, total_steps, cfg.lr, cfg.warmup_frac)

            xb = torch.from_numpy(train.x[idx].astype(np.int64)).to(device)
            yb = torch.from_numpy(train.y[idx].astype(np.float32)).to(device)
            mb = torch.from_numpy(train.mask[idx].astype(np.float32)).to(device)
            fb = torch.from_numpy(train.fam[idx].astype(np.int64)).to(device)

            out = model(xb)
            loss, parts = total_loss(out, yb, mb, fb,
                                     lambda_att=vcfg["lambda_att"],
                                     lambda_ret=vcfg["lambda_ret"],
                                     temperature=cfg.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            agg["n"] += 1
            for k in ("total", "cls", "att", "att_kl", "ret"):
                agg[k] += parts.get(k, 0.0)

        # ------------------------------------------------- validation
        pv = predict(model, val, want_alpha=(model.pooling_kind == "attention"),
                     device=device)
        dm = detection_metrics(val.y, pv["prob"], thr=0.5)
        if "alpha" in pv:
            am = attention_metrics(pv["alpha"], val.mask, val.valid)
            att_lift = am["att_lift"]
        else:
            att_lift = float("nan")

        n = max(1, agg["n"])
        hist["epoch"].append(ep)
        hist["train_loss"].append(agg["total"] / n)
        hist["cls"].append(agg["cls"] / n)
        hist["att"].append(agg["att"] / n)
        hist["att_kl"].append(agg["att_kl"] / n)
        hist["ret"].append(agg["ret"] / n)
        hist["val_f1"].append(dm["f1"])
        hist["val_fpr"].append(dm["fpr"])
        hist["val_auc"].append(dm["roc_auc"])
        hist["val_att_lift"].append(att_lift)
        hist["sec"].append(time.time() - t0)

        if cfg.verbose:
            print("    ep%-2d loss=%.4f (cls %.4f  attKL %.4f  ret %.4f) "
                  "val F1=%.4f FPR=%.4f AUC=%.4f lift=%.2f [%.0fs]"
                  % (ep, agg["total"] / n, agg["cls"] / n, agg["att_kl"] / n,
                     agg["ret"] / n, dm["f1"], dm["fpr"], dm["roc_auc"],
                     att_lift, hist["sec"][-1]))

        if dm["f1"] > best["f1"] + 1e-5:
            best = {"f1": dm["f1"],
                    "state": copy.deepcopy(model.state_dict()),
                    "epoch": ep}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if cfg.verbose:
                    print("    early stop at epoch %d (best epoch %d)" % (ep, best["epoch"]))
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    info = {"variant": variant, "best_epoch": best["epoch"],
            "best_val_f1": best["f1"], "n_params": model.n_params(),
            "train_seconds": time.time() - t_start,
            "lambda_att": vcfg["lambda_att"], "lambda_ret": vcfg["lambda_ret"],
            "pooling": vcfg["pooling"], "seed": cfg.seed}
    return model, hist, info
