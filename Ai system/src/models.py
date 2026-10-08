"""AGNCS -- Attention-Guided Neural Code Search for SQL-injection detection.

Architecture
------------
                        token ids  (B, L)
                             |
                  Embedding + learned position          (B, L, d)
                             |
                  Dilated residual CNN encoder          (B, L, d)
                  dilations 1,2,4 -> receptive field 15 tokens
                  (chosen empirically -- see below)
                             |
             +---------------+----------------+
             |                                |
   Multi-head additive attention        (attention weights alpha)
   pooling  ->  code vector c                 |
             |                                |
     +-------+--------+                       |
     |                |                       |
 Detection head   Search head            guided by the taint mask
 P(vulnerable)    L2-normalised                via L_att
                  embedding z

Why a dilated CNN rather than a Transformer encoder?
----------------------------------------------------
Measured on the target CPU, a 2-layer Transformer encoder runs at 96
samples/s against 376 samples/s for this encoder at equal width -- a 3.9x
difference that decides whether a three-seed, seven-model study is feasible at
all.  The *global* token-to-token interaction that self-attention would add is
supplied instead by the attention-pooling layer, which is where this project
needs attention to be anyway: it is the quantity we supervise.

On the receptive field: measuring the token distance from the first to the last
annotated statement gives a median of 37 tokens in-distribution and 58 in the
held-out families, which suggested the field should be widened to cover the span
(R = 1 + 2*sum(d_i) = 63 or 127).  `scripts/ablate_receptive_field.py` tested
that and **did not support it**: across R in {15, 31, 63, 127} the held-out F1
spread was 0.121 while the in-distribution spread was 0.0013, with a *wider*
field correlating with worse OOD FPR (Spearman rho = +0.80) rather than better.

Two things follow.  First, the decisive evidence for this task is local -- the
quoted-interpolation motif next to SQL keywords -- so a field that spans the
whole taint path buys nothing and gives the model room to fit family-specific
long-range layout that does not transfer.  Second, and more importantly for the
methodology: because in-distribution validation F1 is saturated (~0.999 for
every configuration), the validation signal cannot distinguish these models at
all, so early stopping selects an essentially arbitrary epoch and the OOD number
is dominated by seed noise.  Choosing R on the OOD split would be selecting on
the test set.  We therefore select on validation -- where all four tie -- and
break the tie on cost, giving dilations 1,2,4 (R = 15), the smallest and
fastest.  The limitation is stated in the notebook's discussion rather than
hidden.

The three loss terms
--------------------
L_cls : binary cross-entropy on the detection head.
L_att : attention guidance.  Cross-entropy between the attention distribution
        alpha and the normalised ground-truth taint mask m.  This is the
        contribution the project title names -- it tells the network *where* to
        look, not merely what to answer.
L_ret : supervised contrastive loss on the search head, with "same
        vulnerability family" as the positive relation.  This is what makes the
        embedding space a code-*search* index rather than a by-product of
        classification.

        L = L_cls + lambda_a * L_att + lambda_r * L_ret
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class DilatedBlock(nn.Module):
    """Pre-norm residual depth-wise-separable dilated convolution."""

    def __init__(self, d: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        # Depth-wise 3-tap dilated convolution: d*3 MACs per position instead
        # of the d*2d*3 a dense convolution would cost.
        self.dw = nn.Conv1d(d, d, kernel_size=3, padding=dilation,
                            dilation=dilation, groups=d)
        self.pw = nn.Linear(d, 2 * d)       # point-wise expansion for the GLU
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # h: (B, L, d);  pad_mask: (B, L) True where PAD
        z = self.norm(h)
        z = z.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        z = self.dw(z.transpose(1, 2)).transpose(1, 2)          # (B, L, d)
        a, b = self.pw(z).chunk(2, dim=-1)
        z = a * torch.sigmoid(b)                                # gated linear unit
        return h + self.drop(self.proj(z))


class CodeEncoder(nn.Module):
    def __init__(self, vocab_size: int, d: int = 128, max_len: int = 192,
                 dilations: Tuple[int, ...] = (1, 2, 4),
                 dropout: float = 0.1,
                 pad_id: int = 0):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DilatedBlock(d, k, dropout) for k in dilations])
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.01)
        with torch.no_grad():
            self.tok.weight[pad_id].zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = x.shape
        pad_mask = x.eq(self.pad_id)
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.drop(self.tok(x) + self.pos(pos))
        for blk in self.blocks:
            h = blk(h, pad_mask)
        return self.norm(h), pad_mask


# --------------------------------------------------------------------------- #
# Attention pooling  (the layer we supervise)
# --------------------------------------------------------------------------- #

class AdditiveAttentionPool(nn.Module):
    """Multi-head Bahdanau-style attention pooling.

        e_{h,t} = v_h^T tanh(W_h s_t + b_h)
        alpha_{h,t} = softmax_t(e_{h,t})            (PAD positions masked out)
        c_h = sum_t alpha_{h,t} s_t
        c   = W_o [c_1 ; ... ; c_H]

    The reported attention map is the head-average
    alpha_t = (1/H) sum_h alpha_{h,t}, which is the distribution the guidance
    loss acts on.
    """

    def __init__(self, d: int, n_heads: int = 4, d_att: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.score = nn.Sequential(nn.Linear(d, d_att), nn.Tanh(),
                                   nn.Linear(d_att, n_heads, bias=False))
        self.out = nn.Linear(d * n_heads, d)

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # h (B, L, d) -> c (B, d), alpha (B, L), logits (B, H, L)
        e = self.score(h).transpose(1, 2)                      # (B, H, L)
        e = e.masked_fill(pad_mask.unsqueeze(1), -1e9)
        a = torch.softmax(e, dim=-1)                           # (B, H, L)
        ctx = torch.einsum("bhl,bld->bhd", a, h)               # (B, H, d)
        c = self.out(ctx.flatten(1))                           # (B, d)
        return c, a.mean(dim=1), e


class MeanPool(nn.Module):
    """Attention-free control: length-normalised mean over valid tokens."""

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        valid = (~pad_mask).float()                            # (B, L)
        c = (h * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        alpha = valid / valid.sum(1, keepdim=True).clamp(min=1)
        return c, alpha, None


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class AGNCS(nn.Module):
    """Attention-Guided Neural Code Search network.

    Parameters
    ----------
    pooling : {"attention", "mean"}
        `"mean"` disables attention entirely (the attention-free ablation).
    emb_dim : int
        Dimension of the retrieval embedding produced by the search head.
    """

    def __init__(self, vocab_size: int, d: int = 128, max_len: int = 192,
                 n_heads: int = 4, dropout: float = 0.1, pad_id: int = 0,
                 pooling: str = "attention", emb_dim: int = 96,
                 dilations: Tuple[int, ...] = (1, 2, 4)):
        super().__init__()
        self.pooling_kind = pooling
        self.encoder = CodeEncoder(vocab_size, d, max_len, dilations, dropout, pad_id)
        self.pool = AdditiveAttentionPool(d, n_heads) if pooling == "attention" else MeanPool()
        self.cls_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.search_head = nn.Sequential(
            nn.Linear(d, d), nn.GELU(), nn.Linear(d, emb_dim))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, pad_mask = self.encoder(x)
        c, alpha, att_logits = self.pool(h, pad_mask)
        logit = self.cls_head(c).squeeze(-1)
        z = F.normalize(self.search_head(c), dim=-1)
        return {"logit": logit, "alpha": alpha, "att_logits": att_logits,
                "embed": z, "code_vec": c, "pad_mask": pad_mask,
                "token_states": h}

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)["embed"]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #

def attention_guidance_loss(alpha: torch.Tensor, mask: torch.Tensor,
                            pad_mask: torch.Tensor
                            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy between the attention map and the taint annotation.

        q_t    = m_t / sum_t m_t                (target distribution)
        L_att  = - sum_t q_t log alpha_t        (cross-entropy)
               = H(q) + KL(q || alpha)

    Samples with an empty annotation contribute nothing.  Minimising this is
    equivalent to maximising the attention mass landing on the annotated
    taint-flow (or safety-evidence) statements.

    Note the **irreducible floor**: the cross-entropy cannot fall below the
    target's own entropy H(q) = log|m| (about 3.4 nats for a typical
    30-token annotation), so a raw L_att near 3.5 already means the attention
    distribution closely matches the target.  Reading the raw value as "a big
    loss" would be a mistake, so this function also returns
    KL(q || alpha) = L_att - H(q), which is 0 at a perfect match and is what
    gets logged and plotted.  The two differ by a parameter-independent
    constant, so the gradients are identical.
    """
    m = mask.float().masked_fill(pad_mask, 0.0)
    denom = m.sum(dim=1)
    has = denom > 0
    zero = alpha.sum() * 0.0
    if not torch.any(has):
        return zero, zero.detach()
    q = m[has] / denom[has].unsqueeze(1)
    log_a = torch.log(alpha[has].clamp_min(1e-9))
    ce = -(q * log_a).sum(dim=1)
    h_q = -(q * torch.log(q.clamp_min(1e-9))).sum(dim=1)
    return ce.mean(), (ce - h_q).mean().detach()


def supervised_contrastive_loss(z: torch.Tensor, fam: torch.Tensor,
                                temperature: float = 0.1) -> torch.Tensor:
    """SupCon (Khosla et al., 2020) with the vulnerability family as the label.

        L_ret = - (1/|P(i)|) sum_{p in P(i)} log
                    exp(z_i . z_p / tau) / sum_{a != i} exp(z_i . z_a / tau)

    `P(i)` is the set of in-batch samples sharing i's family.  Using the family
    rather than the binary label is what gives the retrieval space its
    structure: two different injection patterns should not collapse onto the
    same point just because both are vulnerable.
    """
    B = z.size(0)
    if B < 3:
        return z.sum() * 0.0
    sim = z @ z.t() / temperature
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = fam.unsqueeze(0).eq(fam.unsqueeze(1)) & ~eye
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not torch.any(valid):
        return z.sum() * 0.0
    loss = -(log_prob * pos.float()).sum(1)[valid] / n_pos[valid].float()
    return loss.mean()


def total_loss(out: Dict[str, torch.Tensor], y: torch.Tensor,
               mask: torch.Tensor, fam: torch.Tensor,
               lambda_att: float = 0.0, lambda_ret: float = 0.0,
               temperature: float = 0.1,
               pos_weight: Optional[torch.Tensor] = None
               ) -> Tuple[torch.Tensor, Dict[str, float]]:
    l_cls = F.binary_cross_entropy_with_logits(out["logit"], y.float(),
                                               pos_weight=pos_weight)
    loss = l_cls
    parts = {"cls": float(l_cls.detach())}

    if lambda_att > 0.0 and out["att_logits"] is not None:
        l_att, kl = attention_guidance_loss(out["alpha"], mask, out["pad_mask"])
        loss = loss + lambda_att * l_att
        parts["att"] = float(l_att.detach())
        # KL(q||alpha): 0 means the attention map matches the annotation exactly.
        parts["att_kl"] = float(kl)

    if lambda_ret > 0.0:
        l_ret = supervised_contrastive_loss(out["embed"], fam, temperature)
        loss = loss + lambda_ret * l_ret
        parts["ret"] = float(l_ret.detach())

    parts["total"] = float(loss.detach())
    return loss, parts


# --------------------------------------------------------------------------- #
# Model-variant registry: this *is* the ablation design
# --------------------------------------------------------------------------- #

VARIANTS: Dict[str, dict] = {
    # name                        pooling       lambda_att  lambda_ret
    "M4_CNN_meanpool": dict(pooling="mean", lambda_att=0.0, lambda_ret=0.0),
    "M5_CNN_attention": dict(pooling="attention", lambda_att=0.0, lambda_ret=0.0),
    "M6_CNN_attn_contrastive": dict(pooling="attention", lambda_att=0.0, lambda_ret=0.3),
    "M7_AGNCS_full": dict(pooling="attention", lambda_att=0.5, lambda_ret=0.3),
}

VARIANT_LABELS = {
    "M4_CNN_meanpool": "CNN + mean pooling (no attention)",
    "M5_CNN_attention": "CNN + unguided attention",
    "M6_CNN_attn_contrastive": "CNN + attention + contrastive search head",
    "M7_AGNCS_full": "AGNCS (proposed): guided attention + contrastive",
}


def build_model(variant: str, vocab_size: int, max_len: int, d: int = 128,
                n_heads: int = 4, dropout: float = 0.1, emb_dim: int = 96,
                pad_id: int = 0,
                dilations: Tuple[int, ...] = (1, 2, 4)
                ) -> Tuple[AGNCS, dict]:
    cfg = VARIANTS[variant]
    model = AGNCS(vocab_size=vocab_size, d=d, max_len=max_len, n_heads=n_heads,
                  dropout=dropout, pad_id=pad_id, pooling=cfg["pooling"],
                  emb_dim=emb_dim, dilations=dilations)
    return model, cfg


def receptive_field(dilations: Tuple[int, ...], kernel: int = 3) -> int:
    """Token span one output position can see: 1 + (k-1) * sum(dilations)."""
    return 1 + (kernel - 1) * sum(dilations)
