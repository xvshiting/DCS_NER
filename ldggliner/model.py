"""
Bi-encoder + Cross-Attention fusion + Span scoring (GLiNER-like) using mDeBERTa.

- Encode text once with mDeBERTa -> token reps H: (B, N, d)
- Encode each (label + description) with mDeBERTa -> label reps Q: (B, M, d)
- Fuse labels with text via cross-attn: Q' = CrossAttn(Q, H)
- Build span reps from text tokens: S = FFN([H_start ; H_end])  (optionally width embedding)
- Score each span against each label: logits = S @ Q'^T  -> (B, num_spans, M)

This is a clean, extensible skeleton. You can:
- Share encoders (default) or use separate encoders.
- Add top-k label prefiltering (retrieval) before cross-attn.
- Add long-text chunking outside the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class SpanBatch:
    # Text
    text_input_ids: torch.LongTensor         # (B, N)
    text_attention_mask: torch.LongTensor    # (B, N)

    # Labels (label+desc) as a batch-of-batches flattened to (B*M, L)
    label_input_ids: torch.LongTensor        # (B*M, L)
    label_attention_mask: torch.LongTensor   # (B*M, L)

    # Number of labels per sample (M). Assumed fixed M across batch for simplicity.
    num_labels: int

    # Optional supervision:
    # span_targets: (B, S, M)  0/1 — true entity (span, label) pairs
    span_targets: Optional[torch.FloatTensor] = None


class CrossAttentionFusion(nn.Module):
    """Label->Text cross-attention. Q attends to H."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, h: torch.Tensor, h_key_padding_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        q: (B, M, d)
        h: (B, N, d)
        h_key_padding_mask: (B, N) True for PAD (to mask out)
        """
        ctx, _ = self.attn(query=q, key=h, value=h, key_padding_mask=h_key_padding_mask, need_weights=False)
        return self.ln(q + ctx)


class LabelSelfAttention(nn.Module):
    """Label->Label self-attention. Labels attend to each other.

    Output projection is zero-initialized so that at init this module is an
    identity (residual = 0), matching the V1 starting point.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """q: (B, M, d)"""
        ctx, _ = self.attn(query=q, key=q, value=q, need_weights=False)
        return self.ln(q + ctx)


class TextLabelCrossAttention(nn.Module):
    """Text->Label cross-attention. H attends to Q.

    Output projection is zero-initialized so that at init H is unchanged,
    preventing random noise from corrupting span representations early in training.
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        h: (B, N, d)  text tokens (query)
        q: (B, M, d)  label vectors (key/value)
        """
        ctx, _ = self.attn(query=h, key=q, value=q, need_weights=False)
        return self.ln(h + ctx)


class FusionStack(nn.Module):
    """
    One stack of three attention operations:
      1. Label->Text  cross-attention  (Q attends to H)
      2. Label->Label self-attention   (Q' attends to Q')
      3. Text->Label  cross-attention  (H attends to Q'')

    Input:  Q (B, M, d), H (B, N, d)
    Output: Q'' (B, M, d), H' (B, N, d)
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.label_to_text = CrossAttentionFusion(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.label_self    = LabelSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.text_to_label = TextLabelCrossAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        h: torch.Tensor,
        h_key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.label_to_text(q, h, h_key_padding_mask=h_key_padding_mask)  # (B, M, d)
        q = self.label_self(q)                                                 # (B, M, d)
        h = self.text_to_label(h, q)                                           # (B, N, d)
        return q, h


class SpanEnumerator:
    """Enumerate all spans up to max_width for each sequence length N (excluding pads by attention mask later)."""
    def __init__(self, max_width: int):
        self.max_width = int(max_width)

    def enumerate(self, seq_len: int, device: torch.device) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        """
        Returns:
          start_idx: (S,)
          end_idx:   (S,)
          width:     (S,)  (end-start)
        where S = sum_{w=1..max_width} (seq_len - w + 1)
        """
        starts = []
        ends = []
        widths = []
        for w in range(1, self.max_width + 1):
            num_spans = seq_len - w + 1
            if num_spans <= 0:
                break  # seq shorter than span width; no valid spans for this width or wider
            s = torch.arange(0, num_spans, device=device, dtype=torch.long)
            e = s + (w - 1)
            starts.append(s)
            ends.append(e)
            widths.append(torch.full_like(s, w - 1))
        start_idx = torch.cat(starts, dim=0)
        end_idx = torch.cat(ends, dim=0)
        width = torch.cat(widths, dim=0)
        return start_idx, end_idx, width


def span_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    span_mask: torch.BoolTensor,
    use_pos_weight: bool = False,
    pos_weight_cap: float = 200.0,
) -> torch.Tensor:
    """
    BCE loss over all valid (span, label) pairs.

    Args:
        logits:         (B, S, M)
        targets:        (B, S, M)  float 0/1
        span_mask:      (B, S)     True for valid (non-padding) spans
        use_pos_weight: if True, apply dynamic pos_weight to compensate
                        span-level class imbalance (n_neg >> n_pos).
                        Capped at pos_weight_cap to prevent early instability.
        pos_weight_cap: maximum pos_weight value (only used when use_pos_weight=True)
    """
    valid = span_mask.unsqueeze(-1).expand_as(logits)
    flat_logits  = logits[valid]
    flat_targets = targets[valid]

    if flat_logits.numel() == 0:
        return logits.sum() * 0.0

    if use_pos_weight:
        n_pos = flat_targets.sum().clamp(min=1)
        n_neg = (flat_targets.numel() - flat_targets.sum()).clamp(min=1)
        pw = (n_neg / n_pos).clamp(max=pos_weight_cap)
        return F.binary_cross_entropy_with_logits(
            flat_logits, flat_targets, pos_weight=pw, reduction="mean",
        )

    return F.binary_cross_entropy_with_logits(
        flat_logits, flat_targets, reduction="mean",
    )


class DebertaSchemaSpanModel(nn.Module):
    """
    Main model:
      Text encoder -> token reps
      Label encoder -> label reps
      Cross-attn fusion -> fused label reps
      Span scoring -> logits(span, label)
    """
    def __init__(
        self,
        backbone_name: str = "microsoft/mdeberta-v3-base",
        share_encoders: bool = True,
        use_width_embedding: bool = True,
        max_span_width: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        label_chunk_size: int = 16,
        use_pos_weight: bool = False,
        pos_weight_cap: float = 200.0,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(backbone_name)
        if share_encoders:
            self.label_encoder = self.text_encoder
        else:
            self.label_encoder = AutoModel.from_pretrained(backbone_name)

        d_model = self.text_encoder.config.hidden_size
        self.d_model = d_model
        self.max_span_width = int(max_span_width)
        self.span_enum = SpanEnumerator(max_width=max_span_width)
        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_cap = float(pos_weight_cap)

        self.fuse = CrossAttentionFusion(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.use_width_embedding = bool(use_width_embedding)
        if self.use_width_embedding:
            self.width_emb = nn.Embedding(max_span_width, d_model)  # width in [0..max_span_width-1]
            span_in = d_model * 2 + d_model
        else:
            span_in = d_model * 2

        self.span_ffn = make_mlp(span_in, hidden_dim=d_model * 4, output_dim=d_model, dropout=dropout)

        # Optional: bias term per label (helps calibration when many negatives)
        self.label_bias = nn.Parameter(torch.zeros(1))

        # Encode labels in chunks to bound peak memory when M is large.
        # Each chunk passes at most label_chunk_size sequences through the encoder.
        self.label_chunk_size = int(label_chunk_size)

    @staticmethod
    def _pool_cls(last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Use first token as CLS. Shape: (B, L, d) -> (B, d)."""
        return last_hidden_state[:, 0, :]

    def encode_text(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, N, d)

    def encode_labels(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        batch_size: int,
        num_labels: int,
    ) -> torch.Tensor:
        """
        input_ids/attention_mask: (B*M, L)
        returns Q: (B, M, d)

        Labels are encoded in chunks of size `label_chunk_size` to bound peak
        memory regardless of how large M is.
        """
        total = input_ids.size(0)  # B*M
        pooled_chunks = []
        for start in range(0, total, self.label_chunk_size):
            end = min(start + self.label_chunk_size, total)
            chunk_out = self.label_encoder(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
            )
            pooled_chunks.append(self._pool_cls(chunk_out.last_hidden_state))  # (chunk, d)
        pooled = torch.cat(pooled_chunks, dim=0)        # (B*M, d)
        q = pooled.view(batch_size, num_labels, -1)     # (B, M, d)
        return q

    def forward(self, batch: SpanBatch) -> Dict[str, torch.Tensor]:
        """
        Returns:
          logits: (B, S, M) where S = num_spans based on N and max_span_width
          loss: optional
          span_mask: (B, S) valid spans under attention_mask
          start_idx/end_idx/width: (S,)
        """
        B, N = batch.text_input_ids.shape
        M = int(batch.num_labels)
        device = batch.text_input_ids.device

        # 1) Encode text
        H = self.encode_text(batch.text_input_ids, batch.text_attention_mask)  # (B, N, d)

        # 2) Encode labels (label+desc)
        Q = self.encode_labels(batch.label_input_ids, batch.label_attention_mask, batch_size=B, num_labels=M)  # (B, M, d)

        # 3) Cross-attn fusion (labels attend to text)
        # key_padding_mask expects True for padding positions
        pad_mask = batch.text_attention_mask == 0  # (B, N) bool
        Qf = self.fuse(Q, H, h_key_padding_mask=pad_mask)  # (B, M, d)

        # 4) Enumerate spans (based on full N, later masked by attention_mask)
        start_idx, end_idx, width = self.span_enum.enumerate(seq_len=N, device=device)
        S = start_idx.numel()

        # 5) Build span representations from token reps
        H_start = H.index_select(dim=1, index=start_idx)  # (B, S, d)
        H_end   = H.index_select(dim=1, index=end_idx)    # (B, S, d)

        if self.use_width_embedding:
            W = self.width_emb(width).unsqueeze(0).expand(B, S, self.d_model)
            span_in = torch.cat([H_start, H_end, W], dim=-1)
        else:
            span_in = torch.cat([H_start, H_end], dim=-1)

        span_vec = self.span_ffn(span_in)

        # 6) Logits
        logits = torch.einsum("bsd,bmd->bsm", span_vec, Qf) + self.label_bias  # (B, S, M)

        # 7) Span validity mask
        attn = batch.text_attention_mask.bool()
        span_mask = attn.index_select(dim=1, index=start_idx) & \
                    attn.index_select(dim=1, index=end_idx)

        out: Dict[str, torch.Tensor] = {
            "logits": logits, "span_mask": span_mask,
            "start_idx": start_idx, "end_idx": end_idx, "width": width,
        }

        # 8) Optional loss
        if batch.span_targets is not None:
            out["loss"] = span_loss(
                logits, batch.span_targets, span_mask,
                use_pos_weight=self.use_pos_weight,
                pos_weight_cap=self.pos_weight_cap,
            )

        return out


class DebertaSchemaSpanModelV2(nn.Module):
    """
    V2: stacked bi-directional fusion.

    Compared to V1 (single Label->Text cross-attn), each FusionStack runs:
      1. Label->Text  cross-attention
      2. Label->Label self-attention
      3. Text->Label  cross-attention  (updates H, making spans label-aware)

    The stack is applied `num_fusion_stacks` times (default 2).
    Span representations are built from the final updated H', not the original H.
    """
    def __init__(
        self,
        backbone_name: str = "microsoft/mdeberta-v3-base",
        share_encoders: bool = True,
        use_width_embedding: bool = True,
        max_span_width: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        label_chunk_size: int = 16,
        num_fusion_stacks: int = 2,
        use_pos_weight: bool = False,
        pos_weight_cap: float = 200.0,
    ):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(backbone_name)
        if share_encoders:
            self.label_encoder = self.text_encoder
        else:
            self.label_encoder = AutoModel.from_pretrained(backbone_name)

        d_model = self.text_encoder.config.hidden_size
        self.d_model = d_model
        self.max_span_width = int(max_span_width)
        self.span_enum = SpanEnumerator(max_width=max_span_width)
        self.label_chunk_size = int(label_chunk_size)
        self.use_pos_weight = bool(use_pos_weight)
        self.pos_weight_cap = float(pos_weight_cap)

        self.fusion_stacks = nn.ModuleList([
            FusionStack(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_fusion_stacks)
        ])

        self.use_width_embedding = bool(use_width_embedding)
        if self.use_width_embedding:
            self.width_emb = nn.Embedding(max_span_width, d_model)
            span_in = d_model * 2 + d_model
        else:
            span_in = d_model * 2

        self.span_ffn = make_mlp(span_in, hidden_dim=d_model * 4, output_dim=d_model, dropout=dropout)
        self.label_bias = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _pool_cls(last_hidden_state: torch.Tensor) -> torch.Tensor:
        return last_hidden_state[:, 0, :]

    def encode_text(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state  # (B, N, d)

    def encode_labels(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        batch_size: int,
        num_labels: int,
    ) -> torch.Tensor:
        total = input_ids.size(0)
        pooled_chunks = []
        for start in range(0, total, self.label_chunk_size):
            end = min(start + self.label_chunk_size, total)
            chunk_out = self.label_encoder(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
            )
            pooled_chunks.append(self._pool_cls(chunk_out.last_hidden_state))
        pooled = torch.cat(pooled_chunks, dim=0)       # (B*M, d)
        return pooled.view(batch_size, num_labels, -1)  # (B, M, d)

    def forward(self, batch: SpanBatch) -> Dict[str, torch.Tensor]:
        B, N = batch.text_input_ids.shape
        M = int(batch.num_labels)
        device = batch.text_input_ids.device

        # 1) Encode text and labels
        H = self.encode_text(batch.text_input_ids, batch.text_attention_mask)   # (B, N, d)
        Q = self.encode_labels(batch.label_input_ids, batch.label_attention_mask,
                               batch_size=B, num_labels=M)                       # (B, M, d)

        # 2) Apply fusion stacks: each updates both Q and H
        pad_mask = batch.text_attention_mask == 0  # (B, N) True for PAD
        for stack in self.fusion_stacks:
            Q, H = stack(Q, H, h_key_padding_mask=pad_mask)
        # Q: (B, M, d) fully fused label reps
        # H: (B, N, d) label-aware text reps  ← used for span construction

        # 3) Enumerate spans
        start_idx, end_idx, width = self.span_enum.enumerate(seq_len=N, device=device)
        S = start_idx.numel()

        # 4) Build span reps from label-aware H
        H_start = H.index_select(dim=1, index=start_idx)
        H_end   = H.index_select(dim=1, index=end_idx)

        if self.use_width_embedding:
            W = self.width_emb(width).unsqueeze(0).expand(B, S, self.d_model)
            span_in = torch.cat([H_start, H_end, W], dim=-1)
        else:
            span_in = torch.cat([H_start, H_end], dim=-1)

        span_vec = self.span_ffn(span_in)

        # 5) Score spans against labels
        logits = torch.einsum("bsd,bmd->bsm", span_vec, Q) + self.label_bias

        # 6) Span validity mask
        attn      = batch.text_attention_mask.bool()
        span_mask = attn.index_select(dim=1, index=start_idx) & \
                    attn.index_select(dim=1, index=end_idx)

        out: Dict[str, torch.Tensor] = {
            "logits": logits, "span_mask": span_mask,
            "start_idx": start_idx, "end_idx": end_idx, "width": width,
        }

        # 7) Optional loss
        if batch.span_targets is not None:
            out["loss"] = span_loss(
                logits, batch.span_targets, span_mask,
                use_pos_weight=self.use_pos_weight,
                pos_weight_cap=self.pos_weight_cap,
            )

        return out


# -----------------------------
# Example usage (pseudo):
# -----------------------------
if __name__ == "__main__":
    # This is only a shape sanity check, not a full training script.
    B, N = 2, 128
    M, L = 20, 64
    model = DebertaSchemaSpanModel(
        backbone_name="microsoft/mdeberta-v3-base",
        share_encoders=True,
        use_width_embedding=True,
        max_span_width=12,
        num_heads=8,
        dropout=0.1,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = SpanBatch(
        text_input_ids=torch.randint(0, 1000, (B, N), device=device),
        text_attention_mask=torch.ones(B, N, device=device, dtype=torch.long),
        label_input_ids=torch.randint(0, 1000, (B * M, L), device=device),
        label_attention_mask=torch.ones(B * M, L, device=device, dtype=torch.long),
        num_labels=M,
        span_targets=None,
    )

    out = model(batch)
    print(out["logits"].shape)  # (B, S, M)
