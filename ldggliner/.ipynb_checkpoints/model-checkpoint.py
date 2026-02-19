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

    # Optional supervision (multi-label over spans):
    # span_targets: (B, num_spans, M) with 0/1, or float in [0,1]
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
            s = torch.arange(0, seq_len - w + 1, device=device, dtype=torch.long)
            e = s + (w - 1)
            starts.append(s)
            ends.append(e)
            widths.append(torch.full_like(s, w - 1))
        start_idx = torch.cat(starts, dim=0)
        end_idx = torch.cat(ends, dim=0)
        width = torch.cat(widths, dim=0)
        return start_idx, end_idx, width


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
        """
        out = self.label_encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_cls(out.last_hidden_state)  # (B*M, d)
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
        start_idx, end_idx, width = self.span_enum.enumerate(seq_len=N, device=device)  # (S,)
        S = start_idx.numel()

        # 5) Build span representations from token reps
        # Gather start/end token reps: (B, S, d)
        H_start = H.index_select(dim=1, index=start_idx)  # (B, S, d)
        H_end = H.index_select(dim=1, index=end_idx)      # (B, S, d)

        if self.use_width_embedding:
            W = self.width_emb(width).unsqueeze(0).expand(B, S, self.d_model)  # (B, S, d)
            span_in = torch.cat([H_start, H_end, W], dim=-1)                   # (B, S, 3d)
        else:
            span_in = torch.cat([H_start, H_end], dim=-1)                      # (B, S, 2d)

        span_vec = self.span_ffn(span_in)  # (B, S, d)

        # 6) Compute logits via dot-product with fused label vectors
        # logits[b, s, m] = <span_vec[b, s], Qf[b, m]>
        logits = torch.einsum("bsd,bmd->bsm", span_vec, Qf) + self.label_bias  # (B, S, M)

        # 7) Span validity mask (exclude spans that touch padding tokens)
        # Valid if both start and end positions are within attention_mask=1
        attn = batch.text_attention_mask.bool()  # (B, N)
        valid_start = attn.index_select(dim=1, index=start_idx)  # (B, S)
        valid_end = attn.index_select(dim=1, index=end_idx)      # (B, S)
        span_mask = valid_start & valid_end                       # (B, S)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,             # (B, S, M)
            "span_mask": span_mask,       # (B, S)
            "start_idx": start_idx,       # (S,)
            "end_idx": end_idx,           # (S,)
            "width": width,               # (S,)
        }

        # 8) Optional loss (multi-label BCE over spans x labels)
        if batch.span_targets is not None:
            # span_targets expected shape: (B, S, M)
            if batch.span_targets.shape != logits.shape:
                raise ValueError(f"span_targets shape {batch.span_targets.shape} must match logits {logits.shape}")

            # Mask out invalid spans by setting them to ignore (we'll zero their loss weight)
            # BCEWithLogitsLoss supports per-element weights.
            weight = span_mask.unsqueeze(-1).float()  # (B, S, 1)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.span_targets,
                weight=weight,
                reduction="sum",
            )
            denom = weight.sum().clamp_min(1.0) * M
            out["loss"] = loss / denom

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
