"""Mask-aware PatchTST-style temporal encoder for multivariate time series."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.types import NormalizationStats


class MaskedInstanceNorm(nn.Module):
    """Per-sample, per-channel normalization with recoverable statistics."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, x: Tensor, time_mask: Tensor, channel_mask: Tensor
    ) -> Tuple[Tensor, NormalizationStats]:
        valid = time_mask[:, None, :].to(x.dtype) * channel_mask[:, :, None].to(x.dtype)
        safe_x = torch.nan_to_num(x)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (safe_x * valid).sum(dim=-1, keepdim=True) / count
        variance = ((safe_x - mean).square() * valid).sum(dim=-1, keepdim=True) / count
        scale = variance.clamp_min(0.0).add(self.eps).sqrt()
        normalized = ((safe_x - mean) / scale) * valid
        stats: NormalizationStats = {
            "mean": mean,
            "scale": scale,
            "time_mask": time_mask,
            "channel_mask": channel_mask,
        }
        return normalized, stats

    @staticmethod
    def recover(normalized: Tensor, stats: NormalizationStats) -> Tensor:
        """Invert normalization on valid input positions."""

        valid = (
            stats["time_mask"][:, None, :].to(normalized.dtype)
            * stats["channel_mask"][:, :, None].to(normalized.dtype)
        )
        return (normalized * stats["scale"] + stats["mean"]) * valid


def patchify(x: Tensor, patch_len: int, stride: int) -> Tensor:
    """Convert ``[B,C,T]`` into ``[B,C,N,P]`` with unfold."""

    if x.ndim != 3:
        raise ValueError(f"Expected [B,C,T], got {tuple(x.shape)}")
    _, _, length = x.shape
    if length < patch_len:
        raise ValueError(f"sequence length {length} < patch_len {patch_len}")
    return x.unfold(dimension=2, size=patch_len, step=stride).contiguous()


class PatchEmbedding(nn.Module):
    """Project each patch token to model dimension."""

    def __init__(self, patch_len: int, d_model: int) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, patches: Tensor) -> Tensor:
        return self.proj(patches)


class PatchEncoderLayer(nn.Module):
    """Transformer block over patch tokens (channel-independent batching)."""

    def __init__(self, d_model: int, nhead: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.self_attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ff(x))
        return x


class PatchTransformer(nn.Module):
    """Apply self-attention independently to each channel's patch sequence."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                PatchEncoderLayer(d_model, nhead, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )

    def forward(self, h: Tensor) -> Tensor:
        batch, channels, num_patches, d_model = h.shape
        flat = h.reshape(batch * channels, num_patches, d_model)
        for layer in self.layers:
            flat = layer(flat)
        return flat.reshape(batch, channels, num_patches, d_model)


class ChannelMixer(nn.Module):
    """Cross-channel attention at each patch position."""

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor) -> Tensor:
        batch, channels, num_patches, d_model = h.shape
        flat = h.permute(0, 2, 1, 3).reshape(batch * num_patches, channels, d_model)
        attn_out, _ = self.attn(flat, flat, flat, need_weights=False)
        mixed = self.norm(flat + self.dropout(attn_out))
        return mixed.reshape(batch, num_patches, channels, d_model).permute(0, 2, 1, 3)


class TemporalEncoder(nn.Module):
    """Encode ``x[B,C,T]`` into contextual states ``H[B,T,D]`` via PatchTST trunk.

    Pipeline: masked instance norm -> patchify -> patch Transformer ->
    channel mixer -> overlap-aware upsample back to the original timeline.
    """

    def __init__(
        self,
        max_channels: int,
        d_model: int = 128,
        nhead: int = 4,
        dropout: float = 0.1,
        transformer_layers: int = 3,
        patch_len: int = 16,
        patch_stride: int = 8,
        d_ff: int = 256,
    ) -> None:
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.max_channels = max_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.instance_norm = MaskedInstanceNorm()
        self.patch_embed = PatchEmbedding(patch_len, d_model)
        self.patch_transformer = PatchTransformer(
            d_model, nhead, transformer_layers, d_ff, dropout
        )
        self.channel_mixer = ChannelMixer(d_model, nhead, dropout)
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _patch_position_encoding(
        num_patches: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        position = torch.arange(num_patches, device=device, dtype=torch.float32)[:, None]
        div = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / dim)
        )
        encoding = torch.zeros(num_patches, dim, device=device, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div)
        if dim > 1:
            encoding[:, 1::2] = torch.cos(position * div[: encoding[:, 1::2].shape[1]])
        return encoding.to(dtype=dtype)

    @staticmethod
    def _patches_to_timesteps(
        h: Tensor, length: int, patch_len: int, stride: int
    ) -> Tensor:
        """Map ``[B,N,D]`` patch tokens to ``[B,T,D]`` with overlap averaging."""

        batch, num_patches, d_model = h.shape
        out = torch.zeros(batch, length, d_model, device=h.device, dtype=h.dtype)
        counts = torch.zeros(batch, length, 1, device=h.device, dtype=h.dtype)
        for index in range(num_patches):
            start = index * stride
            end = min(start + patch_len, length)
            span = end - start
            if span <= 0:
                continue
            out[:, start:end, :] += h[:, index : index + 1, :].expand(-1, span, -1)
            counts[:, start:end, :] += 1.0
        return out / counts.clamp_min(1.0)

    def forward(
        self, x: Tensor, time_mask: Tensor, channel_mask: Tensor
    ) -> Tuple[Tensor, NormalizationStats]:
        """Encode a padded batch and return normalization metadata."""

        if x.ndim != 3:
            raise ValueError("x must have shape [B,C,T]")
        batch, channels, original_length = x.shape
        if channels > self.max_channels:
            raise ValueError(
                f"received {channels} channels, max_channels={self.max_channels}"
            )
        if time_mask.shape != (batch, original_length) or channel_mask.shape != (
            batch,
            channels,
        ):
            raise ValueError("mask shapes must be [B,T] and [B,C]")
        time_mask = time_mask.to(device=x.device, dtype=torch.bool)
        channel_mask = channel_mask.to(device=x.device, dtype=torch.bool)

        x, stats = self.instance_norm(x, time_mask, channel_mask)
        valid = (
            time_mask[:, None, :].to(x.dtype) * channel_mask[:, :, None].to(x.dtype)
        )
        x = x * valid

        padded_length = original_length
        if padded_length < self.patch_len:
            pad = self.patch_len - padded_length
            x = F.pad(x, (0, pad))
            time_mask = F.pad(time_mask, (0, pad), value=False)
            padded_length = x.shape[-1]

        pad_channels = self.max_channels - channels
        if pad_channels > 0:
            x = F.pad(x, (0, 0, 0, pad_channels))
            padded_channel_mask = F.pad(channel_mask, (0, pad_channels), value=False)
        else:
            padded_channel_mask = channel_mask
        channel_gate = padded_channel_mask[:, :, None].to(x.dtype)
        x = x * channel_gate

        patches = patchify(x, self.patch_len, self.patch_stride)
        h = self.patch_embed(patches)
        h = h + self._patch_position_encoding(
            h.shape[2], self.d_model, h.device, h.dtype
        )[None, None, :, :]
        h = self.patch_transformer(h)
        h = h + self.channel_mixer(h)

        active = padded_channel_mask[:, :channels, None, None].to(h.dtype)
        h = (h[:, :channels] * active).sum(dim=1) / active.sum(dim=1).clamp_min(1.0)
        h_timestep = self._patches_to_timesteps(
            h, padded_length, self.patch_len, self.patch_stride
        )
        h_timestep = self.output_norm(h_timestep)
        h_timestep = h_timestep[:, :original_length, :]
        h_timestep = h_timestep * stats["time_mask"][:, :, None].to(h_timestep.dtype)
        return h_timestep, stats
