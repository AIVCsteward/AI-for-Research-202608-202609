"""Canonical contract-first V2 model implementation.

This is the single source of truth.  The top-level ``model_v2`` package only
re-exports symbols from ``baseline.baseline`` for backwards compatibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import math

import torch
from torch import nn


@dataclass(frozen=True)
class V2Config:
    n_proteins: int
    morgan_dim: int = 2048
    descriptor_dim: int = 217
    genome_dim: int = 24
    latent_dim: int = 64
    protein_rank: int = 32
    medium_vocab_size: int = 2
    batch_vocab_sizes: tuple[int, int, int] = (2, 2, 2)
    dropout: float = 0.05
    batch_enabled: bool = True
    batch_structure: str = "flat_batch"
    response_gate_enabled: bool = False
    response_gate_initial: float = 0.25
    response_rms_cap: float = 0.0
    batch_field_dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class V2Batch:
    morgan: torch.Tensor
    descriptors: torch.Tensor
    chemical_valid_mask: torch.Tensor
    chemical_mapping: torch.Tensor
    chemical_confidence: torch.Tensor
    chemical_structure_valid: torch.Tensor
    genome: torch.Tensor
    genome_valid_mask: torch.Tensor
    genome_mapping: torch.Tensor
    genome_confidence: torch.Tensor
    genome_proxy: torch.Tensor
    medium: torch.Tensor
    condition_numeric: torch.Tensor
    batch_categorical: torch.Tensor
    is_control: torch.Tensor

    def index_select(self, index) -> "V2Batch":
        return V2Batch(**{field.name: getattr(self, field.name)[index] for field in fields(self)})

    def to(self, device) -> "V2Batch":
        return V2Batch(**{field.name: getattr(self, field.name).to(device) for field in fields(self)})

    def replace(self, **changes) -> "V2Batch":
        return replace(self, **changes)


def _mlp(dim_in: int, dim_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim_in, dim_out), nn.LayerNorm(dim_out), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(dim_out, dim_out), nn.GELU(),
    )


class ChemicalEncoder(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.morgan = _mlp(cfg.morgan_dim * 2, cfg.latent_dim, cfg.dropout)
        self.descriptors = _mlp(cfg.descriptor_dim * 2, cfg.latent_dim, cfg.dropout)
        self.mapping = nn.Embedding(4, 8)
        self.confidence = nn.Embedding(4, 4)
        self.fuse = _mlp(cfg.latent_dim * 2 + 13, cfg.latent_dim, cfg.dropout)

    def forward(self, morgan, descriptors, valid_mask, mapping, confidence, structure_valid):
        m_mask = valid_mask[:, :morgan.shape[1]].to(morgan.dtype)
        d_mask = valid_mask[:, morgan.shape[1]:].to(descriptors.dtype)
        morgan_latent = self.morgan(torch.cat([morgan * m_mask, m_mask], 1))
        descriptor_latent = self.descriptors(torch.cat([descriptors * d_mask, d_mask], 1))
        flags = torch.cat([self.mapping(mapping), self.confidence(confidence), structure_valid], 1)
        return self.fuse(torch.cat([morgan_latent, descriptor_latent, flags], 1))


class GenomeEncoder(nn.Module):
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.mapping = nn.Embedding(4, 8)
        self.confidence = nn.Embedding(4, 4)
        self.net = _mlp(cfg.genome_dim * 2 + 13, cfg.latent_dim, cfg.dropout)

    def forward(self, genome, valid_mask, mapping, confidence, proxy):
        valid = valid_mask.to(genome.dtype)
        flags = torch.cat([self.mapping(mapping), self.confidence(confidence), proxy], 1)
        return self.net(torch.cat([genome * valid, valid, flags], 1))


class LowRankDecoder(nn.Module):
    """The basis is a trainable model parameter, never fitted from validation labels."""
    def __init__(self, dim_in: int, cfg: V2Config, bias: bool = True, zero_output: bool = False):
        super().__init__()
        self.coefficients = nn.Linear(dim_in, cfg.protein_rank)
        self.basis = nn.Linear(cfg.protein_rank, cfg.n_proteins, bias=bias)
        if zero_output:
            nn.init.zeros_(self.basis.weight)
            if self.basis.bias is not None:
                nn.init.zeros_(self.basis.bias)

    def forward(self, x):
        return self.basis(self.coefficients(x))


class BaselineBranch(nn.Module):
    """Allowed: genome, medium, temperature and time. Forbidden: chemical/batch."""
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.medium = nn.Embedding(cfg.medium_vocab_size, 8)
        self.condition = _mlp(cfg.latent_dim + 12, cfg.latent_dim, cfg.dropout)
        self.decode = LowRankDecoder(cfg.latent_dim, cfg)

    def forward(self, genome_latent, medium, condition_numeric):
        latent = self.condition(torch.cat([genome_latent, self.medium(medium), condition_numeric], 1))
        return self.decode(latent)


class ResponseBranch(nn.Module):
    """Allowed: chemistry/genome/culture. Forbidden: measurement batch fields."""
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.medium = nn.Embedding(cfg.medium_vocab_size, 8)
        self.interaction = _mlp(cfg.latent_dim * 3 + 12, cfg.latent_dim, cfg.dropout)
        self.decode = LowRankDecoder(cfg.latent_dim, cfg, bias=False)
        if cfg.response_gate_enabled:
            if not 0.0 < cfg.response_gate_initial < 1.0:
                raise ValueError("response_gate_initial must be strictly between zero and one")
            self.gate = nn.Linear(cfg.latent_dim, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(
                self.gate.bias,
                math.log(cfg.response_gate_initial / (1.0 - cfg.response_gate_initial)),
            )

    def forward(self, chemical_latent, genome_latent, medium, condition_numeric):
        interaction = chemical_latent * genome_latent
        value = torch.cat([
            chemical_latent, genome_latent, interaction,
            self.medium(medium), condition_numeric,
        ], 1)
        return self.decode(self.interaction(value))

    def gate_value(self, chemical_latent):
        if not hasattr(self, "gate"):
            return None
        return torch.sigmoid(self.gate(chemical_latent))


class BatchBranch(nn.Module):
    """Allowed: data_source, instrument and plate only."""
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(size, 8) for size in cfg.batch_vocab_sizes])
        self.net = _mlp(24, cfg.latent_dim, cfg.dropout)
        # Zero initialization makes the initial batch correction exactly zero.
        self.decode = LowRankDecoder(cfg.latent_dim, cfg, bias=False, zero_output=True)
        self.field_dropout = float(cfg.batch_field_dropout)
        if not 0.0 <= self.field_dropout < 1.0:
            raise ValueError("batch_field_dropout must be in [0, 1)")

    def forward(self, batch_categorical):
        if self.training and self.field_dropout > 0:
            # Drop the entire technical-field group for selected samples.  Code
            # zero is the vocabulary's explicit unknown/missing category.
            dropped = torch.rand(
                (batch_categorical.shape[0], 1), device=batch_categorical.device,
            ) < self.field_dropout
            batch_categorical = torch.where(
                dropped, torch.zeros_like(batch_categorical), batch_categorical,
            )
        encoded = [embedding(batch_categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.decode(self.net(torch.cat(encoded, 1)))


class HierarchicalBatchLevel(nn.Module):
    """One independently regularizable low-rank technical residual level."""
    def __init__(self, vocab_size: int, cfg: V2Config):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 8, padding_idx=0)
        self.net = _mlp(8, cfg.latent_dim, cfg.dropout)
        self.decode = LowRankDecoder(cfg.latent_dim, cfg, bias=False, zero_output=True)

    def forward(self, codes, enabled):
        value = self.decode(self.net(self.embedding(codes)))
        return value * enabled.to(value.dtype).unsqueeze(1)


class HierarchicalBatchBranch(nn.Module):
    """Source + instrument + plate with deterministic nested OOV fallback."""
    def __init__(self, cfg: V2Config):
        super().__init__()
        self.source = HierarchicalBatchLevel(cfg.batch_vocab_sizes[0], cfg)
        self.instrument = HierarchicalBatchLevel(cfg.batch_vocab_sizes[1], cfg)
        self.plate = HierarchicalBatchLevel(cfg.batch_vocab_sizes[2], cfg)

    def forward(self, codes):
        source_known = codes[:, 0].ne(0)
        instrument_known = source_known & codes[:, 1].ne(0)
        plate_known = instrument_known & codes[:, 2].ne(0)
        source = self.source(codes[:, 0], source_known)
        instrument = self.instrument(codes[:, 1], instrument_known)
        plate = self.plate(codes[:, 2], plate_known)
        return source, instrument, plate


class AnchoredVirtualCellV2(nn.Module):
    def __init__(self, cfg: V2Config, train_protein_mean: torch.Tensor | None = None):
        super().__init__()
        self.cfg = cfg
        self.chemical_encoder = ChemicalEncoder(cfg)
        self.genome_encoder = GenomeEncoder(cfg)
        self.baseline_branch = BaselineBranch(cfg)
        self.response_branch = ResponseBranch(cfg)
        if cfg.batch_structure == "flat_batch":
            self.batch_branch = BatchBranch(cfg)
        elif cfg.batch_structure == "hierarchical_batch":
            self.batch_branch = HierarchicalBatchBranch(cfg)
        else:
            raise ValueError(f"unsupported batch_structure: {cfg.batch_structure}")
        anchor = torch.zeros(cfg.n_proteins) if train_protein_mean is None else train_protein_mean.float()
        if anchor.shape != (cfg.n_proteins,):
            raise ValueError("train protein mean shape does not match feature contract")
        self.register_buffer("train_protein_mean", anchor)

    def forward(self, batch: V2Batch) -> dict[str, torch.Tensor | None]:
        chemical_latent = self.chemical_encoder(
            batch.morgan, batch.descriptors, batch.chemical_valid_mask,
            batch.chemical_mapping, batch.chemical_confidence, batch.chemical_structure_valid,
        )
        genome_latent = self.genome_encoder(
            batch.genome, batch.genome_valid_mask, batch.genome_mapping,
            batch.genome_confidence, batch.genome_proxy,
        )
        y_baseline = self.train_protein_mean + self.baseline_branch(
            genome_latent, batch.medium, batch.condition_numeric
        )
        response_raw = self.response_branch(
            chemical_latent, genome_latent, batch.medium, batch.condition_numeric
        )
        response_gate = self.response_branch.gate_value(chemical_latent)
        if response_gate is not None:
            if self.cfg.response_rms_cap <= 0:
                raise ValueError("score-aligned response gate requires a positive RMS cap")
            rms = response_raw.square().mean(dim=1, keepdim=True).sqrt()
            limiter = (float(self.cfg.response_rms_cap) / rms.clamp_min(1e-8)).clamp(max=1.0)
            response_raw = response_raw * limiter * response_gate
        delta_response = response_raw * (~batch.is_control.bool()).to(response_raw.dtype)
        delta_source = delta_instrument = delta_plate = torch.zeros_like(delta_response)
        if self.cfg.batch_enabled and self.cfg.batch_structure == "hierarchical_batch":
            delta_source, delta_instrument, delta_plate = self.batch_branch(batch.batch_categorical)
            delta_batch = delta_source + delta_instrument + delta_plate
        elif self.cfg.batch_enabled:
            delta_batch = self.batch_branch(batch.batch_categorical)
        else:
            # Structural no-batch mode: the branch is not evaluated and its
            # output is exactly zero regardless of batch metadata.
            delta_batch = torch.zeros_like(delta_response)
        y_pred = y_baseline + delta_response + delta_batch
        return {
            "y_pred": y_pred,
            "y_anchor": y_baseline + delta_batch,
            "y_baseline": y_baseline,
            "delta_response": delta_response,
            "delta_batch": delta_batch,
            "delta_source": delta_source,
            "delta_instrument": delta_instrument,
            "delta_plate": delta_plate,
            "chemical_latent": chemical_latent,
            "genome_latent": genome_latent,
            "response_gate": response_gate,
            "similarity_gate": None,
        }
