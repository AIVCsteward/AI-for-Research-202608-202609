"""Precompute ESM-2 protein embeddings for the competition proteins.

Reads ``data/external/protein_mapping.csv`` (one row per proteome column, in the
feature-contract order) and produces a fixed 480-dim embedding per protein using
ESM-2 ``esm2_t12_35M_UR50D`` — the ``(4232, 480)`` prior from the design doc.

Weights are loaded from the locally-cached Meta CDN checkpoint
(``data/external/cache/esm2_t12_35M_UR50D.pt``) via ``fair-esm``, since
HuggingFace is not reachable from this network.  The contact-regression head is
omitted (``regression_data=None``) — it is only needed for contact prediction,
not for residue/sequence embeddings.

For proteins with a resolved sequence, the embedding is the mean of the last
layer's per-residue representations (special ``<cls>``/``<eos>``/``<pad>`` tokens
excluded; sequences are truncated to 1024, ESM-2's context window).  Unresolved
proteins (pseudogenes, 2-micron/Ty genes, ``1-Oct``, ...) get the mean of all
resolved embeddings, so the output matrix is always full and aligned to the
proteome column order.

Output (aligned to the proteome CSV header order):
  data/external/protein_esm2_480.npy   (N_proteins, 480) float32

Runs on CPU by default.  Checkpoints the matrix to a ``.part`` file every few
batches so an interrupted run can be resumed with ``--resume``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Allow running as ``python scripts/compute_esm_embeddings.py`` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "external"
MAPPING_PATH = DATA_DIR / "protein_mapping.csv"
OUTPUT_PATH = DATA_DIR / "protein_esm2_480.npy"
PART_PATH = DATA_DIR / "protein_esm2_480.part.npy"
MODEL_PATH = DATA_DIR / "cache" / "esm2_t12_35M_UR50D.pt"

MODEL_NAME = "esm2_t12_35M_UR50D"
MAX_LENGTH = 1024  # ESM-2 context window; longer proteins are truncated.


def load_sequences() -> Tuple[List[str], np.ndarray, List[str]]:
    """Return ``(sequences, resolved_mask, names)`` from the mapping CSV."""
    rows = list(csv.DictReader(MAPPING_PATH.open(newline="", encoding="utf-8")))
    names = [r["raw_name"] for r in rows]
    sequences: List[str] = []
    resolved: List[bool] = []
    for r in rows:
        seq = (r.get("sequence") or "").strip()
        resolved.append(bool(seq))
        sequences.append(seq)
    return sequences, np.asarray(resolved, dtype=bool), names


def load_model():
    import torch

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    # torch >= 2.6 defaults torch.load to weights_only=True, which rejects the
    # argparse.Namespace pickled inside the fair-esm checkpoint.  Allowlist it.
    torch.serialization.add_safe_globals([__import__("argparse").Namespace])

    from esm.pretrained import load_model_and_alphabet_core

    model_data = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    model, alphabet = load_model_and_alphabet_core(MODEL_NAME, model_data, None)
    model.eval()
    return model, alphabet


def compute_all(sequences, resolved, resume: bool = False, batch_size: int = 16) -> np.ndarray:
    import torch

    model, alphabet = load_model()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    cls_idx = alphabet.cls_idx
    eos_idx = alphabet.eos_idx
    pad_idx = alphabet.padding_idx
    last_layer = model.num_layers
    print(f"model {MODEL_NAME} -> {last_layer} layers, device {device}, "
          f"special ids cls={cls_idx} eos={eos_idx} pad={pad_idx}")

    # SGD sequences end with a stop codon ``*`` and may contain rare non-standard
    # residues that ESM's alphabet does not have.  Drop ``*`` and map any other
    # unknown character to ``X`` so every resolved sequence encodes cleanly.
    valid_toks = set(alphabet.tok_to_idx.keys())
    def sanitize(seq: str) -> str:
        return "".join(c if c in valid_toks else "X" for c in seq.replace("*", ""))
    sequences = [sanitize(s) for s in sequences]

    n = len(sequences)
    emb = np.zeros((n, 480), dtype=np.float32)
    done = np.zeros(n, dtype=bool)

    if resume and PART_PATH.exists():
        saved = np.load(PART_PATH)
        if saved.shape == (n, 480):
            emb[:] = saved
            done = emb.any(axis=1)
            print(f"resumed: {int(done.sum())}/{n} already computed")
        else:
            print(f"[WARN] part file shape {saved.shape} != {(n, 480)}; ignoring")

    # Sort by length so batches contain similarly-sized proteins (less padding).
    idxs = [i for i in range(n) if resolved[i] and not done[i]]
    idxs.sort(key=lambda i: len(sequences[i]))
    print(f"computing {len(idxs)} embeddings in batches of {batch_size}")

    def mean_pool(repr_t, tokens):
        keep = (tokens != cls_idx) & (tokens != eos_idx) & (tokens != pad_idx)
        denom = keep.sum(dim=1, keepdim=True).clamp_min(1)
        return (repr_t * keep.unsqueeze(-1)).sum(dim=1) / denom

    with torch.no_grad():
        for start in range(0, len(idxs), batch_size):
            batch_idx = idxs[start : start + batch_size]
            batch = [
                (str(i), sequences[i][:MAX_LENGTH])
                for i in batch_idx
            ]
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[last_layer])
            repr_t = out["representations"][last_layer].to(torch.float32)
            emb[batch_idx] = mean_pool(repr_t, tokens).cpu().numpy()
            done[batch_idx] = True

            if (start // batch_size) % 25 == 0:
                np.save(PART_PATH, emb)
                print(f"  {min(start + batch_size, len(idxs))}/{len(idxs)} done, "
                      f"saved checkpoint", flush=True)

    # Fallback: fill unresolved proteins with the mean of resolved embeddings.
    resolved_emb = emb[resolved]
    fallback = resolved_emb.mean(axis=0) if resolved_emb.shape[0] else np.zeros(480, dtype=np.float32)
    emb[~resolved] = fallback

    np.save(PART_PATH, emb)
    return emb


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute ESM-2 protein embeddings")
    parser.add_argument("--resume", action="store_true", help="Resume from .part checkpoint")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    sequences, resolved, names = load_sequences()
    print(f"loaded {len(sequences)} proteins, {int(resolved.sum())} with sequence")

    emb = compute_all(sequences, resolved, resume=args.resume, batch_size=args.batch_size)
    np.save(OUTPUT_PATH, emb)
    print(f"saved {emb.shape} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
