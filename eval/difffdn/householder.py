"""Householder helpers for tiny DiffFDN experiments.

SOURCE:
- flamo HouseholderMatrix reference (Dal Santo et al.):
  https://github.com/gdalsanto/flamo/blob/4c8097d4feda76132691bb2a3e465ebcba11dcea/flamo/processor/dsp.py#L621-L725
- This file keeps a minimal local adaptation for tiny N=4 experiments:
  raw v -> unit u, then apply y = x - 2*u*(u^T x).
"""

from __future__ import annotations

import torch


def unit_vector_from_raw(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Map a raw parameter vector to a unit vector."""
    if v.ndim != 1:
        raise ValueError(f"Expected raw vector shape (N,), got {tuple(v.shape)}")
    norm = torch.linalg.vector_norm(v)
    return v / torch.clamp(norm, min=eps)


def apply_householder(x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Apply y = x - 2*u*(u^T x) for x=(N,) or x=(B,N), u=(N,)."""
    if u.ndim != 1:
        raise ValueError(f"Expected u shape (N,), got {tuple(u.shape)}")
    if x.shape[-1] != u.shape[0]:
        raise ValueError(f"x last dim {x.shape[-1]} must equal u size {u.shape[0]}")

    if x.ndim == 1:
        proj = torch.dot(x, u)
        return x - 2.0 * u * proj
    if x.ndim == 2:
        proj = torch.sum(x * u, dim=-1, keepdim=True)
        return x - 2.0 * proj * u
    raise ValueError(f"Expected x shape (N,) or (B,N), got {tuple(x.shape)}")


def householder_matrix(u: torch.Tensor) -> torch.Tensor:
    """Build U = I - 2uu^T (for diagnostics/regularizers only)."""
    if u.ndim != 1:
        raise ValueError(f"Expected u shape (N,), got {tuple(u.shape)}")
    n = u.shape[0]
    eye = torch.eye(n, dtype=u.dtype, device=u.device)
    return eye - 2.0 * torch.outer(u, u)
