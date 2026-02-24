"""Minimal frequency-sampled differentiable tiny-FDN model (N=4)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from .householder import householder_matrix, unit_vector_from_raw


def _complex_dtype(real_dtype: torch.dtype) -> torch.dtype:
    if real_dtype == torch.float32:
        return torch.complex64
    if real_dtype == torch.float64:
        return torch.complex128
    raise ValueError(f"Unsupported real dtype: {real_dtype}")


def hadamard4_matrix(
    *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    return 0.5 * torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )


def gains_from_rt60(
    delay_samples: Iterable[int], rt60: float, sr: float, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    delay = torch.as_tensor(list(delay_samples), dtype=dtype)
    t60 = max(float(rt60), 1e-3)
    sample_rate = max(float(sr), 1.0)
    gains = torch.pow(10.0, (-3.0 * delay) / (t60 * sample_rate))
    return gains


def default_io_vectors(
    n: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b = torch.full((n,), 1.0 / float(n), dtype=dtype, device=device)
    if n == 4:
        c_l = torch.tensor([0.5, -0.5, 0.5, -0.5], dtype=dtype, device=device)
        c_r = torch.tensor([0.5, 0.5, -0.5, -0.5], dtype=dtype, device=device)
    else:
        c_l = torch.full((n,), 1.0 / float(n), dtype=dtype, device=device)
        c_r = torch.full((n,), 1.0 / float(n), dtype=dtype, device=device)
    return b, c_l, c_r


def matrix_from_type(
    matrix_type: str,
    *,
    u: Optional[torch.Tensor],
    device: Optional[torch.device],
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix_kind = matrix_type.lower()
    if matrix_kind == "householder":
        if u is None:
            raise ValueError("Householder matrix_type requires unit vector u")
        return householder_matrix(u)
    if matrix_kind == "hadamard":
        return hadamard4_matrix(device=device, dtype=dtype)
    raise ValueError(f"Unsupported matrix_type: {matrix_type}")


def _prepare_k_indices(
    nfft: int,
    *,
    k_indices: Optional[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if k_indices is None:
        return torch.arange((nfft // 2) + 1, dtype=dtype, device=device)

    if k_indices.ndim != 1:
        raise ValueError(f"k_indices must be rank-1, got shape {tuple(k_indices.shape)}")

    if not torch.is_floating_point(k_indices):
        k = k_indices.to(dtype=torch.int64, device=device)
    else:
        k = k_indices.to(device=device)
        if not torch.allclose(k, torch.round(k)):
            raise ValueError("k_indices must contain integer-valued bins")
        k = torch.round(k).to(dtype=torch.int64)

    if torch.any(k < 0) or torch.any(k > (nfft // 2)):
        raise ValueError("k_indices must be in [0, nfft//2]")

    return k.to(dtype=dtype)


def transfer_function(
    *,
    sr: float,
    nfft: int,
    delay_samples: Iterable[int],
    gains: torch.Tensor,
    U: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    k_indices: Optional[torch.Tensor] = None,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Compute H(w)=c^T(I-D(w)U G)^-1 D(w)b at selected DFT bins."""
    if nfft <= 0:
        raise ValueError("nfft must be positive")

    n = U.shape[0]
    if U.shape != (n, n):
        raise ValueError(f"Expected square U, got {tuple(U.shape)}")

    delay = torch.as_tensor(list(delay_samples), dtype=gains.dtype, device=U.device)
    if delay.numel() != n:
        raise ValueError(f"delay length {delay.numel()} must match matrix size {n}")
    if gains.numel() != n or b.numel() != n or c.numel() != n:
        raise ValueError("gains, b, c must have shape (N,)")

    c_dtype = _complex_dtype(U.dtype)
    k = _prepare_k_indices(nfft, k_indices=k_indices, device=U.device, dtype=U.dtype)
    omega = (2.0 * math.pi / float(nfft)) * k
    d = torch.exp(-1j * omega[:, None].to(c_dtype) * delay[None, :].to(c_dtype))

    # F(w) = D(w) * U * G.
    UG = U.to(c_dtype) @ torch.diag(gains.to(c_dtype))
    F = d[:, :, None] * UG[None, :, :]

    eye = torch.eye(n, dtype=c_dtype, device=U.device)[None, :, :]
    A = eye - F
    if eps > 0.0:
        A = A + eps * eye

    # rhs(w) = D(w) * b.
    rhs = (d * b.to(c_dtype)[None, :])[:, :, None]
    x = torch.linalg.solve(A, rhs).squeeze(-1)
    H = torch.sum(c.to(c_dtype)[None, :] * x, dim=-1)
    return H


def spectral_loss_colorless(H: torch.Tensor, *, exclude_dc: bool = True) -> torch.Tensor:
    mag = torch.abs(H)
    if exclude_dc and mag.numel() > 1:
        mag = mag[1:]
    return torch.mean((mag - 1.0) ** 2)


def sparsity_loss_eq18(U: torch.Tensor) -> torch.Tensor:
    """EURASIP Eq. (18)-style sparsity term, normalized to [0,1] for orthogonal U."""
    n = U.shape[0]
    sqrt_n = math.sqrt(float(n))
    numerator = torch.sum(torch.abs(U)) - (float(n) * sqrt_n)
    denominator = float(n) * (1.0 - sqrt_n)
    return numerator / denominator


def compute_losses(
    *,
    H_l: torch.Tensor,
    H_r: torch.Tensor,
    U: torch.Tensor,
    alpha_density: float = 0.0,
) -> Dict[str, torch.Tensor]:
    spectral_l = spectral_loss_colorless(H_l, exclude_dc=True)
    spectral_r = spectral_loss_colorless(H_r, exclude_dc=True)
    spectral = 0.5 * (spectral_l + spectral_r)
    sparsity = sparsity_loss_eq18(U)
    total = spectral + float(alpha_density) * sparsity
    return {
        "total": total,
        "spectral": spectral,
        "sparsity": sparsity,
    }


@dataclass
class OptimizeResult:
    matrix_type: str
    u: torch.Tensor
    U: torch.Tensor
    b: torch.Tensor
    c_l: torch.Tensor
    c_r: torch.Tensor
    gains: torch.Tensor
    losses: Dict[str, float]
    history: Dict[str, List[float]]


def _sample_k_indices(
    *,
    nfft: int,
    freq_bins_per_step: int,
    generator: torch.Generator,
    device: torch.device,
) -> Optional[torch.Tensor]:
    max_bins = nfft // 2
    if max_bins <= 1 or freq_bins_per_step <= 0 or freq_bins_per_step >= max_bins:
        return None

    perm = torch.randperm(max_bins, generator=generator, device=device)
    # Shift to exclude DC (k=0), using bins in [1, nfft//2].
    return (perm[:freq_bins_per_step] + 1).to(dtype=torch.int64)


def optimize_householder(
    *,
    sr: float = 48000.0,
    nfft: int = 2048,
    delay_samples: Iterable[int] = (1499, 2377, 3217, 4421),
    rt60: float = 2.8,
    matrix_type: str = "householder",
    steps: int = 800,
    lr: float = 0.03,
    alpha_density: float = 0.0,
    learn_io: bool = False,
    freq_bins_per_step: int = 256,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    log_every: int = 100,
) -> OptimizeResult:
    seed_i = int(seed)
    random.seed(seed_i)
    np.random.seed(seed_i)
    torch.manual_seed(seed_i)

    device = torch.device("cpu") if device is None else torch.device(device)
    rng = torch.Generator(device=device)
    rng.manual_seed(seed_i)

    delay = torch.as_tensor(list(delay_samples), dtype=dtype, device=device)
    n = int(delay.numel())
    if n != 4:
        raise ValueError(f"This tiny model expects N=4, got N={n}")

    gains = gains_from_rt60(delay.tolist(), rt60, sr, dtype=dtype).to(device)
    matrix_kind = matrix_type.lower()
    if matrix_kind not in {"householder", "hadamard"}:
        raise ValueError("matrix_type must be 'householder' or 'hadamard'")

    b_default, c_l_default, c_r_default = default_io_vectors(n, device=device, dtype=dtype)
    b_param = b_default.detach().clone().requires_grad_(learn_io)
    c_l_param = c_l_default.detach().clone().requires_grad_(learn_io)
    c_r_param = c_r_default.detach().clone().requires_grad_(learn_io)

    if matrix_kind == "householder":
        v = torch.randn(n, dtype=dtype, device=device, requires_grad=True)
        params = [v]
    else:
        v = torch.full((n,), 0.5, dtype=dtype, device=device)
        params = []

    if learn_io:
        params += [b_param, c_l_param, c_r_param]

    optimizer = torch.optim.Adam(params, lr=lr) if params else None

    history: Dict[str, List[float]] = {
        "step": [],
        "total": [],
        "spectral": [],
        "sparsity": [],
    }

    total_steps = max(int(steps), 1)
    for step in range(total_steps):
        if optimizer is not None:
            optimizer.zero_grad()

        u = (
            unit_vector_from_raw(v)
            if matrix_kind == "householder"
            else torch.full((n,), 0.5, dtype=dtype, device=device)
        )
        U = matrix_from_type(matrix_kind, u=u, device=device, dtype=dtype)

        b = b_param if learn_io else b_default
        c_l = c_l_param if learn_io else c_l_default
        c_r = c_r_param if learn_io else c_r_default

        sampled_k = _sample_k_indices(
            nfft=nfft,
            freq_bins_per_step=int(freq_bins_per_step),
            generator=rng,
            device=device,
        )

        H_l = transfer_function(
            sr=sr,
            nfft=nfft,
            delay_samples=delay.tolist(),
            gains=gains,
            U=U,
            b=b,
            c=c_l,
            k_indices=sampled_k,
        )
        H_r = transfer_function(
            sr=sr,
            nfft=nfft,
            delay_samples=delay.tolist(),
            gains=gains,
            U=U,
            b=b,
            c=c_r,
            k_indices=sampled_k,
        )

        losses = compute_losses(
            H_l=H_l,
            H_r=H_r,
            U=U,
            alpha_density=alpha_density,
        )

        if optimizer is not None:
            losses["total"].backward()
            optimizer.step()

        history["step"].append(float(step))
        history["total"].append(float(losses["total"].detach().cpu()))
        history["spectral"].append(float(losses["spectral"].detach().cpu()))
        history["sparsity"].append(float(losses["sparsity"].detach().cpu()))

        if log_every > 0 and (step % log_every == 0 or step == total_steps - 1):
            print(
                f"step={step:04d} total={history['total'][-1]:.6e} "
                f"spectral={history['spectral'][-1]:.6e} sparsity={history['sparsity'][-1]:.6e}"
            )

    final_u = (
        unit_vector_from_raw(v.detach())
        if matrix_kind == "householder"
        else torch.full((n,), 0.5, dtype=dtype, device=device)
    )
    final_U = matrix_from_type(matrix_kind, u=final_u, device=device, dtype=dtype)
    final_b = b_param.detach() if learn_io else b_default
    final_c_l = c_l_param.detach() if learn_io else c_l_default
    final_c_r = c_r_param.detach() if learn_io else c_r_default

    final_H_l = transfer_function(
        sr=sr,
        nfft=nfft,
        delay_samples=delay.tolist(),
        gains=gains,
        U=final_U,
        b=final_b,
        c=final_c_l,
    )
    final_H_r = transfer_function(
        sr=sr,
        nfft=nfft,
        delay_samples=delay.tolist(),
        gains=gains,
        U=final_U,
        b=final_b,
        c=final_c_r,
    )
    final_losses_t = compute_losses(
        H_l=final_H_l,
        H_r=final_H_r,
        U=final_U,
        alpha_density=alpha_density,
    )
    final_losses = {k: float(v.detach().cpu()) for k, v in final_losses_t.items()}

    return OptimizeResult(
        matrix_type=matrix_kind,
        u=final_u.detach().cpu(),
        U=final_U.detach().cpu(),
        b=final_b.detach().cpu(),
        c_l=final_c_l.detach().cpu(),
        c_r=final_c_r.detach().cpu(),
        gains=gains.detach().cpu(),
        losses=final_losses,
        history=history,
    )
