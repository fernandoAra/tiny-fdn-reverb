"""Minimal frequency-sampled differentiable tiny-FDN model (N=4).

SOURCE:
- diff-fdn-colorless reference implementation (Dal Santo et al.), pinned commit:
  https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c
- Adapted ideas here: frequency-domain transfer evaluation, sparse-bin training,
  and Eq.(18)-style matrix sparsity regularization for tiny-FDN optimization.
"""

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
    gamma = gamma_from_rt60(sr, rt60)
    return gains_from_gamma(delay_samples, gamma, dtype=dtype)


def gamma_from_rt60(fs: float, rt60: float) -> float:
    fs_safe = max(float(fs), 1.0)
    rt60_safe = max(float(rt60), 1e-3)
    gamma = 10.0 ** (-3.0 / (fs_safe * rt60_safe))
    return min(max(gamma, 1e-6), 0.999999)


def rt60_from_gamma(fs: float, gamma: float) -> float:
    fs_safe = max(float(fs), 1.0)
    gamma_safe = min(max(float(gamma), 1e-6), 0.999999)
    return float(math.log(1e-3) / (fs_safe * math.log(gamma_safe)))


def gains_from_gamma(
    delay_samples: Iterable[int], gamma: float, *, dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    delay = torch.as_tensor(list(delay_samples), dtype=dtype)
    gamma_clamped = min(max(float(gamma), 1e-6), 0.999999)
    return torch.pow(torch.full_like(delay, gamma_clamped), delay)


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


def normalize_l2(vec: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return vec normalized to unit L2 norm."""
    norm = torch.linalg.vector_norm(vec)
    return vec / torch.clamp(norm, min=eps)


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
    max_k: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if k_indices is None:
        return torch.arange(max_k + 1, dtype=dtype, device=device)

    if k_indices.ndim != 1:
        raise ValueError(f"k_indices must be rank-1, got shape {tuple(k_indices.shape)}")

    if not torch.is_floating_point(k_indices):
        k = k_indices.to(dtype=torch.int64, device=device)
    else:
        k = k_indices.to(device=device)
        if not torch.allclose(k, torch.round(k)):
            raise ValueError("k_indices must contain integer-valued bins")
        k = torch.round(k).to(dtype=torch.int64)

    if torch.any(k < 0) or torch.any(k > max_k):
        raise ValueError(f"k_indices must be in [0, {max_k}]")

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
    freq_grid_size: Optional[int] = None,
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
    if freq_grid_size is None:
        max_k = nfft // 2
    else:
        max_k = max(int(freq_grid_size) - 1, 1)
    k = _prepare_k_indices(nfft, k_indices=k_indices, max_k=max_k, device=U.device, dtype=U.dtype)
    if freq_grid_size is None:
        omega = (2.0 * math.pi / float(nfft)) * k
    else:
        # Paper-like dense frequency grid over [0, pi].
        omega = math.pi * k / float(max(freq_grid_size - 1, 1))
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


def spectral_loss_unity(
    H: torch.Tensor, *, exclude_dc: bool = True
) -> torch.Tensor:
    """Paper-style unity-target spectral loss: mean((|H|-1)^2), optionally excluding DC."""
    mag = torch.abs(H)
    if exclude_dc and mag.numel() > 1:
        mag = mag[1:]
    return torch.mean((mag - 1.0) ** 2)


def spectral_loss_mean_normalized(
    H: torch.Tensor, *, exclude_dc: bool = True, eps: float = 1e-12
) -> torch.Tensor:
    """Optional legacy variant: mean-normalized magnitude should stay near 1."""
    mag = torch.abs(H)
    if exclude_dc and mag.numel() > 1:
        mag = mag[1:]
    mag_mean = torch.mean(mag)
    rel = mag / torch.clamp(mag_mean, min=eps)
    return torch.mean((rel - 1.0) ** 2)


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
    spectral_mode: str = "unity",
) -> Dict[str, torch.Tensor]:
    mode = spectral_mode.lower()
    if mode == "unity":
        spectral_l = spectral_loss_unity(H_l, exclude_dc=True)
        spectral_r = spectral_loss_unity(H_r, exclude_dc=True)
    elif mode == "mean":
        spectral_l = spectral_loss_mean_normalized(H_l, exclude_dc=True)
        spectral_r = spectral_loss_mean_normalized(H_r, exclude_dc=True)
    else:
        raise ValueError(f"Unsupported spectral_mode: {spectral_mode}")
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
    gamma_used: float
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
    max_k: int,
    freq_bins_per_step: int,
    generator: torch.Generator,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if max_k <= 1 or freq_bins_per_step <= 0 or freq_bins_per_step >= max_k:
        return None

    perm = torch.randperm(max_k, generator=generator, device=device)
    # Shift to exclude DC (k=0), using bins in [1, max_k].
    return (perm[:freq_bins_per_step] + 1).to(dtype=torch.int64)


def evaluate_transfer_losses(
    *,
    sr: float,
    nfft: int,
    delay_samples: Iterable[int],
    gains: torch.Tensor,
    matrix_type: str,
    u: Optional[torch.Tensor],
    b: torch.Tensor,
    c_l: torch.Tensor,
    c_r: torch.Tensor,
    alpha_density: float = 0.0,
    spectral_mode: str = "unity",
    k_indices: Optional[torch.Tensor] = None,
    freq_grid_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    matrix_kind = matrix_type.lower()
    if matrix_kind == "householder":
        if u is None:
            raise ValueError("Householder evaluation requires u")
        u_unit = normalize_l2(u)
    elif matrix_kind == "hadamard":
        u_unit = None
    else:
        raise ValueError("matrix_type must be 'householder' or 'hadamard'")

    U = matrix_from_type(matrix_kind, u=u_unit, device=gains.device, dtype=gains.dtype)
    H_l = transfer_function(
        sr=sr,
        nfft=nfft,
        delay_samples=delay_samples,
        gains=gains,
        U=U,
        b=b,
        c=c_l,
        k_indices=k_indices,
        freq_grid_size=freq_grid_size,
    )
    H_r = transfer_function(
        sr=sr,
        nfft=nfft,
        delay_samples=delay_samples,
        gains=gains,
        U=U,
        b=b,
        c=c_r,
        k_indices=k_indices,
        freq_grid_size=freq_grid_size,
    )
    losses = compute_losses(
        H_l=H_l,
        H_r=H_r,
        U=U,
        alpha_density=alpha_density,
        spectral_mode=spectral_mode,
    )
    return {
        "total": losses["total"],
        "spectral": losses["spectral"],
        "sparsity": losses["sparsity"],
        "U": U,
        "u": u_unit if u_unit is not None else torch.zeros_like(b),
        "H_l": H_l,
        "H_r": H_r,
    }


def optimize_householder(
    *,
    sr: float = 48000.0,
    nfft: int = 2048,
    delay_samples: Iterable[int] = (1499, 2377, 3217, 4421),
    rt60: float = 2.8,
    matrix_type: str = "householder",
    steps: Optional[int] = None,
    epochs: int = 3,
    M: int = 480000,
    batch_size: int = 2000,
    lr: float = 1e-3,
    gamma: Optional[float] = None,
    train_lossless: bool = False,
    alpha_density: float = 0.0,
    learn_io: bool = False,
    freq_bins_per_step: int = 2000,
    spectral_mode: str = "unity",
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

    gamma_used = gamma_from_rt60(sr, rt60) if gamma is None else float(gamma)
    if train_lossless:
        # Paper-style "colorless core" training: optimize with unit loop gain.
        gains = torch.ones((n,), dtype=dtype, device=device)
        gamma_train = 1.0
    else:
        gains = gains_from_gamma(delay.tolist(), gamma_used, dtype=dtype).to(device)
        gamma_train = gamma_used
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
        "epoch": [],
        "total": [],
        "spectral": [],
        "sparsity": [],
        "u_norm": [],
        "b_norm": [],
        "cL_norm": [],
        "cR_norm": [],
    }

    steps_per_epoch = max(1, int(math.ceil(float(max(M, 1)) / float(max(batch_size, 1)))))
    total_steps = max(int(steps) if steps is not None else (max(int(epochs), 1) * steps_per_epoch), 1)
    max_k = max(int(M) - 1, 1)
    for step in range(total_steps):
        if optimizer is not None:
            optimizer.zero_grad()

        u = (
            unit_vector_from_raw(v)
            if matrix_kind == "householder"
            else torch.full((n,), 0.5, dtype=dtype, device=device)
        )
        if learn_io:
            b = normalize_l2(b_param)
            c_l = normalize_l2(c_l_param)
            c_r = normalize_l2(c_r_param)
        else:
            b = b_default
            c_l = c_l_default
            c_r = c_r_default

        sampled_k = _sample_k_indices(
            max_k=max_k,
            freq_bins_per_step=int(freq_bins_per_step),
            generator=rng,
            device=device,
        )

        losses = evaluate_transfer_losses(
            sr=sr,
            nfft=nfft,
            delay_samples=delay.tolist(),
            gains=gains,
            matrix_type=matrix_kind,
            u=u,
            b=b,
            c_l=c_l,
            c_r=c_r,
            alpha_density=alpha_density,
            spectral_mode=spectral_mode,
            k_indices=sampled_k,
            freq_grid_size=max(int(M), 2),
        )

        if optimizer is not None:
            losses["total"].backward()
            optimizer.step()

        history["step"].append(float(step))
        history["epoch"].append(float(step // steps_per_epoch))
        history["total"].append(float(losses["total"].detach().cpu()))
        history["spectral"].append(float(losses["spectral"].detach().cpu()))
        history["sparsity"].append(float(losses["sparsity"].detach().cpu()))
        history["u_norm"].append(float(torch.linalg.vector_norm(u).detach().cpu()))
        history["b_norm"].append(float(torch.linalg.vector_norm(b).detach().cpu()))
        history["cL_norm"].append(float(torch.linalg.vector_norm(c_l).detach().cpu()))
        history["cR_norm"].append(float(torch.linalg.vector_norm(c_r).detach().cpu()))

        if log_every > 0 and (step % log_every == 0 or step == total_steps - 1):
            epoch = step // steps_per_epoch
            print(
                f"step={step:05d} epoch={epoch:03d} total={history['total'][-1]:.6e} "
                f"spectral={history['spectral'][-1]:.6e} sparsity={history['sparsity'][-1]:.6e} "
                f"||u||={history['u_norm'][-1]:.3f} ||b||={history['b_norm'][-1]:.3f} "
                f"||cL||={history['cL_norm'][-1]:.3f} ||cR||={history['cR_norm'][-1]:.3f}"
            )

    final_u = (
        unit_vector_from_raw(v.detach())
        if matrix_kind == "householder"
        else torch.full((n,), 0.5, dtype=dtype, device=device)
    )
    final_b = normalize_l2(b_param.detach()) if learn_io else b_default
    final_c_l = normalize_l2(c_l_param.detach()) if learn_io else c_l_default
    final_c_r = normalize_l2(c_r_param.detach()) if learn_io else c_r_default

    final_eval = evaluate_transfer_losses(
        sr=sr,
        nfft=nfft,
        delay_samples=delay.tolist(),
        gains=gains,
        matrix_type=matrix_kind,
        u=final_u,
        b=final_b,
        c_l=final_c_l,
        c_r=final_c_r,
        alpha_density=alpha_density,
        spectral_mode=spectral_mode,
        freq_grid_size=max(int(M), 2),
    )
    final_U = final_eval["U"]
    final_losses = {
        "total": float(final_eval["total"].detach().cpu()),
        "spectral": float(final_eval["spectral"].detach().cpu()),
        "sparsity": float(final_eval["sparsity"].detach().cpu()),
    }

    return OptimizeResult(
        matrix_type=matrix_kind,
        gamma_used=float(gamma_train),
        u=final_u.detach().cpu(),
        U=final_U.detach().cpu(),
        b=final_b.detach().cpu(),
        c_l=final_c_l.detach().cpu(),
        c_r=final_c_r.detach().cpu(),
        gains=gains.detach().cpu(),
        losses=final_losses,
        history=history,
    )
