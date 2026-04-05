#!/usr/bin/env python3
"""Quick sanity check for Eq.(18)-style sparsity loss behavior.

Expectation for our definition:
- sparse/permutation-like orthogonal matrices -> higher loss
- dense orthogonal matrices -> lower loss
This matches "penalize sparse, encourage dense mixing".
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from difffdn.difffdn_tiny import hadamard4_matrix, sparsity_loss_eq18  # type: ignore
else:
    from .difffdn_tiny import hadamard4_matrix, sparsity_loss_eq18


def main() -> None:
    dtype = torch.float64
    u_dense = hadamard4_matrix(dtype=dtype)   # dense orthogonal
    u_sparse = torch.eye(4, dtype=dtype)      # sparse/permutation-like orthogonal

    l_dense = float(sparsity_loss_eq18(u_dense))
    l_sparse = float(sparsity_loss_eq18(u_sparse))

    print(f"dense(Hadamard) sparsity_loss={l_dense:.6f}")
    print(f"sparse(identity) sparsity_loss={l_sparse:.6f}")

    if not (l_sparse > l_dense):
        raise SystemExit(
            "Unexpected sparsity loss ordering: sparse should be worse (higher) than dense"
        )
    print("OK: sparse > dense (loss encourages dense mixing)")


if __name__ == "__main__":
    main()
