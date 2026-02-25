# Third-Party References

This project includes concepts adapted from the following external repositories.
No full third-party source tree was vendored into this plugin.

## diff-fdn-colorless (MIT)
- Repository: <https://github.com/gdalsanto/diff-fdn-colorless>
- Pinned commit: `49a9737fb320de6cea7dc85e990eaef8c8cfba0c`
- Commit URL: <https://github.com/gdalsanto/diff-fdn-colorless/commit/49a9737fb320de6cea7dc85e990eaef8c8cfba0c>
- What we adapted:
  - Offline differentiable tiny-FDN flow.
  - Frequency-sampled spectral objective and sparsity-oriented regularization framing.
- What we implemented locally:
  - A project-specific PyTorch optimizer in `eval/difffdn/` and JSON preset export.

## flamo (MIT)
- Repository: <https://github.com/gdalsanto/flamo>
- Pinned commit: `4c8097d4feda76132691bb2a3e465ebcba11dcea`
- Commit URL: <https://github.com/gdalsanto/flamo/commit/4c8097d4feda76132691bb2a3e465ebcba11dcea>
- Referenced section: `HouseholderMatrix`, lines 621–725 at that commit.
- What we adapted:
  - Householder parameterization idea (`u = v / ||v||`) and efficient reflection multiply (`y = x - 2u(u^T x)`).
- What we implemented locally:
  - Our own minimal Householder helpers and DSP integration path for fixed-vs-diff Householder vectors.

## Notes
- These references were used as algorithmic guidance.
- Integration, UI behavior, host-compatibility handling, and plugin-specific code are original adaptations in this repo.
