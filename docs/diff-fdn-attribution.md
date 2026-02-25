# DiffFDN Attribution Map

This document records what parts of this repository are inspired by Dal Santo et al.
and what parts are direct code adaptations.

## Scope split

- Inspired by paper/research framing:
  - Frequency-domain transfer-function training loop for tiny FDNs.
  - Colorless spectral objective and sparsity-oriented regularization.
  - Offline differentiable optimization with learned preset export.
- Directly adapted conceptually from referenced code:
  - Householder parameterization and efficient reflection multiply:
    `u = v / ||v||`, `y = x - 2u(u^T x)`.
- Implemented project-specific:
  - DPF plugin integration, UI/host-compatibility behavior, preset lookup path.
  - Evaluation scripts, figure generation, and paper-asset export.

## Component Mapping

| Component | Our file(s) | What we did | Source (paper eq / repo link) |
|---|---|---|---|
| Householder efficient multiply in DSP | `plugins/tiny-fdn-reverb/Plugintiny-fdn-reverb.cpp` (`householderMix4U`) | Implemented realtime-safe `y = x - 2u(u^T x)` (no explicit `U` matrix) for `N=4` | flamo HouseholderMatrix reference: https://github.com/gdalsanto/flamo/blob/4c8097d4feda76132691bb2a3e465ebcba11dcea/flamo/processor/dsp.py#L621-L725 |
| Differentiable Householder helpers (offline) | `eval/difffdn/householder.py` | Implemented raw vector to unit vector map and differentiable Householder apply/build | Same flamo reference above |
| Frequency-domain tiny-FDN transfer model | `eval/difffdn/difffdn_tiny.py` | Implemented `H(w)=c^T(I-DUG)^-1Db`, sparse-bin evaluation path, stereo handling | diff-fdn-colorless commit: https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c |
| Spectral + sparsity loss design | `eval/difffdn/difffdn_tiny.py` | Implemented composite loss with colorless term and Eq.(18)-style matrix sparsity term | Dal Santo 2025 framing (Eq. 16 composite, Eq. 18 sparsity style) + diff-fdn-colorless commit above |
| Paper-like optimizer settings | `eval/difffdn/optimize_householder.py` | Added paper-oriented defaults (`fs=48k`, `M=480000`, `batch=2000`, `epochs`) with **lossless-first** training option (`train_lossless`: optimize with loop gain=1) and optional decay-aware mode | Dal Santo EURASIP 2025 training framing + diff-fdn-colorless commit above |
| Offline optimizer + preset exporter | `eval/difffdn/optimize_householder.py` | Added reproducible CLI optimizer (seeded), JSON preset output, history logs | diff-fdn-colorless workflow inspiration (offline SGD) |
| Learned preset embedding for plugin | `eval/difffdn/export_cpp_header.py`, `plugins/tiny-fdn-reverb/DiffFdnPresets.hpp` | Converted offline JSON presets into constexpr C++ presets for runtime use | Project integration choice (not copied from source repos) |
| Diff runtime routing (u+b+c) | `plugins/tiny-fdn-reverb/Plugintiny-fdn-reverb.cpp`, `plugins/tiny-fdn-reverb/Plugintiny-fdn-reverb.hpp` | Runtime selection between fixed baseline, Diff u-only fallback, and Diff full (`u+b+c`) using offline preset data only | Project-specific integration of paper framing; no training code in plugin |
| Evaluation/verification plots | `eval/scripts/verify_transfer_match.py`, `eval/scripts/compare_fixed_vs_diff.py` | Added offline IR-vs-transfer verification and Fixed-vs-Diff comparison metrics/figures | Paper objective interpretation + project-specific evaluation tooling |
| Diffusion proxy metric | `eval/scripts/compare_fixed_vs_diff.py` | Updated proxy to short-time **RMS-normalized** excess kurtosis with RMS gating to prevent near-silence blowups; lower values interpreted as more diffuse proxy | Project-specific metric implementation aligned with diffusion interpretation |

## Realtime boundary

- The plugin **does not** run SGD or PyTorch.
- Optimization is strictly offline in `eval/difffdn` (inspired by diff-fdn-colorless).
- Runtime plugin only consumes learned preset parameters (`u`, optional `b/c`) from generated JSON/header data.
- No PyTorch dependency is linked into the plugin runtime.
- Realtime injection/output now use a unified active-vector path (`b_active`, `cL_active`, `cR_active`) for fixed, u-only, and full comparisons.
- Compare pipeline forces LTI render conditions (`mod_depth=0`, `detune=0`, effectively no damping) and uses non-truncating FFT sizing for analytic-vs-IR consistency.
