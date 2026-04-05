# Paper Assets Reproduction

## Regenerate from scratch

- Install Python deps: `python3 -m pip install torch numpy scipy matplotlib`
- Generate a paper-like learned preset (offline, canonical defaults = unity + lossless-core training):  
  `python3 eval/difffdn/optimize_householder.py --config-id householder_prime_rt60_2p8_paper_seed0 --matrix-type householder --fs 48000 --nfft 2048 --M 480000 --batch 2000 --epochs 3 --lr 1e-3 --alpha-sparsity 0.05 --delay-samples 1499,2377,3217,4421 --rt60 2.8 --spectral-mode unity --paper-band-enable --paper-band-min-hz 50 --paper-band-max-hz 12000 --train-lossless --learn-io --seed 0 --out-dir eval/out/presets`
- Run fixed-vs-diff comparison (analytic + IR FFT + EDC + diffusion + metrics + CSV):  
  `python3 eval/scripts/compare_fixed_vs_diff.py --preset eval/out/presets/householder_prime_rt60_2p8_paper_seed0.json --scope all --sanity-check`
- Run reproducible multi-seed sweep (optimizer + compare + aggregate, default learns IO):  
  `python3 eval/scripts/run_multiseed_fixed_vs_diff.py --config-id householder_prime_rt60_2p8_paper --seeds 0,1,2,3,4 --scope all --restarts 8 --select-metric best_val_spectral_dev_db_50_12k --spectral-mode unity --train-lossless --paper-band-enable --paper-band-min-hz 50 --paper-band-max-hz 12000`
- Optional u-only baseline sweep (disable learned IO explicitly):  
  `python3 eval/scripts/run_multiseed_fixed_vs_diff.py --config-id householder_prime_rt60_2p8_paper --seeds 0,1,2,3,4 --scope all --spectral-mode unity --train-lossless --no-learn-io`
- Export paper-ready files:  
  `bash eval/scripts/export_paper_assets.sh`
- Regenerate Chapter 2 theory figures directly from the notebook source:  
  `python theoretical_plots/run_notebook_figures.py`
- Export with optional multiseed bundle:  
  `bash eval/scripts/export_paper_assets.sh "" householder_prime_rt60_2p8_paper`

Notes:
- Training is ML-style autodiff optimization of analytic transfer-function parameters (`u` and optional `b,cL,cR`): minibatches of frequency indices are sampled directly from the paper band (50–12k Hz), losses are backpropagated, and Adam updates parameters offline.
- `compare_fixed_vs_diff.py` forces LTI render settings for analytic-vs-IR consistency (`mod_depth=0`, `detune=0`, `damp_hz=1e9`).
- FFT size is automatically expanded to avoid truncating long IRs (`nfft_used = max(requested_nfft, aligned_ir_length)`).
- Canonical training defaults: `spectral_mode=unity` and `training_mode=lossless-core` (`gamma_train=1.0`, while `gamma_used` is still derived from `rt60` for runtime metadata/preset rendering).
- Paper-band colorless metrics are reported in `50 Hz – 12 kHz`:
  `spectral_loss_like_50_12k` and `spectral_dev_db_50_12k`.
- RT60/ringiness/echo-density/kurtosis are evaluation-only metrics (not part of the training objective).
- Optional ablation mode: pass `--optimize-with-decay` to optimizer/multiseed script.

## Demo playback exports

- `compare_fixed_vs_diff.py` can export offline demo WAVs (`--export-demo-wavs`) by convolving a deterministic dry stimulus with each rendered IR, then level-matching the wet outputs for A/B playback.
- Default matching requests LUFS (`--level-match-method lufs`). If [`pyloudnorm`](https://github.com/csteinmetz1/pyloudnorm) is installed, the script uses BS.1770 integrated loudness; otherwise it prints a warning and falls back to RMS matching in the `50–300 ms` window.
- Add `--require-lufs` if you want the demo export to fail instead of falling back when LUFS cannot be computed.
- Optional dependency for LUFS matching only: `python3 -m pip install pyloudnorm`

### Defense demo bundle (canonical flow)

Use one wrapper command to regenerate the fixed-vs-diff figures, matched demo
WAVs, and a bundle manifest in one place:

`bash eval/scripts/build_demo_bundle.sh --preset eval/out/presets/householder_prime_rt60_2p8_paper_seed0.json --sync-plugin-header`

Bundle output:

- `eval/out/demo_bundle/<preset-slug>/bundle_manifest.json`
- `eval/out/demo_bundle/<preset-slug>/fixed_vs_diff_run.json`
- `eval/out/demo_bundle/<preset-slug>/demo_manifest.json`
- `eval/out/demo_bundle/<preset-slug>/fixed_vs_diff_mag_overlay.png`
- `eval/out/demo_bundle/<preset-slug>/fixed_vs_diff_irfft_overlay.png`
- `eval/out/demo_bundle/<preset-slug>/fixed_vs_diff_diffusion.png`
- `eval/out/demo_bundle/<preset-slug>/fixed_vs_diff_metrics_bars.png`
- `eval/out/demo_bundle/<preset-slug>/summary_fixed_vs_diff.csv`
- `eval/out/demo_bundle/<preset-slug>/demo_<slug>_*_{raw|matched}.wav`

For demo fairness, matched demo WAVs are treated as the primary evidence, while
live plugin toggling is secondary realtime proof.

## Pinned external references

- diff-fdn-colorless commit:  
  `49a9737fb320de6cea7dc85e990eaef8c8cfba0c`  
  https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c
- flamo Householder reference commit:  
  `4c8097d4feda76132691bb2a3e465ebcba11dcea`  
  https://github.com/gdalsanto/flamo/blob/4c8097d4feda76132691bb2a3e465ebcba11dcea/flamo/processor/dsp.py#L621-L725

## Output locations

- Chapter 2 explanatory figures: `paper_assets/figures/`
- Chapter 5 result figures: `paper_assets/figures/`
- Chapter 5 result tables: `paper_assets/tables/`
- Chapter 5 multiseed figures: `paper_assets/figures_multiseed/`
- Chapter 5 multiseed tables: `paper_assets/tables_multiseed/`

## Chapter 2 explanatory figures

- `paper_assets/figures/fig1_feedback_comb_mag.png` (feedback comb magnitude response)
- `paper_assets/figures/fig2_allpass_impulse_comparison.png` (comb-vs-allpass impulse-response comparison)
- `paper_assets/figures/waveform_to_delayed_samples.png` (waveform -> samples -> delayed copies explainer for Section 2.1)
- `paper_assets/figures/fig_fdn_4x4_block_diagram.{svg,png,pdf}` (Section 2.2 FDN signal-flow block diagram)
- `paper_assets/figures/fig_unilossless_feedback_mixing_4x4.{svg,png,pdf}` (Section 2.3 unilossless feedback mixing explainer)
- `paper_assets/figures/fig_homogeneous_decay_rt60_explanation.{svg,png,pdf}` (Section 2.4 homogeneous decay / RT60 explainer)

The last three Chapter 2 figures are generated directly by:

- `python3 eval/scripts/generate_fdn_block_diagram.py`
- `python3 eval/scripts/generate_unilossless_feedback_mixing_figure.py`
- `python3 eval/scripts/generate_homogeneous_decay_rt60_explanation.py`

## Chapter 5 result figures and tables

- `paper_assets/figures/fig_mag_fixed_vs_diff.png` (analytic transfer overlay + delta)
- `paper_assets/figures/fig_irfft_fixed_vs_diff.png` (IR FFT windowed overlay + delta)
- `paper_assets/figures/fig_edc_fixed_vs_diff.png` (EDC overlay / decay comparison)
- `paper_assets/figures/fig_diffusion_fixed_vs_diff.png` (kurtosis diffusion curves)
- `paper_assets/figures/fig_metrics_fixed_vs_diff.png` (bar summary)
- `paper_assets/tables/table_fixed_vs_diff.csv`

Single-run paper/result variants in the same directory follow the same naming
scheme, for example:

- `paper_assets/figures/fig_mag_fixed_vs_diff_<slug>.png`
- `paper_assets/figures/fig_irfft_fixed_vs_diff_<slug>.png`
- `paper_assets/figures/fig_edc_fixed_vs_diff_<slug>.png`
- `paper_assets/figures/fig_diffusion_fixed_vs_diff_<slug>.png`
- `paper_assets/figures/fig_metrics_fixed_vs_diff_<slug>.png`
- `paper_assets/figures/fig_edc_representative_<config>.png`
- `paper_assets/figures/fig_irfft_representative_<config>.png`

## Chapter 5 multiseed aggregates

- `paper_assets/figures_multiseed/fig_multiseed_metrics_<config>.png`
- `paper_assets/figures_multiseed/fig_multiseed_deltas_<config>.png`
- `paper_assets/figures_multiseed/fig_multiseed_diffusion_<config>.png`
- `paper_assets/figures_multiseed/fig_multiseed_echo_density_<config>.png`
- `paper_assets/tables_multiseed/table_multiseed_<config>.csv`
- `paper_assets/tables_multiseed/deltas_multiseed_<config>.csv`
- `paper_assets/tables_multiseed/stats_multiseed_<config>.json`
