# Paper Assets Reproduction

## Regenerate from scratch

- Install Python deps: `python3 -m pip install torch numpy scipy matplotlib`
- Generate a paper-like learned preset (offline only):  
  `python3 eval/difffdn/optimize_householder.py --config-id householder_prime_rt60_2p8_paper --matrix-type householder --fs 48000 --nfft 2048 --M 480000 --batch 2000 --epochs 3 --lr 1e-3 --alpha-sparsity 0.05 --delay-samples 1499,2377,3217,4421 --rt60 2.8 --learn-io --seed 0 --out-dir eval/out/presets`
- Run fixed-vs-diff comparison (analytic + IR FFT + EDC + diffusion + metrics + CSV):  
  `python3 eval/scripts/compare_fixed_vs_diff.py --preset eval/out/presets/householder_prime_rt60_2p8_paper.json --scope all --sanity-check`
- Export paper-ready files:  
  `bash eval/scripts/export_paper_assets.sh`

## Pinned external references

- diff-fdn-colorless commit:  
  `49a9737fb320de6cea7dc85e990eaef8c8cfba0c`  
  https://github.com/gdalsanto/diff-fdn-colorless/tree/49a9737fb320de6cea7dc85e990eaef8c8cfba0c
- flamo Householder reference commit:  
  `4c8097d4feda76132691bb2a3e465ebcba11dcea`  
  https://github.com/gdalsanto/flamo/blob/4c8097d4feda76132691bb2a3e465ebcba11dcea/flamo/processor/dsp.py#L621-L725

## Output locations

- Figures: `paper_assets/figures/`
- Tables: `paper_assets/tables/`

## Exported files

- `paper_assets/figures/fig_mag_fixed_vs_diff.png` (analytic transfer overlay + delta)
- `paper_assets/figures/fig_irfft_fixed_vs_diff.png` (IR FFT windowed overlay + delta)
- `eval/figs/fixed_vs_diff_edc_overlay.png` (EDC overlay with EDT/T20/T30 fit bands)
- `paper_assets/figures/fig_diffusion_fixed_vs_diff.png` (kurtosis diffusion curves)
- `paper_assets/figures/fig_metrics_fixed_vs_diff.png` (bar summary)
- `paper_assets/tables/table_fixed_vs_diff.csv`
