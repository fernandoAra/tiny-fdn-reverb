# tiny-fdn-reverb

Small-order FDN reverb (N=4–8) that tames metallic ring via echo-density targeting and AM-safe unitary modulation.

## Formats

All plugins in this collection come in the following plug-in formats:

* [LV2]
* [VST2]
* [VST3]
* [CLAP]

## Compiling

Make sure you have installed the required build tools and libraries (see
section "Prerequisites" below) and then clone this repository (including
sub-modules) and simply run `make` in the project's root directory:

```con
git clone --recursive https://github.com:fernandoAra/tiny-fdn-reverb
cd tiny-fdn-reverb
make
```

## Installation

To install all plugin formats to their appropriate system-wide location, run
the following command (root priviledges may be required):

```con
make install
```

The makefiles support the usual `PREFIX` and `DESTDIR` variables to change the
installation prefix and set an installation root directory (defaulty: empty).
`PREFIX` defaults to `/usr/local`, but on macOS and Windows it is not used,
since the system-wide installation directories for plugins are fixed.

Use make's `-n` option to see where the plugins would be installed without
actually installing them.

You can also set the installation directory for each plugin format with a
dedicated makefile variable.

* LV2: `LV2_DIR` (`<prefix>/lib/lv2`)
* VST2: `VST2_DIR` (`<prefix>/lib/vst`)
* VST3: `VST3_DIR` (`<prefix>/lib/vst3`)
* CLAP: `CLAP_DIR` (`<prefix>/lib/clap`)

Example:

```con
make DESTDIR=/tmp/build-root PREFIX=/usr VST2_DIR=/usr/lib/lxvst install
```

To install the plugins only for your current user account, run:

```con
make install-user
```

Again, you can also set the installation directory for each plugin format with
a dedicated makefile variable.

* LV2: `USER_LV2_DIR` (`$HOME/.lv2`)
* VST2: `USER_VST2_DIR` (`$HOME/.vst`)
* VST3: `USER_VST3_DIR` (`$HOME/.vst3`)
* CLAP: `USER_CLAP_DIR` (`$HOME/.clap`)

*Note: The given default values for all of the above listed environment
variables differ depending on the target OS.*

## DiffFDN Offline Presets

Offline optimization uses PyTorch in `eval/difffdn/` and exports JSON presets.
The plugin itself stays torch-free and consumes generated C++ preset data from
`plugins/tiny-fdn-reverb/DiffFdnPresets.hpp`.

Reproducible sequence:

```con
# 1) Generate learned presets (example: Prime + Spread @ 48 kHz)
python3 eval/difffdn/optimize_householder.py \
  --config-id householder_prime_rt60_2p8_ad005 \
  --matrix-type householder \
  --sr 48000 --delay-samples "1499,2377,3217,4421" \
  --rt60 2.8 --alpha-density 0.05 --steps 800 --seed 1234 \
  --out-dir eval/out/presets

python3 eval/difffdn/optimize_householder.py \
  --config-id householder_spread_rt60_2p8_ad005 \
  --matrix-type householder \
  --sr 48000 --delay-samples "1200,1800,2400,3000" \
  --rt60 2.8 --alpha-density 0.05 --steps 800 --seed 1234 \
  --out-dir eval/out/presets

# 2) Export embedded C++ preset header consumed by the plugin
python3 eval/difffdn/export_cpp_header.py \
  --preset eval/out/presets/householder_prime_rt60_2p8_ad005.json \
  --preset eval/out/presets/householder_spread_rt60_2p8_ad005.json \
  --out-header plugins/tiny-fdn-reverb/DiffFdnPresets.hpp
```

## Defense Demo Bundle

Use the canonical wrapper to regenerate a defense-ready bundle from a chosen
learned preset and (optionally) sync that same preset into the plugin's
embedded C++ header:

```con
bash eval/scripts/build_demo_bundle.sh \
  --preset eval/out/presets/householder_prime_rt60_2p8_paper_seed0.json \
  --sync-plugin-header
```

Outputs are copied to:

* `eval/out/demo_bundle/<preset-slug>/`

This flow runs `compare_fixed_vs_diff.py` with strict LUFS matching
(`--require-lufs`) and includes matched demo WAVs as the primary
fairness-controlled listening evidence. Live plugin switching is kept as a
secondary realtime proof.

## Paper Assets

The written piece uses exported paper assets under `paper_assets/`. The main
locations are:

* Figures: `paper_assets/figures/`
* Tables: `paper_assets/tables/`
* Multiseed figures: `paper_assets/figures_multiseed/`
* Multiseed tables: `paper_assets/tables_multiseed/`

Chapter 2 explanatory figures currently live in `paper_assets/figures/` and
include:

* `waveform_to_delayed_samples.png` (Section 2.1 explainer)
* `fig_fdn_4x4_block_diagram.{svg,png,pdf}` (Section 2.2)
* `fig_unilossless_feedback_mixing_4x4.{svg,png,pdf}` (Section 2.3)
* `fig_homogeneous_decay_rt60_explanation.{svg,png,pdf}` (Section 2.4)
* `fig1_feedback_comb_mag.png` and `fig2_allpass_impulse_comparison.png`

Chapter 5 results figures and tables are also exported into `paper_assets/`,
including:

* `paper_assets/figures/fig_mag_fixed_vs_diff*.png`
* `paper_assets/figures/fig_irfft_fixed_vs_diff*.png`
* `paper_assets/figures/fig_edc_fixed_vs_diff*.png`
* `paper_assets/figures/fig_diffusion_fixed_vs_diff*.png`
* `paper_assets/figures/fig_metrics_fixed_vs_diff*.png`
* `paper_assets/tables/table_fixed_vs_diff.csv`
* `paper_assets/figures_multiseed/fig_multiseed_{metrics,deltas,diffusion,echo_density}_<config>.png`
* `paper_assets/tables_multiseed/{table,deltas,stats}_multiseed_<config>.{csv,json}`

Use `bash eval/scripts/export_paper_assets.sh` to refresh the Chapter 5 export
bundle. For a fuller reproduction log and asset inventory, see
`paper_assets/README.md`.


## Prerequisites

* The GCC C++ compiler, library and the usual associated software build tools
  (GNU `make`, etc.).

  Debian / Ubuntu users should install the `build-essential` package
  to get these, Arch users the `base-devel` package group.

* [pkgconf]

The [LV2] and [VST2] (vestige) headers are included in the
[DPF] framework, which is integrated as a Git sub-module. These need not be
installed separately to build the software in the respective plug-in formats.


## Author

This software was created by *Fernando de Souza Araujo*.


## Acknowledgements

This project is built using the DISTRHO Plugin Framework ([DPF]) and set up
with the [cookiecutter-dpf-effect] project template.


[cookiecutter-dpf-effect]: https://github.com/SpotlightKid/cookiecutter-dpf-effect
[DPF]: https://github.com/DISTRHO/DPF
[LV2]: http://lv2plug.in/
[pkgconf]: https://github.com/pkgconf/pkgconf
[VST2/3]: https://en.wikipedia.org/wiki/Virtual_Studio_Technology
[CLAP]: https://cleveraudio.org/
