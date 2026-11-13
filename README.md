# lyapy
Lyman alpha reconstruction code used in [Youngblood et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...824..101Y/abstract) and [other papers](https://ui.adsabs.harvard.edu/public-libraries/o5i7z8i2QXWzZSBDV45wqQ).

Edit and run the script "example_start_file.py" to fit the model to the STIS E140M spectrum of GJ 411 "HD-95735-E140M-odn905010_1.xl_stis". Requirements include astropy, numpy, scipy, matplotlib, pandas, time, corner, emcee, joblib, and copy.

## Differentiable model prototype

An optional JAX implementation of the GJ 411-style forward model lives in
`lyapy.jax_models`. It requires 64-bit mode for scientific parity with the
NumPy/Astropy baseline. The test suite is validated against JAX 0.12.0.1:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

Install the optional dependency and run the reproducible CPU benchmark with:

```bash
python -m pip install -r requirements-jax.txt
python tools/benchmark_jax_model.py
python tools/plot_jax_comparison.py
```

## Inference prototype

NumPyro/NUTS and dynesty adapters share the same parameter configuration,
priors, JAX forward model, and Gaussian likelihood. Install the optional
dependencies and run the validation sequence with:

```bash
python -m pip install -e '.[inference]'
python tools/run_inference_smoke_test.py
python tools/run_stellar_recovery.py
python tools/run_full_recovery.py
python tools/run_gj411_inference.py
python tools/run_gj411_model_comparison.py
```

The smoke test varies two parameters. The recovery scripts then validate five
stellar parameters and the full eight-parameter stellar-plus-ISM model on
synthetic spectra. The final command fits the observed GJ 411 STIS E140M
spectrum and writes posterior, fitted-spectrum, reconstruction, reusable NPZ,
and JSON diagnostic products. The reconstruction includes 68% and 95% profile
bands and the integrated intrinsic Ly-alpha flux posterior. These runs keep
D/H, the inactive second absorber, and the tabulated STIS LSF fixed.
The final model-comparison command measures whether the observed spectrum
supports a self-reversed intrinsic profile relative to a zero-reversal model.
Its documented default prior on reversal depth is uniform from 0 to 1.8; use
`--reversal-max` for an explicit prior-sensitivity run.

### Comparing arbitrary models

The evidence framework accepts two or more named inference configurations and
runs them with matched nested-sampling settings:

```python
from lyapy.inference import ModelHypothesis, run_evidence_comparison

result = run_evidence_comparison(
    data,
    (
        ModelHypothesis("model A", configuration_a),
        ModelHypothesis("model B", configuration_b),
        ModelHypothesis("model C", configuration_c),
    ),
    random_seed=93,
    nlive=500,
    dlogz=0.05,
    sample="rwalk",
    bootstrap=0,
)

print(result.best_model)
print(result.as_dict()["ranking"])
sampler = result.get_run("model A").sampler
```

Each hypothesis may supply a different JAX forward model, likelihood function,
ordered parameter schema, fixed parameter vector, free parameters, and priors.
Each report records those
callables and parameter definitions alongside evidence uncertainties, runtimes,
and Bayes factors relative to the highest-evidence model. The retained samplers
remain available for model-specific posterior diagnostics and plots. Evidence
comparisons require properly normalized likelihoods and scientifically
defensible priors for every model.

## Composable high-resolution forward model

Optional JAX components support single or multiple Gaussian and Voigt emission
profiles, relative component weights, total post-reversal intrinsic flux,
Gaussian-optical-depth or legacy self-reversal, and one or more H I plus D I
absorption components. The composable ISM model evaluates untruncated Voigt
damping wings directly on the physical-model grid; the historical ±2 Å
compatibility model remains available only for reproducing legacy results. A
static observation operator evaluates the physical model on an oversampled
grid, applies ISM absorption, convolves with a tabulated LSF, and then
integrates flux density into detector-pixel bins.

Run the numerical convergence and local performance check with:

```bash
python tools/validate_composable_model.py
python tools/validate_ism_absorption.py
```

The composable-model report measures convergence against a 64-sample-per-bin
reference, quantifies the bias from binning before coarse convolution, and
records JAX compilation and evaluation timings for the current machine. The
ISM report compares the corrected damping wings with the legacy ±2 Å plateau
and measures detector-space convergence for the absorbed model.

Components are deliberately duck-typed: a custom emission component implements
`unit_profile(wavelength, parameters)` and `weight(parameters)`, while a custom
reversal implements `transmission(wavelength, emission, parameters)`. Standard
components can be assembled into a model used directly by the evidence
framework:

```python
from lyapy.jax_components import (
    CompositeAbsorptionModel,
    CompositeEmissionModel,
    GaussianOpticalDepthReversal,
    HydrogenDeuteriumAbsorber,
    VoigtEmissionComponent,
)
from lyapy.jax_composable import ComposableBinnedModel
from lyapy.jax_instrument import build_high_resolution_operator

emission = CompositeEmissionModel(
    components=(VoigtEmissionComponent(1215.67, 0, 3, 2),),
    log_total_flux_index=1,
    reversal=GaussianOpticalDepthReversal(1215.67, 4, 5, 6),
)
absorption = CompositeAbsorptionModel(
    components=(HydrogenDeuteriumAbsorber(7, 8, 9),),
)
operator = build_high_resolution_operator(
    data_wavelength,
    lsf_offsets_angstrom,
    lsf_profile,
    oversample=16,
)
forward_model = ComposableBinnedModel(
    emission,
    operator,
    absorption_model=absorption,
)
```

The parameter vector in this example is ordered as velocity, log total flux,
Lorentz FWHM, Gaussian FWHM, reversal velocity, reversal FWHM, peak reversal
optical depth, log H I column, H I Doppler b, and H I velocity. LSF offsets must
be wavelength offsets.
