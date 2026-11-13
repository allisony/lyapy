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
python tools/run_composable_model_comparison.py
python tools/run_catalog_model_comparison.py --list-models
python tools/run_catalog_model_comparison.py --models single_gaussian single_voigt
python tools/run_noise_model_comparison.py
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

The composable comparison command uses the corrected high-resolution order of
operations, untruncated H I/D I absorption, a direct intrinsic-total-flux
parameter, and the detector-bin integration operator. It records the complete
model and prior specification alongside the evidence results.

### Curated stellar-model catalog

The catalog intentionally provides a small number of interpretable starting
models rather than every possible component combination:

- `single_gaussian`: one Gaussian without reversal;
- `single_voigt`: one Voigt without reversal;
- `double_gaussian`: ordered narrow and broad Gaussians without reversal;
- `gaussian_gaussian_reversal`: Gaussian emission with Gaussian optical-depth reversal;
- `voigt_legacy_reversal`: Voigt emission with the historical intensity-linked reversal;
- `voigt_gaussian_reversal`: Voigt emission with Gaussian optical-depth reversal.

The double-Gaussian width priors are non-overlapping to prevent component-label
switching. Each catalog entry records its components, complete parameter
schema, prior types, and bounds. User-defined emission or absorption objects
can be registered with `make_model_catalog_entry` as long as they implement the
same small component interfaces used by the standard models.

### Uncertainty and robust-likelihood models

Reported spectral uncertainties can be used unchanged, multiplied by an
inferred global scale, or combined in quadrature with an inferred additive
jitter floor. A Student-t likelihood provides a separate robust option for
isolated outliers. These are deliberately distinct hypotheses: a global scale
addresses broadly underestimated errors, jitter primarily protects bins with
very small formal errors, and Student-t tails reduce the leverage of a few
discordant bins.

`run_noise_model_comparison.py` compares all four treatments for any stellar
model in the catalog. Noise parameters and their priors are included in the
evidence calculation and serialized report; Lyapy does not silently replace or
clip the supplied HST uncertainties.

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

### Fitting uncertainty in the STIS LSF width

The tabulated STIS LSF can either remain fixed at its calibrated shape or gain
one explicit nuisance parameter with
`augment_configuration_with_lsf_width`. The parameter
`ln_lsf_width_scale` evaluates the full profile as `LSF(offset / scale)` and
renormalizes it on the oversampled grid, so convolution continues to conserve
flux. A scale of one exactly reproduces the fixed operator; the default prior
allows 0.8--1.25. Because the whole tabulation is scaled, this changes both the
core width and the locations of existing wings without inventing a separate
analytic LSF.

```python
from lyapy.inference.lsf import augment_configuration_with_lsf_width

fitted_configuration, fitted_forward_model = (
    augment_configuration_with_lsf_width(
        catalog_entry.configuration,
        catalog_entry.forward_model,
        width_scale_bounds=(0.8, 1.25),
    )
)
```

Compare the fixed and fitted-width hypotheses for any curated stellar model:

```bash
python tools/run_lsf_model_comparison.py \
  --stellar-model voigt_legacy_reversal \
  --no-progress
```

The output plot shows detector-space fits, standardized residuals, and the two
LSF shapes. The JSON report records the prior, posterior width scale, evidence,
and sampler settings. The comparison deliberately does not add an LSF-centroid
parameter: for one spectrum that shift is degenerate with wavelength
calibration and the stellar/ISM velocity parameters, and should instead be
handled by an explicitly constrained wavelength-calibration model.

### Batched posterior reconstruction

Posterior profiles are evaluated with JAX batches rather than a Python loop.
For workflows that evaluate more than one posterior sample set for the same
data and model—such as the paired NUTS and dynesty products—reuse one compiled
`ReconstructionEvaluator`:

```python
from lyapy.inference.reconstruction import ReconstructionEvaluator

evaluator = ReconstructionEvaluator(
    data,
    configuration,
    forward_model=forward_model,
)
nuts_reconstruction = evaluator.evaluate(nuts_parameter_draws)
dynesty_reconstruction = evaluator.evaluate(dynesty_parameter_draws)
```

This avoids retracing and rebuilding the same JAX batch. Set `batch_size` when
constructing the evaluator to process very large posterior sets in bounded
chunks; the default vectorizes all draws for maximum throughput. Composable
models read `total_flux` directly from the sampled parameter, so posterior
flux summaries do not numerically reintegrate every intrinsic profile.

Measure performance and verify full-batch/chunked numerical equivalence on the
current machine with:

```bash
python tools/benchmark_posterior_products.py
```

The generated JSON records both throughputs, speedup, and maximum numerical
difference; timings are hardware- and JAX-backend-dependent.

The H I and D I Voigt calculation uses direct linear interpolation indices for
the legacy Harris coefficient tables. Those tables share an exactly uniform
0.01 grid, so direct indexing is mathematically the same interpolation as a
general binary search while avoiding four searches per forward-model call.
Benchmark the retained binary-search reference against the optimized model
with:

```bash
python tools/benchmark_ism_absorption.py
```

On the development CPU at 8× oversampling, direct indexing reduced the isolated
ISM calculation by roughly a factor of three and the complete GJ 411 forward
model by roughly 1.5×. The benchmark also records the maximum absorption and
detector-space differences; timings remain machine-dependent.

Detector integration stores only the high-resolution cells that overlap each
detector pixel, rather than a mostly-zero dense matrix. The public
`binning_matrix` diagnostic remains available and is materialized only on
request. Compare dense and indexed binning—and the rejected FFT-convolution
alternative—with:

```bash
python tools/benchmark_instrument_operator.py
```

For the default GJ 411 grid, indexed binning reduces stored weights by roughly
178×, accelerates detector integration by roughly 5×, and improves the complete
forward model by roughly 25%. Direct LSF convolution remains in use because FFT
convolution is slower for the measured 2,669-cell, 155-kernel-sample problem.

## Joint fits to multiple spectral lines

`JointSpectrumData` and `JointForwardModel` allow any number of named spectra
to share one ordered parameter vector. Each spectrum may use a different
wavelength grid, detector LSF, emission/absorption model, number of pixels, and
likelihood. Shared quantities such as stellar velocity appear once in the
`InferenceConfiguration`; line-specific fluxes, widths, continua, and noise
terms occupy their own indices. Both dynesty and NumPyro sum the normalized
per-spectrum log likelihoods, so the same joint hypothesis also works with the
evidence-comparison framework.

```python
from lyapy.inference.joint import (
    JointForwardModel,
    JointSpectrumData,
    NamedForwardModel,
    NamedSpectrumData,
)

data = JointSpectrumData((
    NamedSpectrumData("Ly alpha", lya_data),
    NamedSpectrumData("Si III", si3_data),
))
forward_model = JointForwardModel((
    NamedForwardModel("Ly alpha", lya_forward_model),
    NamedForwardModel("Si III", si3_forward_model),
))

sampler = run_nuts(
    data,
    shared_configuration,
    rng_key,
    forward_model=forward_model,
)
```

`JointLikelihood` can assign, for example, a Gaussian likelihood to one line
and a Student-t likelihood to another. Names are checked across the data,
forward models, likelihoods, serialized reports, and NumPyro prediction sites
to prevent spectra from being silently paired in the wrong order.

Run the synthetic two-line recovery check with:

```bash
python tools/run_joint_line_validation.py --no-progress
```

This validation shares stellar velocity between synthetic Ly-alpha and Si III
while fitting separate fluxes and widths. It demonstrates inference mechanics,
not a recommended physical Si III prescription. Scientific component catalogs
and curated priors for Si III, N V, O V, C II, and Mg II remain separate work.
