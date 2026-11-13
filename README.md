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

### High-level inference workflow

`InferenceProblem` keeps the data, parameter configuration, forward model, and
likelihood together. The same problem can be run with NUTS or dynesty, and both
backends return an `InferenceResult` with the same posterior-sample,
detector-prediction, summary, and persistence interface:

```python
from lyapy.inference import InferenceProblem, load_inference_result

problem = InferenceProblem(
    data,
    configuration,
    forward_model,
    name="single Voigt fit",
)

nuts_result = problem.run(
    "nuts",
    random_seed=17,
    num_warmup=500,
    num_samples=1000,
)
dynesty_result = problem.run(
    "dynesty",
    random_seed=17,
    nlive=500,
    dlogz=0.05,
)

print(nuts_result.parameter_summary())
print(dynesty_result.log_evidence)
print(nuts_result.prediction_bands())

json_path, arrays_path = nuts_result.save("single_voigt_fit")
restored_result = load_inference_result(json_path)
```

The saved JSON contains readable metadata, diagnostics, and the complete model
and prior specification. Numerical arrays are stored separately in compressed
NPZ format and are loaded with pickling disabled. Detector-space posterior
predictions are always available. Intrinsic profiles and integrated-flux
summaries are included when the forward model exposes those physical products;
set `reconstruct=False` to skip them. Dynesty results include normalized
posterior weights and evidence, while NUTS results do not claim an evidence
estimate. `JointSpectrumData` and `JointForwardModel` use this same workflow for
joint fits of multiple lines.

The fully commented
[`tools/example_inference_workflow.py`](tools/example_inference_workflow.py)
tutorial constructs the data container, parameter vector and priors, Voigt
emission, self-reversal, H I/D I absorption, high-resolution STIS operator,
inference problem, sampler settings, saved result, and diagnostic figure one
piece at a time. Run its non-scientific mechanics check with:

```bash
python tools/example_inference_workflow.py --quick --no-progress
```

Several complete problems can be compared through the same interface. Model
comparison always uses dynesty because NUTS does not estimate the evidence:

```python
from lyapy.inference import compare_inference_problems

comparison = compare_inference_problems(
    (single_gaussian_problem, single_voigt_problem, reversed_voigt_problem),
    random_seed=93,
    nlive=500,
    dlogz=0.05,
)

print(comparison.best_model)
print(comparison.ranking)
best_result = comparison.get_result(comparison.best_model)
pair = comparison.compare("single_voigt", "single_gaussian")
manifest_path, result_paths = comparison.save("stellar_model_comparison")
```

Every named problem may have different parameters, priors, forward model, and
likelihood, but all must contain identical observed spectra. Each candidate's
full `InferenceResult` is retained, so parameter summaries, prediction bands,
intrinsic profiles, integrated fluxes, and evidence can be inspected after
ranking. Saving writes one portable result per model plus a comparison
manifest; `load_inference_comparison` restores the collection. Lyapy validates
the mechanics but does not decide which hypotheses or priors are scientifically
appropriate.

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

`InferenceConfigurationBuilder` assigns indices as parameters are declared,
so custom and joint models do not need fragile hand-counted parameter vectors.
It never selects priors automatically:

```python
from lyapy.inference import InferenceConfigurationBuilder

parameters = InferenceConfigurationBuilder()
stellar_velocity = parameters.add_uniform(
    "stellar_velocity", initial=-20.0, lower=-60.0, upper=20.0
)
lya_flux = parameters.add_uniform(
    "lya_log_total_flux", initial=-11.8, lower=-12.5, upper=-11.0
)
si3_flux = parameters.add_uniform(
    "si3_log_total_flux", initial=-13.0, lower=-13.8, upper=-12.3
)
configuration = parameters.build()
```

Pass `stellar_velocity` to every emission component that shares that velocity;
pass `lya_flux` and `si3_flux` only to their respective models. Use
`parameters.index("stellar_velocity")` when a later custom component needs an
already-declared shared parameter. Fixed and truncated-normal parameters are
also supported.

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
not a recommended physical Si III prescription. Curated physical priors for
Si III, N V, O V, C II, and Mg II remain separate work.

### UV spectral-line and multiplet catalog

`lyapy.atomic_data` provides vacuum rest wavelengths, oscillator strengths,
radiative damping rates, and provenance for H I Ly-alpha, Si III 1206, the
N V 1238/1242 doublet, O V 1218, the C II 1334/1335 multiplet, and Mg II h
and k. Values are transcribed from Morton (2003), ApJS, 149, 205, Table 2
(doi:10.1086/377639). `lyapy.spectral_lines` organizes those transitions into
emission-line groups. Gaussian or Voigt components can share velocity and
width parameters while using either fixed or sampled relative strengths:

```python
from lyapy.spectral_lines import (
    SPECTRAL_LINE_GROUPS,
    gaussian_emission_model_for_line_group,
)

nv = SPECTRAL_LINE_GROUPS["n_v_1238_1242"]
nv_emission = gaussian_emission_model_for_line_group(
    nv,
    velocity_index=0,
    fwhm_index=2,
    log_total_flux_index=1,
    log_relative_weight_indices=(3,),
)
```

Here parameter 3 is the log10 strength of N V 1242 relative to N V 1238.
Alternatively, pass `relative_weights=(2, 1)` for an explicitly chosen fixed
ratio. The catalog records the familiar optically thin 2:1 references for N V
and Mg II, but multiplet constructors never apply a catalog ratio implicitly:
optical depth and radiative transfer can change those ratios substantially.
C II intentionally has no catalog ratio.

Users can define an `AtomicTransition` and `SpectralLineGroup` and pass it to
the same constructors, avoiding an endless hard-coded combination catalog.
The transition metadata are building blocks, not universal profile choices or
prior recommendations. The synthetic joint Ly-alpha/Si III validation uses
these catalog constructors directly.

Each `AtomicTransition` exposes
`(rest_wavelength, oscillator_strength, damping_rate)` through
`absorption_parameters()`. The NumPy ISM calculation accepts either a catalog
key or an `AtomicTransition`, for example
`tau_profile(..., line="n_v_1238")`; the JAX equivalent is
`transition_optical_depth_jax(..., transition="n_v_1238")`.
Phenomenological stellar-emission models use the wavelengths but do not use
the oscillator strengths or atomic damping rates as fitted profile widths.

Catalog transitions can also be composed directly into differentiable ISM
models. Reuse parameter indices when transitions share the same absorbing
column, Doppler `b`, and velocity, as for the N V resonance doublet:

```python
from lyapy.jax_components import (
    AtomicTransitionAbsorber,
    CompositeAbsorptionModel,
)

nv_absorption = CompositeAbsorptionModel((
    AtomicTransitionAbsorber("n_v_1238", 0, 1, 2),
    AtomicTransitionAbsorber("n_v_1242", 0, 1, 2),
))
```

Here parameters 0, 1, and 2 are log column density, Doppler `b`, and velocity.
Sharing is always explicit. In particular, C II 1334 arises from the ground
fine-structure level while C II* 1335 arises from an excited level, so they
normally need separate column-density parameters even if their kinematics are
tied.

The historical absorption names `mg2_h` and `mg2_k` were reversed: the former
evaluated the 2796-A k line and the latter the 2803-A h line. They remain as
deprecated aliases so old calculations are reproducible. New code should use
the physically named `mg_ii_k` and `mg_ii_h` catalog keys. The D I Ly-alpha
constants used by the legacy H I + D I model remain explicitly identified as
legacy values because D I is not a Morton (2003) Table 2 entry. Likewise, the
historical `h1` key retains lyapy's previous H I constants so frozen analyses
remain reproducible; use `h_i_1215` to select the Morton values explicitly.
