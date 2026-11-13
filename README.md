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
