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
```

The smoke test varies two parameters. The recovery scripts then validate five
stellar parameters and the full eight-parameter stellar-plus-ISM model on
synthetic spectra. The final command fits the observed GJ 411 STIS E140M
spectrum and writes posterior, fitted-spectrum, reconstruction, reusable NPZ,
and JSON diagnostic products. The reconstruction includes 68% and 95% profile
bands and the integrated intrinsic Ly-alpha flux posterior. These runs keep
D/H, the inactive second absorber, and the tabulated STIS LSF fixed.
