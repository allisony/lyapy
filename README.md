# lyapy
Lyman alpha reconstruction code used in [Youngblood et al. 2016](https://ui.adsabs.harvard.edu/abs/2016ApJ...824..101Y/abstract) and [other papers](https://ui.adsabs.harvard.edu/public-libraries/o5i7z8i2QXWzZSBDV45wqQ).

Edit and run the script "example_start_file.py" to fit the model to the STIS E140M spectrum of GJ 411 "HD-95735-E140M-odn905010_1.xl_stis". Requirements include astropy, numpy, scipy, matplotlib, pandas, time, corner, emcee, joblib, and copy.

## Differentiable model prototype

An optional JAX implementation of the GJ 411-style forward model lives in
`lyapy.jax_models`. It requires 64-bit mode for scientific parity with the
NumPy/Astropy baseline:

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
