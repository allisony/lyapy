# Legacy GJ 411 reference data

`gj411_legacy_bfb27ca.npz` is a deterministic characterization artifact for
the GJ 411 example included with lyapy. It records the legacy preprocessing,
intrinsic profile, two-component ISM attenuation, STIS LSF convolution, and
integrated intrinsic flux.

The artifact characterizes existing behavior; it is not an independent claim
that every legacy numerical or scientific choice is correct. In particular,
known issues should be corrected through an explicitly reviewed change rather
than silently incorporated into this file.

## Provenance

- Source Git commit: `bfb27ca`
- Reference case: `src/lyapy/example_start_file.py`
- Spectrum: `src/lyapy/HD-95735-E140M-odn905010_1.xl_stis`
- LSF: `src/lyapy/STIS_E140M_LSF.dat`
- Environment: Miniforge `base` at `/Users/aayoungb/miniforge3`
- Python: 3.12.12
- NumPy: 2.4.2
- SciPy: 1.17.0
- Astropy: 7.2.0

The parameter names and values are stored inside the NPZ artifact. Numerical
regression tests use relative tolerance rather than requiring bitwise equality
so that insignificant platform-level differences do not obscure real model
changes.

## Regeneration

From the repository root, with lyapy importable from `src`:

```bash
PYTHONPATH=src python tools/generate_legacy_reference.py
```

Regenerate this artifact only after reviewing and accepting the corresponding
scientific change. Never regenerate it solely to silence a failing test.
