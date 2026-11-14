# Local user data

Place private or unpublished science files here for local lyapy development.
Git ignores everything in this directory except this README, so large FITS
files and target data are not accidentally published.

Recommended organization:

```text
user_data/
  stis/
    target-name/
      e140m/
      g140m/
      g140l/
```

Keep the original archive filenames and FITS headers. When possible, include
the individual calibrated exposure products as well as any coadded product;
the individual files are needed to compare native-grid joint fits with a
coadd. Do not copy these files into `src/lyapy`, which is reserved for small,
reviewed data distributed with the package.

For each target, record or communicate the grating, central wavelength,
aperture, wavelength frame, flux and uncertainty units, HST program, visit and
orbit grouping, whether the spectrum is already coadded, and whether airglow
or background subtraction has been applied.
