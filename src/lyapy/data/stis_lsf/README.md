# STScI STIS ultraviolet line-spread functions

These 16 numerical tables were downloaded without modification on 2026-11-14
from the [STScI STIS spectral-resolution reference page][source]. Each file
contains relative detector-pixel offset followed by four aperture-specific LSF
profiles. The reference wavelength and grating are encoded in the filename and
repeated in the file header.

Included modes and reference wavelengths:

- G140L, G140M, E140M, and E140H: 1200 and 1500 Angstrom;
- G230L, G230M, E230M, and E230H: 1700 and 2400 Angstrom.

The G430 and G750 files at 3200 Angstrom and longer are intentionally excluded
because they are outside lyapy's ultraviolet scope. Code should use
`lyapy.load_stis_lsf` rather than hard-coding a resource path; this selects the
nearest reference wavelength and permits a user-supplied replacement table.

[source]: https://www.stsci.edu/hst/instrumentation/stis/performance/spectral-resolution
