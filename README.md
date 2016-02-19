# lyapy
Code for determining the intrinsic Lyman-alpha flux of nearby low-mass
stars and empirically determining their extreme-UV spectrum.

Run the script "GJ176_MCMC_d2h_fixed.py" to test the MCMC fitting on the "p_msl_pan_-----_gj176_panspec_native_resolution_waverange1100.0-1300.0_modAY.fits" spectrum (STIS G140M). Requirements include emcee, triangle, lyapy (lyapy.py), astropy, voigt (voigt.py), and matplotlib. There are several imports in lyapy.py that are not necessary for the MCMC (pyspeckit, itertools, and multiprocessing) so you can comment those out if you don't have them.
