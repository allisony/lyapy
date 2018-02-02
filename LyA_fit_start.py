# -*- coding: utf-8 -*-

import lyapy
import emcee
import numpy as np
import astropy.io.fits as pyfits
import lya_plot
import copy

## 17 Jan 2018 - To Do: add example for Gaussian priors.
    
import matplotlib.pyplot as plt

## Define priors and likelihoods, where parameters are fixed
def lnprior(theta, minmax):
    assert len(theta) == len(minmax)
    
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
    # ... no parameter was out of range
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    return np.log(h1_b)

def lnlike(theta, x, y, yerr, singcomp=False):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    y_model = lyapy.damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
                                       h1_b,h1_vel,d2h=d2h,resolution=resolution,
                                       single_component_flux=singcomp)/1e14

    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)

def lnprob(theta, x, y, yerr, variables):
    order = ['vs_n', 'am_n', 'fw_n', 'vs_b', 'am_b', 'fw_b', 'h1_col', 'h1_b', 'h1_vel', 'd2h']
    theta_all = []
    range_all = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
                
    assert (i) == len(theta)
    lp = lnprior(theta_all, range_all)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike(theta_all, x, y, yerr, singcomp = variables['am_b']['single_comp'])
    return lp + ll


## Could include an example for how to change any of these to a Gaussian prior



## Read in fits file ##
input_filename = '../lyalpha/new_x1d_modAY.fits' #raw_input("Enter fits file name: ")
input_filename = 'p_msl_pan_-----_gj176_panspec_native_resolution_waverange1100.0-1300.0_modAY.fits'

## Set fitting parameters
do_emcee = True      # do MCMC fitting (no other options)
start_uniform = True # starting positions for MCMC are uniform (alternate: Gaussian)
# for single component flux, set variables['am_b']['single_comp'] = False

## Read in the data ##

spec_hdu = pyfits.open(input_filename)
spec = spec_hdu[1].data
spec_header = spec_hdu[1].header

## Define wave, flux, and error variables ##
wave_to_fit = spec['wave']
flux_to_fit = spec['flux']
error_to_fit = spec['error']
#resolution = 12200. ## a float here is the resolving power that will define the FWHM
                    ## of a Gaussian for convolution of the model. Currently done with
                    ## lyapy.make_kernel
#resolution = spec_hdu[0].header['SPECRES']

## If you want to use one of the LSF's from STScI's website, currently you must download it locally
## as a text file and read it in here. Comment the next 5 lines out if not using this, and set 
## the resolution keyword to a float.
lsf_filename = 'STIS_G140M_LSF.dat'
stis_lsf = np.loadtxt(lsf_filename,skiprows=2) # 52X0.1 G140M
stis_dispersion = wave_to_fit[1]-wave_to_fit[0] # can either set this keyword manually, or derive it 
                                                # from the data.
kernel_for_convolution = lyapy.ready_stis_lsf(stis_lsf[:,0],stis_lsf[:,1],stis_dispersion,wave_to_fit)
resolution = kernel_for_convolution.copy()


## This part is just making sure the error bars in the low-flux wings aren't smaller than the RMS 
## in the wings
rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & (wave_to_fit < 1214.5))])
error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing

## Masking the core of the HI absorption because it may be contaminated by geocoronal emission. Perhaps
## not necessary for all data sets.
mask = lyapy.mask_spectrum(flux_to_fit,interactive=False,mask_lower_limit=36.,mask_upper_limit=42.)
flux_masked = np.transpose(np.ma.masked_array(spec['flux'],mask=mask))
wave_masked = np.transpose(np.ma.masked_array(spec['wave'],mask=mask))
error_masked = np.transpose(np.ma.masked_array(spec['error'],mask=mask))


## Here, set up the dictionary of parameters, their values, and their ranges

## Create the dictionary of parameters
oneparam = {'texname':'', 'vary': True, 'value':1., 
            'scale': 1., 'min':0., 'max':1.,
            'best': np.array([np.nan, np.nan, np.nan])}
variables = {'vs_n': copy.deepcopy(oneparam), 'am_n': copy.deepcopy(oneparam), 'fw_n': copy.deepcopy(oneparam), 
             'vs_b': copy.deepcopy(oneparam), 'am_b': copy.deepcopy(oneparam), 'fw_b': copy.deepcopy(oneparam), 
             'h1_col': copy.deepcopy(oneparam), 
             'h1_b': copy.deepcopy(oneparam), 'h1_vel': copy.deepcopy(oneparam), 'd2h': copy.deepcopy(oneparam)}

## Set values for each parameter
p = 'vs_n'
variables[p]['texname'] = r'$v_n$'
variables[p]['value'] = 40.
variables[p]['vary'] = True
variables[p]['scale'] = 1.
variables[p]['min'] = -100.
variables[p]['max'] = 100.

p = 'am_n'
variables[p]['texname'] = r'$log A_n$'
variables[p]['value'] = -13.6
variables[p]['vary'] = True
variables[p]['scale'] = 0.1
variables[p]['min'] = -16.
variables[p]['max'] = -12.

p = 'fw_n'
variables[p]['texname'] = r'$FW_n$'
variables[p]['value'] = 220.
variables[p]['vary'] = True
variables[p]['scale'] = 5.
variables[p]['min'] = 50.
variables[p]['max'] = 275.

p = 'vs_b'
variables[p]['texname'] = r'$v_b$'
variables[p]['value'] = 34.
variables[p]['vary'] = True
variables[p]['scale'] = 1.
variables[p]['min'] = -100.
variables[p]['max'] = 100.

p = 'am_b'
variables[p]['texname'] = r'$log A_b$'
variables[p]['value'] = -13.68
variables[p]['vary'] = True
variables[p]['scale'] = 0.1
variables[p]['min'] = -19.
variables[p]['max'] = -13.
variables[p]['single_comp'] = False

p = 'fw_b'
variables[p]['texname'] = r'$FW_b$'
variables[p]['value'] = 547.
variables[p]['vary'] = True
variables[p]['scale'] = 50.
variables[p]['min'] = 500.
variables[p]['max'] = 2000.

p = 'h1_col'
variables[p]['texname'] = r'$log N(HI)$'
variables[p]['value'] = 17.64
variables[p]['vary'] = True
variables[p]['scale'] = 0.2
variables[p]['min'] = 16.
variables[p]['max'] = 18.5

p = 'h1_b' #  h1_b_true = 11.5 - for a T=8000 K standard ISM
variables[p]['texname'] = r'$b$',
variables[p]['value'] = 11.98
variables[p]['vary'] = True
variables[p]['scale'] = 0.2
variables[p]['min'] = 1.
variables[p]['max'] = 20.

p = 'h1_vel'
variables[p]['texname'] = r'$v_{HI}$'
variables[p]['value'] = 29.3
variables[p]['vary'] = False
variables[p]['scale'] = 1.
variables[p]['min'] = -50.
variables[p]['max'] = 50.

p = 'd2h' # Fixing the D/H ratio at 1.5e-5.  (Wood+ 2004 is the reference, I believe)
variables[p]['texname'] = r'$D/H$'
variables[p]['value'] = 1.5e-5
variables[p]['vary'] = False
variables[p]['scale'] = 0
variables[p]['min'] = 1e-5
variables[p]['max'] = 2e-5

## This is the order of the parameters that the profile function needs!
param_order = ['vs_n', 'am_n', 'fw_n', 'vs_b', 'am_b', 'fw_b', 'h1_col', 'h1_b', 'h1_vel', 'd2h']


##################
## EMCEE ##
##################
 
if do_emcee:
    ## Change this value around sometimes
    np.random.seed(82)
        
    descrip = '_d2h_fixed' ## appended to saved files throughout - this is probably not necessary anymore.
    ## MCMC parameters
    nwalkers = 30
    nsteps = 5000
    burnin = 500
    # number steps included = nsteps - burnin
    
    
    # Set up the sampler. There are multiple ways to initialize the walkers,
    # and I chose uniform sampling of the parameter ranges.

    varyparams = [] # list of parameters that are being varied this run
    theta, scale, mins, maxs = [], [], [], [] # to be filled with parameter values
    
    for p in param_order: # record if this parameter is being varied
        if variables[p]['vary']:
            varyparams.append(p)
            theta.append(variables[p]['value'])
            scale.append(variables[p]['scale'])
            mins.append(variables[p]['min'])
            maxs.append(variables[p]['max'])
        else: # if parameter fixed, just record the starting value as the best value
            variables[p]['best'][0] = variables[p]['value']  
            
    print "Varying: ", varyparams
    for p in variables.keys():
        print p, variables[p]['value']
    ndim = len(varyparams)

    if start_uniform:
        pos = [np.random.uniform(low=mins, high=maxs) for i in range(nwalkers)]
    else:
        pos = [theta + scale*np.random.randn(ndim) for i in range (nwalkers)]

    import time
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave_to_fit,flux_to_fit,error_to_fit,variables))

    
    # Clear and run the production chain.
    print("Running MCMC...")
    start = time.time()
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    end = time.time()
    print("Done.")
    print(end-start)

    ## checking the autocorrelation time to see if the chains have converged.
    try:
        acor = format(sampler.get_autocorr_time(low=10, high=None, step=1, c=5, fast=False))
        print("Autocorrelation times = " + str(acor))
    except emcee.autocorr.AutocorrError:
        print("AutocorrError: The chain is too short to reliably estimate the autocorrelation time.")
        print("You should re-run with a longer chain. Continuing.")
    
    ## remove the burn-in period from the sampler
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

    
    ## extract the best fitting values
    best = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                                zip(*np.percentile(samples, [16, 50, 84], axis=0)))
                                
    i = 0
    for p in param_order:
        if variables[p]['vary']:
            variables[p]['best'] = best[i]
            i = i+1
            print p, variables[p]['best']

    assert (i) == len(theta)


    ## print best fit parameters + uncertainties.
   
    print("""MCMC result:
        vs_n = {0[0]} +{0[1]} -{0[2]}
        am_n = {1[0]} +{1[1]} -{1[2]}
        fw_n = {2[0]} +{2[1]} -{2[2]}
        vs_b = {3[0]} +{3[1]} -{3[2]}
        am_b = {4[0]} +{4[1]} -{4[2]}
        fw_b = {5[0]} +{5[1]} -{5[2]}
        h1_col = {6[0]} +{6[1]} -{6[2]}
        h1_b = {7[0]} +{7[1]} -{7[2]}
        h1_vel = {8[0]} +{8[1]} -{8[2]}
    
""".format( variables['vs_n']['best'], variables['am_n']['best'], variables['fw_n']['best'], 
           variables['vs_b']['best'], variables['am_b']['best'], variables['fw_b']['best'], 
           variables['h1_col']['best'], variables['h1_b']['best'], variables['h1_vel']['best'],
           variables['d2h']['best'] ) )
    
    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))
    print("should be between 0.25 and 0.5")
    
    
    ## best fit intrinsic profile
    lya_intrinsic_profile_mcmc,lya_intrinsic_flux_mcmc = lyapy.lya_intrinsic_profile_func(wave_to_fit,
         variables['vs_n']['best'][0],10**variables['am_n']['best'][0],variables['fw_n']['best'][0],
         variables['vs_b']['best'][0],10**variables['am_b']['best'][0],variables['fw_b']['best'][0],
         return_flux=True, single_component_flux=variables['am_b']['single_comp'])
    
    ## best fit attenuated profile
    model_best_fit = lyapy.damped_lya_profile(wave_to_fit,
         variables['vs_n']['best'][0],10**variables['am_n']['best'][0],variables['fw_n']['best'][0],
         variables['vs_b']['best'][0],10**variables['am_b']['best'][0],variables['fw_b']['best'][0],
         variables['h1_col']['best'][0], variables['h1_b']['best'][0],variables['h1_vel']['best'][0],
         variables['d2h']['best'][0],
         resolution,
         single_component_flux=variables['am_b']['single_comp'])/1.e14
    
    
    ## Here's the big messy part where I determine the 1-sigma error bars on the
    ## reconstructed, intrinsic LyA flux. From my paper: "For each of the 9 parameters, 
    ## the best-fit values are taken as the 50th percentile (the median) of the marginalized 
    ## distributions, and 1-σ error bars as the 16th and 84th percentiles (shown as dashed 
    ## vertical lines in Figures 3 and 4). The best-fit reconstructed Lyα fluxes are determined 
    ## from the best-fit amplitude, FWHM, and velocity centroid parameters, and the 1-σ error bars
    ## of the reconstructed Lyα flux are taken by varying these parameters individually between 
    ## their 1-σ error bars and keeping all others fixed at their best-fit value. 
    ## The resulting minimum and maximum Lyα fluxes become the 1-σ error bars (Table 2)."
       
    
    lya_plot.walkers(sampler, variables, param_order, subset=False)
    lya_plot.corner(samples, variables, param_order)
    lya_plot.profile(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_best_fit, lya_intrinsic_profile_mcmc, variables, param_order, samples = samples)
