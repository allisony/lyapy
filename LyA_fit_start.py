# -*- coding: utf-8 -*-

import lyapy
import emcee
import numpy as np
import astropy.io.fits as pyfits
import lya_plot
import pyspeckit

# Define the probability function as likelihood * prior.
# I use a flat/uniform prior for everything but h1_b.
def lnprior(theta):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel = theta
    if (vs_n_min < vs_n < vs_n_max) and (am_n_min < am_n < am_n_max) and (fw_n_min < fw_n < fw_n_max) and (vs_b_min < vs_b < vs_b_max) and (am_b_min < am_b < am_b_max) and (fw_b_min < fw_b < fw_b_max) and (h1_col_min < h1_col < h1_col_max) and (h1_b_min < h1_b < h1_b_max) and (h1_vel_min < h1_vel < h1_vel_max):
        return np.log(h1_b)
    return -np.inf

def lnlike(theta, x, y, yerr):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel = theta
    y_model = lyapy.damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
                                       h1_b,h1_vel,d2h=d2h_true,resolution=resolution,
                                       single_component_flux=False)/1e14
    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)

def lnprior_single(theta):
    vs_n, am_n, fw_n,  h1_col, h1_b, h1_vel = theta
    if (vs_n_min < vs_n < vs_n_max) and (am_n_min < am_n < am_n_max) and (fw_n_min < fw_n < fw_n_max) and (h1_col_min < h1_col < h1_col_max) and (h1_b_min < h1_b < h1_b_max) and (h1_vel_min < h1_vel < h1_vel_max):
        return np.log(h1_b)
    return -np.inf

def lnlike_single(theta, x, y, yerr):
    vs_n, am_n, fw_n, h1_col, h1_b, h1_vel = theta
    y_model = lyapy.damped_lya_profile(x,vs_n,10**am_n,fw_n,h1_col, 0., 0., 0.,
                                       h1_b,h1_vel,d2h=d2h_true,resolution=resolution,
                                       single_component_flux=True)/1e14
    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

def lnprob_single(theta, x, y, yerr):
    lp = lnprior_single(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_single(theta, x, y, yerr)


## Read in fits file ##
input_filename = '../lyalpha/new_x1d_modAY.fits' #raw_input("Enter fits file name: ")
input_filename = 'p_msl_pan_-----_gj176_panspec_native_resolution_waverange1100.0-1300.0_modAY.fits'

## Set fitting parameters
global single_component_flux
single_component_flux = False
do_mpfit = True
do_emcee = True


## Read in the data ##

spec_hdu = pyfits.open(input_filename)
spec = spec_hdu[1].data
spec_header = spec_hdu[1].header

## Define wave, flux, and error variables ##
wave_to_fit = spec['wave']
flux_to_fit = spec['flux']
error_to_fit = spec['error']
resolution = 12200.
#resolution = spec_hdu[0].header['SPECRES']

## This part is just making sure the error bars in the low-flux wings aren't smaller than the RMS 
## in the wings
rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & (wave_to_fit < 1214.5))])
error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing

## Masking the core of the HI absorption because it may be contaminated by geocoronal emission
## WARNING: masked flux seems to only be used for emcee, not mpfit
mask = lyapy.mask_spectrum(flux_to_fit,interactive=False,mask_lower_limit=36.,mask_upper_limit=42.)
flux_masked = np.transpose(np.ma.masked_array(spec['flux'],mask=mask))
wave_masked = np.transpose(np.ma.masked_array(spec['wave'],mask=mask))
error_masked = np.transpose(np.ma.masked_array(spec['error'],mask=mask))

## Fit only one component for LyA profile or two?
if single_component_flux: 
    variable_names = [
         r'$v_n$',
         r'$log A_n$',
         r'$FW_n$',
         r'$log N(HI)$',
         r'$b$',
         r'$v_{HI}$' ]
else:
    variable_names = [
         r'$v_n$',
         r'$log A_n$',
         r'$FW_n$',
         r'$v_b$',
         r'$log A_b$',
         r'$FW_b$',
         r'$log N(HI)$',
         r'$b$',
         r'$v_{HI}$' ]

ndim = len(variable_names)


##################
## MPFIT ##
##################

if do_mpfit:
    ## Define Initial Guess Parameters ##
    lya_rest = 1215.67
    vs_n = -12.93  # narrow LyA velocity shift     ; [km/s]  
    am_n = 13.5e-12     # amplitude of narrow LyA       ; [erg/cm2/s/A]
    fw_n = 131.17      # FWHM of narrow LyA            ; [km/s]  
    
    vs_b = -93.04   # broad LyA velocity shift      ; [km/s]
    am_b = 1.46e-14          # amplitude of broad LyA        ; [erg/cm2/s/A]
    fw_b = 317.14            # FWHM of broad LyA             ; [km/s]
   
    h1_col = 18.282         # ISM N(HI)                     ; [cm-2]
    h1_b   = 12.83       # ISM b(HI)
    h1_vel = -3.69  # HI velocity  ; mpfit does NOT like letting the velocity run negative....need to fix. 
    d2h = 1.5e-5                                     
    
    fixed = [False,False,False,False,False,False,False,False,False,True,True]
    if single_component_flux:
        vs_b = 0.
        am_b = 0.
        fw_b = 100.
        fixed = [False,False,False,True,True,True,False,False,False,True,True]
    initial_parameters = [vs_n,am_n,fw_n,vs_b,am_b,fw_b,h1_col,h1_b,h1_vel,d2h,resolution]
    
    spec = pyspeckit.Spectrum(xarr=wave_to_fit,data=flux_to_fit*1e14,
                      error=error_to_fit*1e14,doplot=False,header=spec_header)
    spec.Registry.add_fitter('lyapy.damped_lya',lyapy.damped_lya_fitter(),11)


    print "Beginning fitting"


    spec.specfit(fittype='lyapy.damped_lya',guesses=initial_parameters,quiet=False,fixed=fixed)

    print spec.specfit.parinfo

    fit_parameters = spec.specfit.parinfo.values
    fit_parameters_errors = spec.specfit.parinfo.errors
    chi2 = spec.specfit.chi2
    dof = spec.specfit.dof
    reduced_chi2 = chi2/dof

    vs_n_mpfit = fit_parameters[0]
    am_n_mpfit = fit_parameters[1]
    fw_n_mpfit = fit_parameters[2]
    vs_b_mpfit = fit_parameters[3]
    am_b_mpfit = fit_parameters[4]
    fw_b_mpfit = fit_parameters[5]
    h1_col_mpfit = fit_parameters[6]
    h1_b_mpfit = fit_parameters[7]
    h1_vel_mpfit = fit_parameters[8]
    d2h_mpfit = fit_parameters[9]
    resolution_mpfit = fit_parameters[10]

    vs_n_err = fit_parameters_errors[0]
    am_n_err = fit_parameters_errors[1]
    fw_n_err = fit_parameters_errors[2]
    vs_b_err = fit_parameters_errors[3]
    am_b_err = fit_parameters_errors[4]
    fw_b_err = fit_parameters_errors[5]
    h1_col_err = fit_parameters_errors[6]
    h1_b_err = fit_parameters_errors[7]
    h1_vel_err = fit_parameters_errors[8]
    d2h_err = fit_parameters_errors[9]
    resolution_err = fit_parameters_errors[10]

    ## best fit intrinsic profile
    lya_intrinsic_profile_mpfit,lya_intrinsic_flux_mcmc = lyapy.lya_intrinsic_profile_func(wave_to_fit,
             vs_n_mpfit,am_n_mpfit,fw_n_mpfit,
             vs_b_mpfit,am_b_mpfit,fw_b_mpfit,
             return_flux=True)
    
    model_mpfit = lyapy.damped_lya_profile(wave_to_fit,vs_n_mpfit,am_n_mpfit,fw_n_mpfit,
                                        vs_b_mpfit,am_b_mpfit,fw_b_mpfit,h1_col_mpfit,
                                        h1_b_mpfit,h1_vel_mpfit,d2h_mpfit,resolution,
                                        single_component_flux=single_component_flux)/1e14

    lya_plot.profile(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_mpfit, lya_intrinsic_profile_mpfit, samples = None,
            d2h_true = 1.5e-5)

##################
## EMCEE ##
##################
 
if do_emcee:
    ## Change this value around sometimes
    np.random.seed(82)
    
    ## Fixing the D/H ratio at 1.5e-5.  1.56e-5 might be a better value to use though (Wood+ 2004 is the reference, I believe)
    d2h_true = 1.5e-5
    
    descrip = '_d2h_fixed' ## appended to saved files throughout
    ## MCMC parameters
    nwalkers = 30 #ndim is # of fitted parameters
    nsteps = 1000
    burnin = 500
    # number steps included = nsteps - burnin
    
    
    
    
    ## Defining parameter ranges. Below I use uniform priors for most of the parameters -- as long
    ## as they fit inside these ranges.
    vs_n_min = -100.
    vs_n_max = 100.
    am_n_min = -16.
    am_n_max = -12.
    fw_n_min = 50.
    fw_n_max = 275.
    vs_b_min = -100.
    vs_b_max = 100.
    am_b_min = -19.
    am_b_max = -13.
    fw_b_min = 500.
    fw_b_max = 2000.
    h1_col_min = 16.0
    h1_col_max = 18.5
    h1_b_min = 1.
    h1_b_max = 20.
    h1_vel_min = -50
    h1_vel_max = 50.
    
    
    
    
    # Set up the sampler. There are multiple ways to initialize the walkers,
    # and I chose uniform sampling of the parameter ranges.
    if single_component_flux:
        pos = [np.array([np.random.uniform(low=vs_n_min,high=vs_n_max,size=1)[0],
                         np.random.uniform(low=am_n_min,high=am_n_max,size=1)[0],
                         np.random.uniform(low=fw_n_min,high=fw_n_max,size=1)[0],
                         np.random.uniform(low=h1_col_min,high=h1_col_max,size=1)[0],
                         np.random.uniform(low=h1_b_min,high=h1_b_max,size=1)[0],
                         np.random.uniform(low=h1_vel_min,high=h1_vel_max,size=1)[0]]) for i in range(nwalkers)]
    else:
        pos = [np.array([np.random.uniform(low=vs_n_min,high=vs_n_max,size=1)[0],
                         np.random.uniform(low=am_n_min,high=am_n_max,size=1)[0],
                         np.random.uniform(low=fw_n_min,high=fw_n_max,size=1)[0],
                         np.random.uniform(low=vs_b_min,high=vs_b_max,size=1)[0],
                         np.random.uniform(low=am_b_min,high=am_b_max,size=1)[0],
                         np.random.uniform(low=fw_b_min,high=fw_b_max,size=1)[0],
                         np.random.uniform(low=h1_col_min,high=h1_col_max,size=1)[0],
                         np.random.uniform(low=h1_b_min,high=h1_b_max,size=1)[0],
                         np.random.uniform(low=h1_vel_min,high=h1_vel_max,size=1)[0]]) for i in range(nwalkers)]
    
    
    if single_component_flux:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_single, args=(wave_to_fit,flux_to_fit,error_to_fit))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave_to_fit,flux_to_fit,error_to_fit))
    
        
    vs_n_pos = np.zeros(len(pos))
    am_n_pos = np.zeros(len(pos))
    fw_n_pos = np.zeros(len(pos))
    if not single_component_flux:
        vs_b_pos = np.zeros(len(pos))
        am_b_pos = np.zeros(len(pos))
        fw_b_pos = np.zeros(len(pos))
    h1_col_pos = np.zeros(len(pos))
    h1_b_pos = np.zeros(len(pos))
    h1_vel_pos = np.zeros(len(pos))
    
    
    for i in range(len(pos)):
        vs_n_pos[i] = pos[i][0]
        am_n_pos[i] = pos[i][1]
        fw_n_pos[i] = pos[i][2]
        if not single_component_flux:
            vs_b_pos[i] = pos[i][3]
            am_b_pos[i] = pos[i][4]
            fw_b_pos[i] = pos[i][5]
        h1_col_pos[i] = pos[i][-3]
        h1_b_pos[i] = pos[i][-2]
        h1_vel_pos[i] = pos[i][-1]
    
    
    
    # Clear and run the production chain.
    print("Running MCMC...")
    sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
    print("Done.")
    
    
    ## remove the burn-in period from the sampler
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    
    
    ## print best fit parameters + uncertainties
    if single_component_flux:
        vs_b_mcmc, am_b_mcmc, fw_b_mcmc = [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]
        vs_n_mcmc, am_n_mcmc, fw_n_mcmc, h1_col_mcmc, h1_b_mcmc, \
                                    h1_vel_mcmc  = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                                    zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    else:
        vs_n_mcmc, am_n_mcmc, fw_n_mcmc, vs_b_mcmc, am_b_mcmc, fw_b_mcmc, h1_col_mcmc, h1_b_mcmc, \
                                    h1_vel_mcmc  = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                                    zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    
    
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
    
    """.format(vs_n_mcmc, am_n_mcmc, fw_n_mcmc, vs_b_mcmc, am_b_mcmc, fw_b_mcmc, h1_col_mcmc, 
               h1_b_mcmc, h1_vel_mcmc))
    
    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))
    print("should be between 0.25 and 0.5")
    
    
    ## best fit intrinsic profile
    lya_intrinsic_profile_mcmc,lya_intrinsic_flux_mcmc = lyapy.lya_intrinsic_profile_func(wave_to_fit,
             vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],
             return_flux=True)
    
    ## best fit attenuated profile
    model_best_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10.**am_n_mcmc[0],fw_n_mcmc[0],
                                              vs_b_mcmc[0],10.**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                                              h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution,
                                              single_component_flux=single_component_flux)/1.e14
    
    
    ## Here's the big messy part where I determine the 1-sigma error bars on the
    ## reconstructed, intrinsic LyA flux. From my paper: "For each of the 9 parameters, 
    ## the best-fit values are taken as the 50th percentile (the median) of the marginalized 
    ## distributions, and 1-σ error bars as the 16th and 84th percentiles (shown as dashed 
    ## vertical lines in Figures 3 and 4). The best-fit reconstructed Lyα fluxes are determined 
    ## from the best-fit amplitude, FWHM, and velocity centroid parameters, and the 1-σ error bars
    ## of the reconstructed Lyα flux are taken by varying these parameters individually between 
    ## their 1-σ error bars and keeping all others fixed at their best-fit value. 
    ## The resulting minimum and maximum Lyα fluxes become the 1-σ error bars (Table 2)."
    
    lya_intrinsic_profile_limits = []  ## this is for showing the gray-shaded regions in my Figure 1
    lya_intrinsic_flux_limits = [] ## this is for finding the 1-σ LyA flux error bars
    
    vs_n_limits = [vs_n_mcmc[0] + vs_n_mcmc[1], vs_n_mcmc[0] - vs_n_mcmc[2]]
    am_n_limits = [am_n_mcmc[0] + am_n_mcmc[1], am_n_mcmc[0] - am_n_mcmc[2]]
    fw_n_limits = [fw_n_mcmc[0] + fw_n_mcmc[1], fw_n_mcmc[0] - fw_n_mcmc[2]]
    vs_b_limits = [vs_b_mcmc[0] + vs_b_mcmc[1], vs_b_mcmc[0] - vs_b_mcmc[2]]
    am_b_limits = [am_b_mcmc[0] + am_b_mcmc[1], am_b_mcmc[0] - am_b_mcmc[2]]
    fw_b_limits = [fw_b_mcmc[0] + fw_b_mcmc[1], fw_b_mcmc[0] - fw_b_mcmc[2]]
    h1_col_limits = [h1_col_mcmc[0] + h1_col_mcmc[1], h1_col_mcmc[0] - h1_col_mcmc[2]]
    h1_b_limits = [h1_b_mcmc[0] + h1_b_mcmc[1], h1_b_mcmc[0] - h1_b_mcmc[2]]
    h1_vel_limits = [h1_vel_mcmc[0] + h1_vel_mcmc[1], h1_vel_mcmc[0] - h1_vel_mcmc[2]]
    
    ## Whenever I say "damped" I mean "attenuated"      
    
    #lya_plot.walkers(sampler)
    #lya_plot.corner(samples)
    lya_plot.profile(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
                model_best_fit, lya_intrinsic_profile_mcmc, samples=samples, d2h_true = 1.5e-5)
