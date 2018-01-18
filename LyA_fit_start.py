# -*- coding: utf-8 -*-

import lyapy
import emcee
import numpy as np
import astropy.io.fits as pyfits
import lya_plot
import pyspeckit

# Define the probability function as likelihood * prior.
# I use a flat/uniform prior for everything but h1_b.


## 17 Jan 2018 - To Do: add example for Gaussian priors.
    

## for a 2 Gaussian model with a Voigt profile for the ISM - all 9 parameters free
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

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

## for a 1 Gaussian model with a Voigt profile for the ISM - all 6 parameters free
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

def lnprob_single(theta, x, y, yerr):
    lp = lnprior_single(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_single(theta, x, y, yerr)

## Could include an example for how to change any of these to a Gaussian prior


## for a 2 Gaussian model with a Voigt profile for the ISM - h1_b fixed at h1_b_true, all others free
def lnprior_h1_b_fixed(theta):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_vel = theta
    if (vs_n_min < vs_n < vs_n_max) and (am_n_min < am_n < am_n_max) and (fw_n_min < fw_n < fw_n_max) and (vs_b_min < vs_b < vs_b_max) and (am_b_min < am_b < am_b_max) and (fw_b_min < fw_b < fw_b_max) and (h1_col_min < h1_col < h1_col_max) and (h1_vel_min < h1_vel < h1_vel_max):
        return 0.0
    return -np.inf

def lnlike_h1_b_fixed(theta, x, y, yerr):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_vel = theta
    y_model = lyapy.damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
                                       h1_b_true,h1_vel,d2h=d2h_true,resolution=resolution,
                                       single_component_flux=False)/1e14
    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)

def lnprob_h1_b_fixed(theta, x, y, yerr):
    lp = lnprior_h1_b_fixed(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_h1_b_fixed(theta, x, y, yerr)

## for a 1 Gaussian model with a Voigt profile for the ISM - h1_b fixed at h1_b_true, all others free
def lnprior_single_h1_b_fixed(theta):
    vs_n, am_n, fw_n, h1_col, h1_vel = theta
    if (vs_n_min < vs_n < vs_n_max) and (am_n_min < am_n < am_n_max) and (fw_n_min < fw_n < fw_n_max) and (h1_col_min < h1_col < h1_col_max) and (h1_vel_min < h1_vel < h1_vel_max):
        return 0.0
    return -np.inf

def lnlike_single_h1_b_fixed(theta, x, y, yerr):
    vs_n, am_n, fw_n, h1_col, h1_vel = theta
    y_model = lyapy.damped_lya_profile(x,vs_n,10**am_n,fw_n,0,0,0,h1_col,
                                       h1_b_true,h1_vel,d2h=d2h_true,resolution=resolution,
                                       single_component_flux=True)/1e14
    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)

def lnprob_single_h1_b_fixed(theta, x, y, yerr):
    lp = lnprior_single_h1_b_fixed(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_single_h1_b_fixed(theta, x, y, yerr)




## Read in fits file ##
input_filename = '../lyalpha/new_x1d_modAY.fits' #raw_input("Enter fits file name: ")
input_filename = 'p_msl_pan_-----_gj176_panspec_native_resolution_waverange1100.0-1300.0_modAY.fits'

## Set fitting parameters
global single_component_flux
single_component_flux = False
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

## 17 Jan 2018 - To Do: add in option to use STIS LSF from STScI's website. Need toggles for grating
## and slit. Allison has pasted her code here for later incorporation either here or in lyapy.py. Example use:
#stis_lsf = np.loadtxt('STIS_G140L_LSF.dat',skiprows=2) # HIP 23309 52X0.1 G140L
#kernel_for_convolution = ready_stis_lsf(stis_lsf[:,0],stis_lsf[:,1],0.6,wave_to_fit[mask])

#def ready_stis_lsf(orig_lsf_wave,orig_lsf,stis_grating_disp,data_wave):
#
#  """ 
#  Using this function (output aa) will allow the user to use a STIS LSF
#  to convolve their model (lya_obs_high), so lyman_fit = np.convolve(lya_obs_high,aa,mode='same')
#  to match the resolution/resolving power of their data (lyman_fit)
#  orig_lsf_wave is the x array of the STIS LSF
#  orig_lsf is the y array of the STIS LSF
#  stis_grating_disp is the dispersion of the STIS grating for your chosen LSF (units: Ang/pix)
#  data_wave is the wavelength array of your data
#  """
#
#  data_wave_spacing = data_wave[1]-data_wave[0]
#  data_wave_length = len(data_wave)
#  lsf_lam_min = np.round(np.min(orig_lsf_wave*stis_grating_disp)/data_wave_spacing) * data_wave_spacing
#  lsf_lam_onesided = np.arange(lsf_lam_min,0,data_wave_spacing)  ### Make sure it's even and doesn't include zero
#  if len(lsf_lam_onesided) % 2 != 0:
#    lsf_lam_onesided = lsf_lam_onesided[1::] # get rid of the first element of the array
#
#  lsf_lam_flipped = lsf_lam_onesided[::-1]
#  lsf_lam_pluszero=np.append(lsf_lam_onesided,np.array([0]))
#  lsf_lam=np.append(lsf_lam_pluszero,lsf_lam_flipped) # should be odd
#
#  lsf_interp = np.interp(lsf_lam,orig_lsf_wave*stis_grating_disp,orig_lsf/np.sum(orig_lsf))
#  lsf_interp_norm = lsf_interp/np.sum(lsf_interp) # I don't know why I do np.sum() for normalization...
#
#  if data_wave_length < len(lsf_interp_norm):
#      lsf_interp_norm = np.delete(lsf_interp_norm,np.where(lsf_interp_norm == 0))
#      lsf_interp_norm = np.insert(lsf_interp_norm,0,0)
#      lsf_interp_norm = np.append(lsf_interp_norm,0)
#
#  return lsf_interp_norm


## This part is just making sure the error bars in the low-flux wings aren't smaller than the RMS 
## in the wings
rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & (wave_to_fit < 1214.5))])
error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing

## Masking the core of the HI absorption because it may be contaminated by geocoronal emission
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
## EMCEE ##
##################
 
if do_emcee:
    ## Change this value around sometimes
    np.random.seed(82)
    
    ## Fixing the D/H ratio at 1.5e-5.  (Wood+ 2004 is the reference, I believe)
    d2h_true = 1.5e-5
    
    descrip = '_d2h_fixed' ## appended to saved files throughout - this is probably not necessary anymore.
    ## MCMC parameters
    nwalkers = 30
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
