# -*- coding: utf-8 -*-


import emcee
import triangle  ## I made some modifications to my forked copy of triangle to make the kind of plots I want
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import lyapy
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter, MaxNLocator


plt.ion()

## Change this value around sometimes
np.random.seed(82)


## Fixing the D/H ratio at 1.5e-5.  1.56e-5 might be a better value to use though (Wood+ 2004 is the reference, I believe)
d2h_true = 1.5e-5

descrip = '_d2h_fixed' ## appended to saved files throughout
## MCMC parameters
ndim, nwalkers = 9, 150 #ndim is # of fitted parameters
nsteps = 1000 
burnin = 300


## Read in fits file ##
## I've used my lyapy.geocoronal_subtraction function to put the FITS file into the
## format I want... hence the "modAY" suffix to the file below.
input_filename = '/Users/alyo7318/Data/HST/MUSCLES_panchromatic_spectra/p_msl_pan_-----_gj176_panspec_native_resolution_waverange1100.0-1300.0_modAY.fits'

spec_hdu = pyfits.open(input_filename)
spec = spec_hdu[1].data
spec_header = spec_hdu[1].header

## Define wave, flux, and error variables ##
wave_to_fit = spec['wave']
flux_to_fit = spec['flux']
error_to_fit = spec['error']
resolution = float(spec_header['RES_POW'])

## This part is just making sure the error bars in the low-flux wings aren't smaller than the RMS 
## in the wings
rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & (wave_to_fit < 1214.5))])
error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing

## Masking the core of the HI absorption because it may be contaminated by geocoronal emission
mask = lyapy.mask_spectrum(flux_to_fit,interactive=False,mask_lower_limit=36.,mask_upper_limit=42.)
flux_masked = np.ma.masked_array(spec['flux'],mask=mask)
wave_masked = np.ma.masked_array(spec['wave'],mask=mask)
error_masked = np.ma.masked_array(spec['error'],mask=mask)



## Defining parameter ranges. Below I use uniform priors for most of the parameters -- as long
## as they fit inside these ranges.
vs_n_min = 0.
vs_n_max = 75.
am_n_min = -13.1
am_n_max = -11.8
fw_n_min = 125.
fw_n_max = 275.
vs_b_min = -5.
vs_b_max = 100.
am_b_min = -15.4
am_b_max = -13.4
fw_b_min = 300.
fw_b_max = 700.
h1_col_min = 17.0
h1_col_max = 19.5
h1_b_min = 1.
h1_b_max = 20.
h1_vel_min = 20.
h1_vel_max = 100.


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


def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)



# Set up the sampler. There are multiple ways to initialize the walkers,
# and I chose uniform sampling of the parameter ranges.

pos = [np.array([np.random.uniform(low=vs_n_min,high=vs_n_max,size=1)[0],
                 np.random.uniform(low=am_n_min,high=am_n_max,size=1)[0],
                 np.random.uniform(low=fw_n_min,high=fw_n_max,size=1)[0],
                 np.random.uniform(low=vs_b_min,high=vs_b_max,size=1)[0],
                 np.random.uniform(low=am_b_min,high=am_b_max,size=1)[0],
                 np.random.uniform(low=fw_b_min,high=fw_b_max,size=1)[0],
                 np.random.uniform(low=h1_col_min,high=h1_col_max,size=1)[0],
                 np.random.uniform(low=h1_b_min,high=h1_b_max,size=1)[0],
                 np.random.uniform(low=h1_vel_min,high=h1_vel_max,size=1)[0]]) for i in range(nwalkers)]


sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(wave_masked,flux_masked,error_masked))



vs_n_pos = np.zeros(len(pos))
am_n_pos = np.zeros(len(pos))
fw_n_pos = np.zeros(len(pos))
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
    vs_b_pos[i] = pos[i][3]
    am_b_pos[i] = pos[i][4]
    fw_b_pos[i] = pos[i][5]
    h1_col_pos[i] = pos[i][6]
    h1_b_pos[i] = pos[i][7]
    h1_vel_pos[i] = pos[i][8]



# Clear and run the production chain.
print("Running MCMC...")
sampler.run_mcmc(pos, nsteps, rstate0=np.random.get_state())
print("Done.")


# Plot the walkers
fig, axes = plt.subplots(9, 1, sharex=True, figsize=(8, 9))
axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
axes[0].yaxis.set_major_locator(MaxNLocator(5))
axes[0].set_ylabel("vsn")

axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
axes[1].yaxis.set_major_locator(MaxNLocator(5))
axes[1].set_ylabel("amn")

axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
axes[2].yaxis.set_major_locator(MaxNLocator(5))
axes[2].set_ylabel("fwn")

axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
axes[3].yaxis.set_major_locator(MaxNLocator(5))
axes[3].set_ylabel("vsb")

axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
axes[4].yaxis.set_major_locator(MaxNLocator(5))
axes[4].set_ylabel("amb")

axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
axes[5].yaxis.set_major_locator(MaxNLocator(5))
axes[5].set_ylabel("fwb")

axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
axes[6].yaxis.set_major_locator(MaxNLocator(5))
axes[6].set_ylabel("h1col")

axes[7].plot(sampler.chain[:, :, 7].T, color="k", alpha=0.4)
axes[7].yaxis.set_major_locator(MaxNLocator(5))
axes[7].set_ylabel("h1b")

axes[8].plot(sampler.chain[:, :, 8].T, color="k", alpha=0.4)
axes[8].yaxis.set_major_locator(MaxNLocator(5))
axes[8].set_ylabel("h1vel")
axes[8].set_xlabel("step number")

outfile_str = spec_header['STAR'] + descrip + '_walkers.png'
plt.savefig(outfile_str)

# End walker plot


## remove the burn-in period from the sampler
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))


## print best fit parameters + uncertainties

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

# Make the triangle plot. I've made some alterations to triangle.py to make it do what I want.
latex_name_dictionary = {
         'vs_n': 'v$_n$',
         'am_n': r'log A$_n$',
         'fw_n': 'FW$_n$',
         'vs_b': 'v$_b$',
         'am_b': r'log A$_b$',
         'fw_b': 'FW$_b$',
         'h1_col': '$log$ N(HI)',
         'h1_b': 'b',
         'h1_vel': 'v$_{HI}$'}

fig, axes = plt.subplots(ndim, ndim, figsize=(12.5,9))
triangle.corner(samples, bins=80, labels=[latex_name_dictionary['vs_n'], latex_name_dictionary['am_n'], 
                      latex_name_dictionary['fw_n'], latex_name_dictionary['vs_b'], 
                      latex_name_dictionary['am_b'], latex_name_dictionary['fw_b'], 
                      latex_name_dictionary['h1_col'], latex_name_dictionary['h1_b'], 
                      latex_name_dictionary['h1_vel']],
                      max_n_ticks=3,plot_datapoints=False,quantiles=[0.16,0.5,0.84],fig=fig,
                      show_titles=True,verbose=True)


outfile_str = spec_header['STAR'] + descrip + '_cornerplot.png'
plt.savefig(outfile_str)

# End triangle plot


## best fit attenuated profile
model_best_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],
                                          vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                                          h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution,
                                          single_component_flux=False)/1e14

## best fit intrinsic profile
lya_intrinsic_profile_mcmc,lya_intrinsic_flux_mcmc = lyapy.lya_intrinsic_profile_func(wave_to_fit,
         vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],
         return_flux=True)


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

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_limits[i],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_limits[i],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],
                     10**am_b_mcmc[0],fw_b_mcmc[0],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_limits[i],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_mcmc[0],10**am_n_limits[i],fw_n_mcmc[0],vs_b_mcmc[0],
                     10**am_b_mcmc[0],fw_b_mcmc[0],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_limits[i],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_limits[i],vs_b_mcmc[0],
                     10**am_b_mcmc[0],fw_b_mcmc[0],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_limits[i],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_limits[i],
                     10**am_b_mcmc[0],fw_b_mcmc[0],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)


for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_limits[i],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],
                     10**am_b_limits[i],fw_b_mcmc[0],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_limits[i],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)
  lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                     vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],
                     10**am_b_mcmc[0],fw_b_limits[i],return_flux=True)
  lya_intrinsic_flux_limits.append(lya_intrinsic_flux)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_limits[i],
                     h1_b_mcmc[0],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_limits[i],h1_vel_mcmc[0],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)

for i in range(2):
  lya_damped_profile = lyapy.damped_lya_profile(wave_to_fit,vs_n_mcmc[0],10**am_n_mcmc[0],
                     fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0],h1_col_mcmc[0],
                     h1_b_mcmc[0],h1_vel_limits[i],d2h_true,resolution)/1e14
  lya_intrinsic_profile_limits.append(lya_damped_profile)


lya_intrinsic_flux_max_error = np.max(lya_intrinsic_flux_limits) - lya_intrinsic_flux_mcmc
lya_intrinsic_flux_min_error = lya_intrinsic_flux_mcmc - np.min(lya_intrinsic_flux_limits)

## End big messy part -- Error bar stuff complete!






do_plot=True

if do_plot:
      ### MAKING FINAL LYA FIT PLOT ############ I didn't end up using this code for my paper plots


      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(1,1,1)

      for i in np.arange(len(lya_intrinsic_profile_limits)/2)*2:
        ax.fill_between(wave_to_fit,lya_intrinsic_profile_limits[i],
                        lya_intrinsic_profile_limits[i+1],color='lightgray')

      ax.step(wave_to_fit,flux_to_fit,'k')

      short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)
      ax.plot(wave_to_fit,lya_intrinsic_profile_mcmc,'b--',linewidth=1.3)

      chi2_mcmc = np.sum( ( (flux_to_fit[~mask] - model_best_fit[~mask]) / error_to_fit[~mask] )**2 )
      dof_mcmc = len(flux_to_fit[~mask]) - ndim - 1 #####

      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8) ## plotting "masked" region

      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)

      # defining max of y axis 
      y_max = np.max( np.array( [np.max(flux_to_fit),np.max(model_best_fit)] ) )
      y_min = 0.0
      ax.set_ylim([y_min,y_max+0.5e-13])
      ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
      plt.ticklabel_format(useOffset=False)

      am_n_mcmc_float_str = "{0:.2g}".format(10**am_n_mcmc[0])
      base, exponent = am_n_mcmc_float_str.split("e")
      am_n_exponent = float('1e'+exponent)


      # Inserting text
      ax.text(0.03,0.97,'V$_n$ = ' + str(round(vs_n_mcmc[0],1)) + '$^{+' + str(round(vs_n_mcmc[1],1)) + '}_{-' + str(round(vs_n_mcmc[2],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,
        fontsize=12., color='black')
      am_n_p = (10**(am_n_mcmc[0] + am_n_mcmc[1])-10**am_n_mcmc[0])/am_n_exponent
      am_n_m = (10**am_n_mcmc[0]-10**(am_n_mcmc[0] - am_n_mcmc[2]))/am_n_exponent
      ax.text(0.03,0.91,'A$_n$ = ('+ str(round(10**am_n_mcmc[0]/am_n_exponent,1)) + '$^{+' + str(round(am_n_p,1)) + '}_{-' + str(round(am_n_m,1)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')


      ax.text(0.03,0.85,'FW$_n$ = '+ str(round(fw_n_mcmc[0],1)) + '$^{+' + str(round(fw_n_mcmc[1],1)) + '}_{-' + str(round(fw_n_mcmc[2],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.03,0.79,'V$_b$ = '+ str(round(vs_b_mcmc[0],1)) + '$^{+' + str(round(vs_b_mcmc[1],1)) + '}_{-' + str(round(vs_b_mcmc[2],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')

      am_b_p = (10**(am_b_mcmc[0] + am_b_mcmc[1])-10**am_b_mcmc[0])/am_n_exponent
      am_b_m = (10**am_b_mcmc[0]-10**(am_b_mcmc[0] - am_b_mcmc[2]))/am_n_exponent
      ax.text(0.03,0.73,'A$_b$ = ('+ str(round(10**am_b_mcmc[0]/am_n_exponent,2)) + '$^{+' + str(round(am_b_p,2)) + '}_{-' + str(round(am_b_m,2)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')

      ax.text(0.03,0.67,'FW$_b$ = '+ str(round(fw_b_mcmc[0],1)) + '$^{+' + str(round(fw_b_mcmc[1],0)) + '}_{-' + str(round(fw_b_mcmc[2],0)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.03,0.61,'log N(HI) = '+ str(round(h1_col_mcmc[0],2)) + '$^{+' + str(round(h1_col_mcmc[1],2)) + '}_{-' + str(round(h1_col_mcmc[2],2)) + '}$',
        verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.03,0.55,'b = '+ str(round(h1_b_mcmc[0],1)) + '$^{+' + str(round(h1_b_mcmc[1],1)) + '}_{-' + str(round(h1_b_mcmc[2],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12., 
        color='black')
      ax.text(0.03,0.49,'V$_{HI}$ = '+ str(round(h1_vel_mcmc[0],1)) + '$^{+' + str(round(h1_vel_mcmc[1],1)) + '}_{-' + str(round(h1_vel_mcmc[2],1)) + '}$',verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.03,0.43,r'D/H = 1.5$\times$10$^{-5}$',verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')


      lya_intrinsic_flux_argument = float(("%e" % lya_intrinsic_flux_mcmc).split('e')[0])
      lya_intrinsic_flux_exponent = float(("%e" % lya_intrinsic_flux_mcmc).split('e')[1])
      ax.text(0.65,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + '$^{+' + str(round(lya_intrinsic_flux_max_error/10**lya_intrinsic_flux_exponent,2)) + '}_{-' + str(round(lya_intrinsic_flux_min_error/10**lya_intrinsic_flux_exponent,2)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
        verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.93,r'erg s$^{-1}$ cm$^{-2}$',verticalalignment='top',horizontalalignment='right',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.88,r'$\chi^{2}_{\nu}$ = ' + str(round(chi2_mcmc/dof_mcmc,1)),verticalalignment='top',horizontalalignment='right',
        transform=ax.transAxes,fontsize=12., color='black') 

      outfile_str = spec_header['STAR'] + descrip + '_bestfit.png'
      plt.savefig(outfile_str)




## Estimating EUV spectrum -- 9.27 pc is the distance to GJ 176

lyapy.EUV_spectrum(lya_intrinsic_flux_mcmc,9.27,spec_header['STAR'],Mstar=True,savefile=True,doplot=True)

## Saving intrinsic profile

outfile_str = spec_header['STAR'] + '_LyA_intrinsic_profile.txt'
dx = wave_to_fit[1] - wave_to_fit[0]
wave_to_fit_extended = np.arange(1209.5,1222.0+dx,dx)
lya_intrinsic_profile_mcmc_extended = lyapy.lya_intrinsic_profile_func(wave_to_fit_extended,
         vs_n_mcmc[0],10**am_n_mcmc[0],fw_n_mcmc[0],vs_b_mcmc[0],10**am_b_mcmc[0],fw_b_mcmc[0])
np.savetxt(outfile_str,np.transpose([wave_to_fit_extended,lya_intrinsic_profile_mcmc_extended]))

