import lyapy
import time
import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import astropy.table

## Read in fits file ##
input_filename = raw_input("Enter fits file name: ")
#input_filename = '../../final_code/u_hst_sts_g140m_gj832_ock116030_custom_spec_modAY.fits'

## Read in fits file ##
spec_hdu = pyfits.open(input_filename)
spec = spec_hdu[1].data
spec_header = spec_hdu[1].header

## Define wave, flux, error, dq variables ##
wave_to_fit = spec['wave']
flux_to_fit = spec['flux']
error_to_fit = spec['error']
resolution = float(spec_header['RES_POW'])

rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & (wave_to_fit < 1214.5))])
error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing


## Define Range of Parameters for Grid Search ##

vs_n_range = np.arange(22.,34.1,2.)
am_n_range = np.arange(5.4e-13,7.32e-13,0.1e-13)
fw_n_range = np.arange(148.,166.1,2.)

vs_b_range = np.arange(0.,68.6,7.)
am_b_range = np.arange(2.0e-14,4.91e-14,0.25e-14)
fw_b_range = np.arange(380.,560.1,10.)

h1_col_range = np.arange(17.9,18.051,1.)
h1_b_range = np.arange(10.,16.1,0.5)
h1_vel_range = np.arange(25.,34.1,1.)

num_jobs = len(vs_n_range)*len(am_n_range)*len(fw_n_range)*len(vs_b_range)*len(am_b_range)*len(fw_b_range)*len(h1_col_range)*len(h1_b_range)*len(h1_vel_range)

num_cores=1

print "Number of models to compute = " + str(num_jobs)
print "Time now = " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
print "Estimated time = " + str(num_jobs/20855./60./num_cores) + " minutes"

                            

parameter_range = [vs_n_range,am_n_range,fw_n_range,vs_b_range,am_b_range,
                      fw_b_range,h1_col_range,h1_b_range,h1_vel_range]


t0 = time.time()

reduced_chisq_grid = lyapy.LyA_gridsearch(input_filename,parameter_range,num_cores,
                                          brute_force=True)

hdu = pyfits.PrimaryHDU(data=reduced_chisq_grid)
outfilename = spec_header['STAR'] + '_chisq_grid.fits'
hdu.writeto(outfilename,clobber=True)

t1 = time.time()

print 'Time elapsed (s) = ' + str(t1-t0)
print 'Number of jobs = ' + str(num_jobs)
print 'Rate (per second) = ' + str(num_jobs/(t1-t0))

global_min = np.min(reduced_chisq_grid)
## get the indices for this global minimum ##

sig1_contour = global_min + lyapy.delta_chi2(len(parameter_range),0.32)
sig2_contour = global_min + lyapy.delta_chi2(len(parameter_range),0.05)
sig3_contour = global_min + lyapy.delta_chi2(len(parameter_range),.003)

contour_levels = [sig1_contour]#,sig2_contour,sig3_contour]
contour_colors = ['red']#,'darkorange','gold']


## Best fit parameters
best_fit_indices = np.where( reduced_chisq_grid == np.min(reduced_chisq_grid) )
vs_n_final = vs_n_range[best_fit_indices[0][0]]
am_n_final = am_n_range[best_fit_indices[1][0]]
fw_n_final = fw_n_range[best_fit_indices[2][0]]
vs_b_final = vs_b_range[best_fit_indices[3][0]]
am_b_final = am_b_range[best_fit_indices[4][0]]
fw_b_final = fw_b_range[best_fit_indices[5][0]]
h1_col_final = h1_col_range[best_fit_indices[6][0]]
h1_b_final = h1_b_range[best_fit_indices[7][0]]
h1_vel_final = h1_vel_range[best_fit_indices[8][0]]

## Reconstructing intrinsic LyA flux
model_best_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_final,am_n_final,fw_n_final,
                                          vs_b_final,am_b_final,fw_b_final,h1_col_final,
                                          h1_b_final,h1_vel_final,1.5e-5,resolution)/1e14

lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
         vs_n_final,am_n_final,fw_n_final,vs_b_final,am_b_final,fw_b_final,return_flux=True)
    ##########################################



## Print best fit parameters
print ' '
print 'BEST FIT PARAMETERS'
print 'Reduced Chi-square = ' + str(global_min)
print 'vs_n = ' + str(vs_n_final)
print 'am_n = ' + str(am_n_final)
print 'fw_n = ' + str(fw_n_final)
print 'vs_b = ' + str(vs_b_final)
print 'am_b = ' + str(am_b_final)
print 'fw_b = ' + str(fw_b_final)
print 'h1_col = ' + str(h1_col_final)
print 'h1_b = ' + str(h1_b_final)
print 'h1_vel = ' + str(h1_vel_final)
print 'Total LyA Flux = ' + str(lya_intrinsic_flux) + ' erg/s/cm^2'





