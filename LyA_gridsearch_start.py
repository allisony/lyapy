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

vs_n_range = np.arange(-35.,25.1,7.5)
am_n_range = np.arange(19e-12,20.1e-12,1e-12)
fw_n_range = np.arange(10.,130.1,15.)

vs_b_range = np.arange(-225.,200.1,25.)
am_b_range = np.arange(1e-14,21e-14,2.e-14)
fw_b_range = np.arange(100.,550.1,50.)

h1_col_range = np.arange(18.0,19.21,.15)
h1_b_range = np.arange(6.,20.1,1.5)
h1_vel_range = np.arange(-30.,40.,7.5)

single_component=False

if not single_component:
  num_jobs = len(vs_n_range)*len(am_n_range)*len(fw_n_range)*len(vs_b_range)*len(am_b_range)*len(fw_b_range)*len(h1_col_range)*len(h1_b_range)*len(h1_vel_range)
else:
  num_jobs = len(vs_n_range)*len(am_n_range)*len(fw_n_range)*len(h1_col_range)*len(h1_b_range)*len(h1_vel_range)


num_cores=1

print "Number of models to compute = " + str(num_jobs)
print "Time now = " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
print "Estimated time = " + str(num_jobs/26855./60./num_cores) + " minutes"

                            

parameter_range = [vs_n_range,am_n_range,fw_n_range,vs_b_range,am_b_range,
                      fw_b_range,h1_col_range,h1_b_range,h1_vel_range]



t0 = time.time()

reduced_chisq_grid,mask = lyapy.LyA_gridsearch(input_filename,parameter_range,num_cores,
                            brute_force=True,single_component_flux=single_component)


#import pdb; pdb.set_trace()

hdu = pyfits.PrimaryHDU(data=reduced_chisq_grid)
outfilename = spec_header['STAR'] + '_chisq_grid1e.fits'
hdu.writeto(outfilename,clobber=True)

t1 = time.time()

print 'Time elapsed (s) = ' + str(t1-t0)
print 'Number of jobs = ' + str(num_jobs)
print 'Rate (per second) = ' + str(num_jobs/(t1-t0))


## Best fit parameters
best_fit_indices = np.where( reduced_chisq_grid == np.min(reduced_chisq_grid) )
global_min = reduced_chisq_grid[best_fit_indices]
if not single_component:
  vs_n_final = vs_n_range[best_fit_indices[0][0]]
  am_n_final = am_n_range[best_fit_indices[1][0]]
  fw_n_final = fw_n_range[best_fit_indices[2][0]]
  vs_b_final = vs_b_range[best_fit_indices[3][0]]
  am_b_final = am_b_range[best_fit_indices[4][0]]
  fw_b_final = fw_b_range[best_fit_indices[5][0]]
  h1_col_final = h1_col_range[best_fit_indices[6][0]]
  h1_b_final = h1_b_range[best_fit_indices[7][0]]
  h1_vel_final = h1_vel_range[best_fit_indices[8][0]]
else:
  vs_n_final = vs_n_range[best_fit_indices[0][0]]
  am_n_final = am_n_range[best_fit_indices[1][0]]
  fw_n_final = fw_n_range[best_fit_indices[2][0]]
  h1_col_final = h1_col_range[best_fit_indices[3][0]]
  h1_b_final = h1_b_range[best_fit_indices[4][0]]
  h1_vel_final = h1_vel_range[best_fit_indices[5][0]]
  vs_b_final = 2.
  am_b_final = 2.
  fw_b_final=2.

sig1_contour = global_min + lyapy.delta_chi2(len(parameter_range),0.32)
sig2_contour = global_min + lyapy.delta_chi2(len(parameter_range),0.05)
sig3_contour = global_min + lyapy.delta_chi2(len(parameter_range),.003)

contour_levels = [sig1_contour]#,sig2_contour,sig3_contour]
contour_colors = ['red']#,'darkorange','gold']



## Reconstructing intrinsic LyA flux
model_best_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_final,am_n_final,fw_n_final,
                                          vs_b_final,am_b_final,fw_b_final,h1_col_final,
                                          h1_b_final,h1_vel_final,1.5e-5,resolution,
                                          single_component_flux=single_component)/1e14

lya_intrinsic_profile,lya_intrinsic_flux = lyapy.lya_intrinsic_profile_func(wave_to_fit,
         vs_n_final,am_n_final,fw_n_final,vs_b_final,am_b_final,fw_b_final,return_flux=True,single_component_flux=single_component)
    ##########################################


if not single_component:
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
else:
  ## Print best fit parameters
  print ' '
  print 'BEST FIT PARAMETERS'
  print 'Reduced Chi-square = ' + str(global_min)
  print 'vs_n = ' + str(vs_n_final)
  print 'am_n = ' + str(am_n_final)
  print 'fw_n = ' + str(fw_n_final)
  print 'h1_col = ' + str(h1_col_final)
  print 'h1_b = ' + str(h1_b_final)
  print 'h1_vel = ' + str(h1_vel_final)
  print 'Total LyA Flux = ' + str(lya_intrinsic_flux) + ' erg/s/cm^2'




### To concatenate reduced_chisq_grid searches
#reduced_chisq_grid_big = np.concatenate((a,b,d),axis=0)
