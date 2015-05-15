import lyapy
import time
import numpy as np

## Read in fits file ##
#input_filename = raw_input("Enter fits file name: ")
input_filename = '../../final_code/u_hst_sts_g140m_gj832_ock116030_custom_spec_modAY.fits'

## Define Range of Parameters for Grid Search ##

vs_n_range = np.arange(-15.,-5.,1.)
am_n_range = np.arange(2.,3.1,0.05)
fw_n_range = np.arange(131.,141.1,1.)

vs_b_range = np.arange(23.,29.1,1.)
am_b_range = np.arange(0.06,.10,0.05)
fw_b_range = np.arange(389.,399.1,1.)

h1_col_range = np.arange(17.95,18.01,.01)
h1_b_range = np.arange(12.,18.,0.5)
h1_vel_range = np.arange(-21.,-16.,1.)

num_jobs = len(vs_n_range)*len(am_n_range)*len(fw_n_range)*len(vs_b_range)*len(am_b_range)*len(fw_b_range)*len(h1_col_range)*len(h1_b_range)*len(h1_vel_range)

num_cores=2

print "Number of models to compute = " + str(num_jobs)
print "Time now = " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
print "Estimated time = " + str(num_jobs/20855./60./num_cores) + " minutes"

                            

parameter_range = [vs_n_range,am_n_range,fw_n_range,vs_b_range,am_b_range,
                      fw_b_range,h1_col_range,h1_b_range,h1_vel_range]


t0 = time.time()

reduced_chisq_grid = lyapy.LyA_gridsearch(input_filename,parameter_range,num_cores,
                                          do_plot=True,brute_force=True)

hdu = pyfits.PrimaryHDU(data=reduced_chisq_grid)
hdu.writeto('GJ832_chisq_grid.fits',clobber=True)

t1 = time.time()

print 'Time elapsed (s) = ' + str(t1-t0)
print 'Number of jobs = ' + str(num_jobs)
print 'Rate (per second) = ' + str(num_jobs/(t1-t0))










