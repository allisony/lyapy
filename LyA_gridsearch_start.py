import lyapy
import time
import numpy as np

## Read in fits file ##
#input_filename = raw_input("Enter fits file name: ")
input_filename = '../../final_code/h_hd165341_uvsum_1x_51779_spc_modAY.fits'

## Define Range of Parameters for Grid Search ##

vs_n_range = np.arange(-8.36,-6.36,1.)
am_n_range = np.arange(2.,4.1,0.5)
fw_n_range = np.arange(163.,168.1,1.5)

vs_b_range = np.arange(-6.5,-4.49,1.)
am_b_range = np.arange(0.2,.46,0.075)
fw_b_range = np.arange(455.,462.1,1.5)

h1_col_range = np.arange(18.0,18.6,.075)
h1_b_range = np.arange(13.,14.,0.15)
h1_vel_range = np.arange(-24.,-19.,1.5)

num_jobs = len(vs_n_range)*len(am_n_range)*len(fw_n_range)*len(vs_b_range)*len(am_b_range)*len(fw_b_range)*len(h1_col_range)*len(h1_b_range)*len(h1_vel_range)

num_cores=2

print "Number of models to compute = " + str(num_jobs)
print "Time now = " + time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
print "Estimated time = " + str(num_jobs/5000./60./num_cores) + " minutes"

                            

parameter_range = [vs_n_range,am_n_range,fw_n_range,vs_b_range,am_b_range,
                      fw_b_range,h1_col_range,h1_b_range,h1_vel_range]


t0 = time.time()

reduced_chisq_grid = lyapy.LyA_gridsearch(input_filename,parameter_range,num_cores,do_plot=True)

t1 = time.time()

print 'Time elapsed (s) = ' + str(t1-t0)
print 'Number of jobs = ' + str(num_jobs)
print 'Rate (per second) = ' + str(num_jobs/(t1-t0))










