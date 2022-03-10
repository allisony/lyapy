import numpy as np
from lyapy import lyapy
from lyapy.fitting import *
from scipy.io import readsav
import time
from lyapy import lya_plot
from astropy.modeling.models import Voigt1D
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import gridspec


plt.ion()
#########################
#                       #
# Top-Level Parameters  #
#                       #
#########################


nwalkers = 100 # how many walkers do you want for your fit?
nsteps = 1000 # how many steps?
burnin = 500 # how long is your burnin period?


np.random.seed(982) # change as you see fit. Doesn't really matter.

start_uniform = False # if True: start the walkers randomly within each variable's min/max range (defined below)
                     # if False: start the walkers in a Gaussian distribution with mean and std dev defined with the variables (below)

nruns = 1 # how many runs do you want to do? Keep at 1 until you know what you're doing.


fresh_start = True # if True: starts the fit from scratch. if False: picks up where the fit left off.
perform_error = False # if True: does a long error estimation. Don't set to True until your fit has converged and you're satisfied with it.
mask_data = False # if True: allows the user to mask out part(s) of their data from being evaluated by the log probability function (e.g., airglow)

#######################################
#                                     #
# Reading in and formatting the data  #
#                                     #
#######################################

## Read in the data ##
input_filename = 'HD-95735-E140M-odn905010_1.xl_stis'
sav = readsav(input_filename)
wave_to_fit = sav['wave']
flux_to_fit = sav['flux']
error_to_fit = sav['photerr']#*np.sqrt(2)
#input_filename = '/Users/aayoungb/Data/HST/FUMES/HIP112312/HIP112312_FUVe_clip.fits'

#spec_hdu = pyfits.open(input_filename)
#spec = spec_hdu[0].data


## Define wave, flux, and error variables ##
#wave_to_fit = spec[0,:]
#flux_to_fit = spec[1,:]
#error_to_fit = spec[2,:]

fit_mask = (wave_to_fit >= 1214.5) & (wave_to_fit <= 1216.55)

# fit the continuum and subtract - not going to do this for E140M...
cont_mask1 = (wave_to_fit > 1212.) & (wave_to_fit < 1213)
cont_mask2 = (wave_to_fit > 1217.) & (wave_to_fit < 1217.9)

cont_mask = cont_mask1 + cont_mask2 

cont_fit = np.polyfit(wave_to_fit[cont_mask],flux_to_fit[cont_mask],deg=1)

cont = wave_to_fit * cont_fit[0] + cont_fit[1]
# I don't want to do this linear continuum fit anymore.

plt.figure()
plt.plot(wave_to_fit,flux_to_fit)
plt.plot(wave_to_fit,cont)

flux_to_fit = flux_to_fit[fit_mask] - cont[fit_mask]

wave_to_fit = wave_to_fit[fit_mask]
error_to_fit = error_to_fit[fit_mask]

## subtracting the geocorona 

lya_rest=1215.67
c_km = 3e5
lam_geocorona = 1215.59
lamshft_geocorona = lam_geocorona - lya_rest
lam_o_geocorona = lya_rest + lamshft_geocorona
vshft_geocorona = c_km * (lamshft_geocorona/lya_rest)
peak_geocorona = 7e-14
base_geocorona=0
dlam_geocorona=0.05
sig_geocorona =  dlam_geocorona/2.355
### create the functional form
lz0_geocorona =  ( (wave_to_fit - lam_o_geocorona) / sig_geocorona )**2 
geo_mod = (peak_geocorona * np.exp(-lz0_geocorona/2.0) ) 
  
#flux_to_fit = flux_to_fit - geo_mod

#plt.plot(wave_to_fit, geo_mod)


if mask_data:
    mask = (wave_to_fit >= 1215.6081) & (wave_to_fit <=1215.7470) # example wavelength to mask

else:
    mask = np.isreal(wave_to_fit) # otherwise all elements of wave_to_fit evaluated by fit


#### Define the resolution
#resolution = 12200. ## a float here is the resolving power that will define the FWHM
                    ## of a Gaussian for convolution of the model. Currently done with
                    ## lyapy.make_kernel
#resolution = spec_hdu[0].header['SPECRES']

## If you want to use one of the LSF's from STScI's website, currently you must download it locally
## as a text file and read it in here. Comment the next 5 lines out if not using this, and set 
## the resolution keyword to a float.
lsf_filename = 'STIS_E140M_LSF.dat'
lsf = np.loadtxt(lsf_filename,skiprows=2) # 0.2x0.2 E140M - confirmed from MAST
lsf_wave = lsf[:,0]
lsf_array = lsf[:,3]
lsf_list = [lsf_wave,lsf_array]


####################################
#                                  #
# DEFINE THE DATA TO FIT           #
#                                  #
####################################


# must be a list - this gives the option of fitting multiple spectra at once (you probably won't be doing that though)
wave_to_fit = [wave_to_fit]
flux_to_fit = [flux_to_fit]
error_to_fit = [error_to_fit]
resolution = [lsf_list]

##############################
#                            #
# DEFINE THE MODEL           #
#                            #
##############################






def my_model(x, resolution, parameters, variables, lnlike=True, convolve=True): 

    vs, am, fw_L, fw_G,  \
    h1_col, h1_b, h1_vel, h1_col2, h1_b2, h1_vel2, d2h, p = parameters

    wave_lya = x[0]

    kernel_for_convolution = lyapy.ready_stis_lsf(resolution[0][0],resolution[0][1],wave_lya[1]-wave_lya[0],wave_lya)
    resolution_lya = kernel_for_convolution.copy()

    



    if variables['h1_vel']['offset'] == True:
        h1_vel = h1_vel + vs_n



    line_center = vs/3e5*1215.67+1215.67
    sigma_G = fw_G/3e5 * 1215.67 
    sigma_L = fw_L/3e5 * 1215.67 
                                           

    voigt_profile_func = Voigt1D(x_0 = line_center, amplitude_L = 10**am, 
                                                fwhm_L = sigma_L, fwhm_G = sigma_G)



                       



    rev_profile = np.exp(-p * voigt_profile_func(wave_lya) / np.max(voigt_profile_func(wave_lya)))

    lya_intrinsic_profile = voigt_profile_func(wave_lya) * rev_profile 


    total_attenuation = lyapy.total_tau_profile_func(wave_lya,
                                               h1_col,h1_b,h1_vel,d2h)
    total_attenuation2 = lyapy.total_tau_profile_func(wave_lya,
                                           h1_col2,h1_b2,h1_vel2,d2h)

    y_model_lya = lya_intrinsic_profile * total_attenuation * total_attenuation2
    y_model_lya_convolved = np.convolve(y_model_lya, resolution_lya, mode='same')

    lya_intrinsic_profile_convolved = np.convolve(lya_intrinsic_profile, resolution_lya, mode='same')


    if convolve:
        y_model_list = [y_model_lya_convolved]

        y_intrinsic_list = [lya_intrinsic_profile_convolved]

        y_ISM_attenuation_list = [np.convolve(total_attenuation*total_attenuation2, resolution_lya, mode='same')]

        y_reversal_list = [np.convolve(rev_profile, resolution_lya, mode='same')]

    elif convolve == 'both':
        y_model_list = [y_model_lya_convolved, y_model_lya]
        
        y_intrinsic_list = [lya_intrinsic_profile_convolved, lya_intrinsic_profile]

        y_ISM_attenuation_list = [np.convolve(total_attenuation*total_attenuation2, resolution_lya, mode='same'), total_attenuation*total_attenuation2]

        y_reversal_list = [np.convolve(rev_profile, resolution_lya, mode='same'), rev_profile]


    else:
        y_model_list = [y_model_lya]

        y_intrinsic_list = [lya_intrinsic_profile]

        y_ISM_attenuation_list = [total_attenuation*total_attenuation2]

        y_reversal_list = [rev_profile]


    if lnlike:

        if not convolve:
            raise ValueError('convolve=False when it should =True')

        return y_model_list 

    else:

        return [y_model_list, y_intrinsic_list, y_ISM_attenuation_list, y_reversal_list] # must be a list



##########################################################################
#                                                                        #
# Setting up the dictionary of parameters, their values and their ranges #
#                                                                        #
##########################################################################

variables_order = ['vs', 'am', 'fw_L', 'fw_G', 
                  'h1_col', 'h1_b', 'h1_vel', 'h1_col2', 'h1_b2', 'h1_vel2', 'd2h', 'p'] # list of strings corresponding to your variables


variables = make_parameter_dictionary(variables_order)

p = 'vs'
variables[p]['texname'] = r'$v$' # for the cornerplot
variables[p]['value'] = -87 # if vary = False, then this is the value of this variable assumed by the model. If uniform=False (set at the beginning), then this is the mean of the Gaussian distribution for the walkers' starting points for this variable
variables[p]['vary'] = True 
variables[p]['scale'] = 3.6 #  if uniform=False (set at beginning), then this is the stddev of the Gaussian distribution for the walkers' starting points for this variable
variables[p]['min'] = -100  # minimum of the parameter range
variables[p]['max'] = -75 # maximum of the parameter range
variables[p]['my model'] = my_model # make sure this points to your model function
variables[p]['Gaussian prior'] = False # do you want this parameter to have a Gaussian prior? If False, then the prior is uniform between min and max
variables[p]['prior mean'] = -129.3 # Gaussian prior mean
variables[p]['prior stddev'] = 0.6 # Gaussian prior std dev


p = 'am'
variables[p]['texname'] = r'$log A$'
variables[p]['value'] = -10.06
variables[p]['vary'] = True
variables[p]['scale'] = 0.1
variables[p]['min'] = -18.
variables[p]['max'] = -8.

p = 'fw_L'
variables[p]['texname'] = r'$FW_{L}$'
variables[p]['value'] = 7.76
variables[p]['vary'] = True
variables[p]['scale'] = 1.
variables[p]['min'] = 1.
variables[p]['max'] = 1000.

p = 'fw_G'
variables[p]['texname'] = r'$FW_{G}$'
variables[p]['value'] = 88.68
variables[p]['scale'] = 9.
variables[p]['vary'] = True
variables[p]['min'] = 1.
variables[p]['max'] = 1000.



p = 'h1_col'
variables[p]['texname'] = r'$log N(HI)$'
variables[p]['value'] = 17.84
variables[p]['vary'] = True
variables[p]['scale'] = 0.05
variables[p]['min'] = 17.5
variables[p]['max'] = 18.9

p = 'h1_b' 
variables[p]['texname'] = 'b',
variables[p]['value'] = 11.84
variables[p]['vary'] = True 
variables[p]['scale'] = 0.7
variables[p]['min'] = 8.6
variables[p]['max'] = 20.
variables[p]['Gaussian prior'] = False
variables[p]['prior mean'] = 11.5
variables[p]['prior stddev'] = 3


p = 'h1_vel'
variables[p]['texname'] = r'$v_{HI}$'
variables[p]['value'] = 4.18
variables[p]['vary'] = True 
variables[p]['scale'] = 1.
variables[p]['min'] = -30
variables[p]['max'] = 30.
variables[p]['offset'] = False # do you want this parameter to be offset from the vs parameter, or independent?
variables[p]['Gaussian prior'] = False
variables[p]['prior mean'] = -26.58
variables[p]['prior stddev'] = 2. # 


p = 'h1_col2'
variables[p]['texname'] = r'$log N(HI,2)$'
variables[p]['value'] = 0
variables[p]['vary'] = False
variables[p]['scale'] = 0.01
variables[p]['min'] = 0 
variables[p]['max'] = 19.

p = 'h1_b2' 
variables[p]['texname'] = r'$b_2$',
variables[p]['value'] = 11.5
variables[p]['vary'] = False 
variables[p]['scale'] = 0.2
variables[p]['min'] = 7.
variables[p]['max'] = 15.

p = 'h1_vel2'
variables[p]['texname'] = r'$v_{HI,2}$'
variables[p]['value'] = 5.87
variables[p]['vary'] = False 
variables[p]['scale'] = 0.5
variables[p]['min'] = -50.
variables[p]['max'] = 200.
variables[p]['offset'] = True



p = 'd2h' 
variables[p]['texname'] = r'$D/H$'
variables[p]['value'] = 1.5e-5
variables[p]['vary'] = False
variables[p]['scale'] = 0
variables[p]['min'] = 1e-5
variables[p]['max'] = 2e-5

p = 'p'
variables[p]['texname'] = r'$p$'
variables[p]['value'] = 1.5
variables[p]['vary'] = True  
variables[p]['scale'] = 0.1
variables[p]['min'] = 1.0
variables[p]['max'] = 1.8




perform_variable_check(variables)


#################
#               # 
# Sampler setup # #edits not needed unless something is very wrong
#               #
#################    


if not perform_error:

    sampler, pos0 = setup_sampler(wave_to_fit, flux_to_fit, error_to_fit, resolution,  
                              nwalkers, variables, variables_order, my_model, mask, start_uniform)


######################
#                    #
# PERFORM THE EMCEE  #  #edits not needed unless something is very wrong
#                    #
######################

if not perform_error:
    sampler_chain = perform_mcmc(sampler, pos0, nsteps, nruns=nruns, fresh_start = fresh_start)
else:
    sampler_chain = get_sampler_chain()


ndim = 0
for p in variables:
    if variables[p]['vary']:
        ndim += 1

samples = sampler_chain[:, burnin:, :].reshape((-1, ndim))


best, variables = extract_best_fit_parameters(samples, variables, variables_order)


##################################################################
#                                                                #
# Diagnostics: mean acceptance fraction and convergence plots    # #edits not needed unless something is very wrong
#                                                                #
##################################################################

if not perform_error:
    print("Mean acceptance fraction: {0:.3f}"
                    .format(np.mean(sampler.acceptance_fraction)))
    print("should be between 0.25 and 0.5")


make_convergence_plot(sampler_chain, ndim, burnin)



##############################
#                            #
# Calling the plot functions # # the only edits in this section are for the "thin out" and "convolve" parameters
#                            #
##############################


subset=True if sampler_chain.shape[1] > 1e5 else False
lya_plot.walkers(sampler_chain, variables, variables_order, burnin, subset=subset)

if subset: ## if your sample chain is REALLY long, increase the "100" by at least an order of magnitude to make this step computationally quicker
    make_corner_plot(samples[::100], best, variables, variables_order, ndim)
else:
    make_corner_plot(samples, best, variables, variables_order, ndim)





line_percentiles_to_store_dic, reconstructed_fluxes_dic = profile_plot(wave_to_fit, flux_to_fit, error_to_fit, resolution, samples, my_model, variables, variables_order, perform_error=perform_error, thin_out=10, convolve='both') # thin_out is a parameter that makes this run faster (only makes a difference if perform_error = True)



##############################
#                            #
# Save fit results           #
#                            #
##############################



if perform_error:

    filename="GJ411_MCMC_results.csv"  # EDIT ME FOR EACH FIT!



    line_names = ['lya']
    array_per_line_names = ['model', 'model unconvolved', 'intrinsic', 'intrinsic unconvolved', 'ism', 'ism unconvolved', 
                                      'reversal', 'reversal unconvolved']
    percentile_names = ['low_2sig','low_1sig','median','high_1sig','high_2sig']
    data_names = ['wave', 'flux', 'error','data_mask']
    data = [wave_to_fit, flux_to_fit, error_to_fit, mask]


    for i in range(len(line_names)):
        line_name = line_names[i]

        df_line = pd.DataFrame()
        
        for j in range(len(array_per_line_names)):
            array_name = array_per_line_names[j]

            for k in range(len(percentile_names)):

                percentile_name = percentile_names[k]

                df_line[line_name + "_" + array_name + "_" + percentile_name] = \
                                                        line_percentiles_to_store_dic["Line{0}".format(i)][j,:,k]

        if i ==0:
            df_lines_to_plot = df_line.copy()

        else:

            df_lines_to_plot = df_lines_to_plot.join(df_line)




    index_for_fluxes = [1]


    for i in range(len(line_names)):
        line_name = line_names[i]

        df_flux = pd.DataFrame()

        for j in range(len(index_for_fluxes)):

            array_name = array_per_line_names[index_for_fluxes[j]]

            df_flux[line_name + "_" + array_per_line_names[index_for_fluxes[j]] + "_fluxes"] = \
                         pd.Series(np.percentile(reconstructed_fluxes_dic["Line{0}".format(i)][:,index_for_fluxes[j]], \
                         [2.5, 15.9, 50, 84.1, 97.5]))

        if i ==0:
            df_reconstructed_fluxes = df_flux.copy()

        else:

            df_reconstructed_fluxes = df_reconstructed_fluxes.join(df_flux)




    df_variables_percentiles = pd.DataFrame()

    for variable in variables:

        df_variables_percentiles[variable+" value"] = variables[variable]['best']



    for i in range(len(line_names)):
        line_name = line_names[i]

        df_data = pd.DataFrame()

        for j in range(len(data_names)):

            df_data[data_names[j] + "_" + line_name] = data[j][i].astype('float64')

        if i ==0:
            df_data_to_save = df_data.copy()

        else:

            df_data_to_save = df_data_to_save.join(df_data)




    
    df_to_save = df_lines_to_plot.join([df_reconstructed_fluxes, df_variables_percentiles, df_data_to_save])


    df_to_save.to_csv(filename)
   





##############################
#                            #
# Make final plots           #
#                            #
##############################

if perform_error: 
    intrinsic_LyA_fluxes_to_histogram = reconstructed_fluxes_dic["Line0"][:,1]


    plt.figure()
    plt.hist(intrinsic_LyA_fluxes_to_histogram,bins=100)
    plt.xlabel('Intrinsic Ly$\\alpha$ flux (erg cm$^{-2}$ s$^{-1}$)',fontsize=18)


    df = pd.read_csv(filename)
    fig=plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax = plt.subplot(gs[0])
    axx = plt.subplot(gs[1])
    ax.step(df['wave_lya'],df['flux_lya'],where='mid',color='k')
    ax.errorbar(df['wave_lya'],df['flux_lya'],yerr=df['error_lya'],fmt='none',ecolor='limegreen',elinewidth=0.8)
    ax.plot(df['wave_lya'],df['lya_model_median'],color='deeppink',linewidth=1.5)
    ax.plot(df['wave_lya'],df['lya_intrinsic_median'],color='b',linestyle='--')
    ax2 = ax.twinx()

    mask = (df['wave_lya'] > 1214.6) & (df['wave_lya'] < 1216.46)
    ax2.plot(df['wave_lya'][mask],df['lya_ism_median'][mask],color='purple',linestyle=':')
    ax2.plot(df['wave_lya'][mask],df['lya_reversal_median'][mask],color='gold',linestyle=':')

    axx.errorbar(df['wave_lya'],df['flux_lya']-df['lya_model_median'],yerr=df['error_lya'],fmt='ko')
    #est_errs = (df['lya_model_high_1sig'] - df['lya_model_low_1sig'])/2.
    #axx.errorbar(df['wave_lya'],df['flux_lya']-df['lya_model_median'],yerr=np.sqrt(df['error_lya']**2 + est_errs**2),fmt='ko')

    #axx.set_ylim([-8e-13,8e-13])

    axx.set_xlabel('Wavelength (\AA)',fontsize=18)
    ax.set_ylabel('Flux Density (erg cm$^{-2}$ s$^{-1}$ \AA$^{-1}$)',fontsize=18)

    ax.minorticks_on()








