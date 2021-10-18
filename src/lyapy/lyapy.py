import numpy as np
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import astropy.table
from lyapy import voigt ## this is my voigt.py file
import scipy.stats
from astropy.modeling.models import Voigt1D, Lorentz1D, Gaussian1D

pc_in_cm = 3e18
au_in_cm = 1.5e13
lya_rest = 1215.67
c_km = 2.998e5

lya_rest = 1215.67
ccgs = 3e10
e=4.8032e-10            # electron charge in esu
mp=1.6726231e-24        # proton mass in grams
me=mp/1836.             # electron mass in grams


from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def geocoronal_subtraction(input_filename,resolution,starname,sub_geo=False):
    """ From a high resolution spectrum, interactively and iteratively fit
        a gaussian to the residual geocoronal emission and subtract it.  
        Saves a cropped version of the resulting spectrum (wave, flux, error),
        and in the header of the *_modAY.fits file, the star's name, spectral
        resolution, and the original filename the spectrum came from. """
        
    ## Editing spectrum! ##

    spec = pyfits.getdata(input_filename, 1)
    spec_header = pyfits.getheader(input_filename, 1)

    ## Define wave, flux, error, dq variables ##
    #wave = (spec['w0'] + spec['w1']) / 2.
    wave = spec['wavelength']
    flux = spec['flux']
    error = spec['error']

    flux_min = 0.0

    ## Define wavelength range to fit ##
    wave_range = [1213.7,1217.6]

    ## Crop the spectrum for fitting ##
    fit_indices = np.where( (wave > wave_range[0]) & (wave < wave_range[1]) )
    wave_to_fit = wave[fit_indices]
    flux_to_fit = flux[fit_indices]
    error_to_fit = error[fit_indices]


    flux_to_fit_copy = np.copy(flux_to_fit)

    ## Geocoronal Subtraction ##

    while sub_geo == True:
      plt.ion()
      plt.figure()


      plt.plot(wave_to_fit,flux_to_fit)

      print("Beginning Geocoronal Subtraction")
      lam_geocorona= input("Wavelength center of geocorona: ")
      lamshft_geocorona = float(lam_geocorona) - lya_rest
      lam_o_geocorona = lya_rest + lamshft_geocorona
      vshft_geocorona = c_km * (lamshft_geocorona/lya_rest)

      ## line amplitude parameters
      peak_geocorona_string = input("Peak of geocorona: ")
      peak_geocorona = float(peak_geocorona_string)
      base_geocorona_string = input("Base of geocorona: ")
      base_geocorona = float(base_geocorona_string)
      dlam_geocorona_string = input("Wavelength FWHM of geocorona: ")
      dlam_geocorona = float(dlam_geocorona_string)
      sig_geocorona =  dlam_geocorona/2.355
      ### create the functional form
      lz0_geocorona =  ( (wave_to_fit - lam_o_geocorona) / sig_geocorona )**2 
      geo_mod = (peak_geocorona * np.exp(-lz0_geocorona/2.0) ) 
  
      flux_mod_temp = flux_to_fit_copy - geo_mod
      plt.plot(wave_to_fit,geo_mod)
      plt.plot(wave_to_fit,flux_mod_temp)

      continue_parameter = input("Happy? (yes or no): ")
      print("you entered " + continue_parameter)
  
      if continue_parameter == 'yes':
        print("Stopping while loop")
        sub_geo = False
        flux_to_fit -=geo_mod
        #zero_indices = np.where(flux_to_fit < 0.0)
        #flux_to_fit[zero_indices] = flux_min

      plt.clf()

    plt.plot(wave_to_fit,flux_to_fit)

    output_data = np.array([wave_to_fit,flux_to_fit,error_to_fit])
    output_data_table = astropy.table.Table(np.transpose(output_data),
                             names=['wave', 'flux', 'error'],masked=True)
    output_data_table.meta['FILE'] = input_filename
    output_data_table.meta['RES_POW'] = resolution
    output_data_table.meta['STAR'] = starname
    output_filename = input_filename.replace('.fits','_modAY.fits')
    output_data_table.write(output_filename,overwrite=True)

    return

def EUV_spectrum(total_Lya_flux,distance_to_target,starname,
                 Mstar=True,savefile=True,doplot=True):

    """
    Computes the EUV spectrum of an M or K star from 100 - 1170 Angstroms
    based on the star's Ly-alpha flux at 1 au (Linsky et al. 2014).

    """

    total_Lya_flux_1au = total_Lya_flux * (distance_to_target*pc_in_cm/au_in_cm)**2

    ## Data from Linsky et al. 2014 Table 5
    ## Conversion factors (cf) for estimating EUV Fluxes in erg/cm^2/s
    ## log[f(waveband)/f(Lya)] = conversion factor
    ## f(waveband) = f(Lya) * 10^(conversion factor)
    ##### F5-M5 V Stars AT 1 AU!!!!
    logLya = np.log10(total_Lya_flux_1au)
    wavebands = [100,200,300,400,500,600,700,800,912,1171]
    wavebandwidth = [100.,100.,100.,100.,100.,100.,100.,112.,258.]

    cfs_M = [-0.491,-0.548,-0.602,-2.294 + 0.258 * logLya,-2.098 + 0.572 * logLya,
             -1.920 + 0.240 * logLya,-1.894 + 0.518 * logLya,-1.811 + 0.764 * logLya,
             -1.004 + 0.065 * logLya]
    cfs_K = [-1.357 + 0.344 * logLya,-1.300 + 0.309 * logLya,-0.882,
             -2.294 + 0.258 * logLya,-2.098 + 0.572 * logLya,
             -1.920 + 0.240 * logLya,-1.894 + 0.518 * logLya,-1.811 + 0.764 * logLya,
             -1.004 + 0.065 * logLya]
  
    if Mstar:
      cfs = cfs_M
    else:
      cfs = cfs_K

    EUV_wavelength_array = np.arange(100.,1171.,1.)
    EUV_flux_array = np.copy(EUV_wavelength_array)
    EUV_luminosity = 0.

    for i in range(len(cfs)):
    
      ## Applying correction factors
      waveband_flux = total_Lya_flux_1au * 10**cfs[i] / wavebandwidth[i]
      waveband_index = np.where((EUV_wavelength_array >= wavebands[i]) & 
                                (EUV_wavelength_array < wavebands[i+1]))
      EUV_flux_array[waveband_index] = waveband_flux
      #EUV_luminosity += (waveband_flux * (wavebandwidth[i]) * 4*np.pi*(au_in_cm)**2
   
    ## converting EUV_flux_array to distance from target -- still erg/s/cm^2/Angstrom

    EUV_flux_array *= (au_in_cm/(distance_to_target*pc_in_cm))**2
    
   
  
    if doplot:
      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(111)
      ax.plot(EUV_wavelength_array,EUV_flux_array,'Navy',linewidth=1.5)
      ax.set_yscale('log')
      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      ax.set_xlim([100,1170])
      #ax.set_title(starname + ' EUV spectrum',fontsize=16)
      #ax.text(0.97,0.97,r'Ly$\alpha$ flux = '+"{:.2E}".format(total_Lya_flux),
      #        verticalalignment='top', horizontalalignment='right',
      #        transform=ax.transAxes,fontsize=12., color='black')
      #ax.text(0.97,0.93,'[erg/s/cm2]',verticalalignment='top', 
      #        horizontalalignment='right',transform=ax.transAxes,fontsize=12., 
      #        color='black')

      EUV_luminosity_argument = float(("%e" % EUV_luminosity).split('e')[0])
      EUV_luminosity_exponent = float(("%e" % EUV_luminosity).split('e')[1])

      ax.text(0.97,0.89,'EUV luminosity = '+ str(round(EUV_luminosity_argument,2)) + r'$\times$10$^{' + str(int(EUV_luminosity_exponent)) + '}$',
              verticalalignment='top',horizontalalignment='right',
              transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.85,r'erg s$^{-1}$',verticalalignment='top',horizontalalignment='right',
              transform=ax.transAxes,fontsize=12., color='black')

      outfile_str = starname + '_EUV_spectrum.png'
      plt.savefig(outfile_str)

    EUV_luminosity = np.trapz(EUV_flux_array,EUV_wavelength_array) * 4*np.pi*(distance_to_target*pc_in_cm)**2

    print('Total EUV luminosity (100-1170 Angstroms) = ' + str( EUV_luminosity) + ' erg/s')
    if savefile:
      ## Writing EUV flux array to txt file
      #comment_str = 'Wavelength in Angstroms and EUV flux in erg/s/cm^2/AA'
      outfile_str = starname + '_EUV_spectrum.txt'
      np.savetxt(outfile_str,np.transpose([EUV_wavelength_array,EUV_flux_array]))#,header=comment_str)

    return EUV_wavelength_array,EUV_flux_array,EUV_luminosity


def gaussian_flux(amplitude,sigma):

    """ Computes the flux of a Gaussian given the amplitude and sigma. """

    flux = np.sqrt(2*np.pi) * sigma * amplitude

    return flux

def lya_intrinsic_profile_func(x,vs_n,am_n,fw_n,vs_b=2.,am_b=2.,fw_b=2.,
                               return_flux=False,single_component_flux=False,
                               return_individual_components=False, line_center = 1215.67):

    """
    Given a wavelength array and parameters (velocity centroid, amplitude, and FWHM)
    of 2 Gaussians, computes the resulting intrinsic Lyman-alpha profile, and 
    optionally returns the flux.  Can also choose only 1 Gaussian emission component.

    """

    ## Narrow Ly-alpha ##

    lam_o_n = line_center * (1 + (vs_n/c_km))
    sig_n = line_center * (fw_n/c_km) / 2.3548

    lz0_n = ( (x - lam_o_n) / sig_n )**2
    lya_profile_narrow = ( am_n * np.exp(-lz0_n/2.0) )

    ## Broad Ly-alpha ##

    lam_o_b = line_center * (1 + (vs_b/c_km))
    sig_b = line_center * (fw_b/c_km) / 2.3548

    lz0_b = ( (x - lam_o_b) / sig_b )**2
    lya_profile_broad = ( am_b * np.exp(-lz0_b/2.0) )

    ## Calculating intrinsic Lyman alpha flux
    lya_flux_narrow = gaussian_flux(am_n,sig_n)
    lya_flux_broad = gaussian_flux(am_b,sig_b)
    
    if return_flux:
      if not single_component_flux:
          if return_individual_components:
            return lya_profile_narrow,lya_profile_broad,lya_flux_narrow+lya_flux_broad
          else:
            return lya_profile_narrow+lya_profile_broad,lya_flux_narrow+lya_flux_broad
      else:
          return lya_profile_narrow,lya_flux_narrow
    else:
      if not single_component_flux:
        if return_individual_components:
          return lya_profile_narrow,lya_profile_broad
        else:
          return lya_profile_narrow+lya_profile_broad
      else:
        return lya_profile_narrow


def total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5,which_line='h1_d1',wave_cut_off=2.0):

    """
    Given a wavelength array and parameters (H column density, b value, and 
    velocity centroid), computes the Voigt profile of HI and DI Lyman-alpha
    and returns the combined absorption profile.

    """

    ##### ISM absorbers #####

    if which_line == 'h1_d1':

        ## HI ##
   
        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'h1',wave_cut_off=wave_cut_off)
        tauh1=np.interp(wave_to_fit,hwave_all,htau_all)

        ## DI ##

        d1_col = np.log10( (10.**h1_col)*d2h )

        dwave_all,dtau_all=tau_profile(d1_col,h1_vel,h1_b/np.sqrt(2.),'d1',wave_cut_off=wave_cut_off)
        taud1=np.interp(wave_to_fit,dwave_all,dtau_all)


        ## Adding the optical depths and creating the observed profile ##

        tot_tau = tauh1 + taud1
        tot_ism = np.exp(-tot_tau)

    else:

        hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,which_line,wave_cut_off=wave_cut_off)
        tau=np.interp(wave_to_fit,hwave_all,htau_all)

        tot_ism = np.exp(-tau)



    return tot_ism

def damped_lya_profile(wave_to_fit,vs_n,am_n,fw_n,vs_b,am_b,fw_b,h1_col,h1_b,
                       h1_vel,d2h,resolution,single_component_flux=False,
                       return_components=False,
                       return_hyperfine_components=False):

    """
    Computes a damped (attenuated) Lyman-alpha profile (by calling the functions
    lya_intrinsic_profile_func and total_tau_profile_func) and convolves it 
    to the proper resolution.

    """
    
    lya_intrinsic_profile = lya_intrinsic_profile_func(wave_to_fit,vs_n,am_n,fw_n,
                             vs_b,am_b,fw_b,single_component_flux=single_component_flux)
    total_tau_profile = total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h)

    lya_obs_high = lya_intrinsic_profile * total_tau_profile

    ## Convolving the data ##
    if type(resolution) is not float:
      aa = np.copy(resolution)
    else:
      fw = lya_rest/resolution
      aa = make_kernel(grid=wave_to_fit,fwhm=fw)

    lyman_fit = np.convolve(lya_obs_high,aa,mode='same')

    return lyman_fit*1e14

def damped_lya_profile_shortcut(wave_to_fit,resolution,lya_intrinsic_profile,total_tau_profile):

    """
    Computes a damped Lyman-alpha profile (by calling the functions
    lya_intrinsic_profile_func and total_tau_profile_func) and convolves it 
    to the proper resolution.

    """
    
    lya_obs_high = lya_intrinsic_profile * total_tau_profile

    ## Convolving the data ##
    if type(resolution) is not float:
      aa = np.copy(resolution)
    else:
      fw = lya_rest/resolution
      aa = make_kernel(grid=wave_to_fit,fwhm=fw)

    lyman_fit = np.convolve(lya_obs_high,aa,mode='same')

    return lyman_fit*1e14

def make_kernel(grid,fwhm,nfwhm=4.):

    """
    Creates a Gaussian kernel used for convolution to a certain resolution (resolving power).

    grid is the wavelength array of your data (not your model)

    fwhm = lya_rest/resolution where resolution is resolving power

    nfwhm helps controls the length of the kernel - default is 4
    and probably shouldn't be changed unless you have a good reason.
    """

    ngrd    = len(grid)
    spacing = (grid[ngrd-1]-grid[0])/(ngrd-1.)
    nkpts   = round(nfwhm*fwhm/spacing)

    if (nkpts % 2) != 0:
      nkpts += 1

    kernel = spacing* (np.arange(nkpts)-(nkpts/2.))
    kernel = np.exp(-np.log(2.0)/(fwhm/2.0)**2*(kernel)**2)  ## Gaussian kernel
    kernel_norm = kernel/np.sum(kernel)  ## Normalize
    kernel_norm=np.append(kernel_norm,kernel_norm[0])  ## Make sure it's symmetric

    return kernel_norm

def tau_profile(ncols,vshifts,vdop,which_line,wave_cut_off=2.0):

    """ 
    Computes a Lyman-alpha Voigt profile for HI or DI given column density,
    velocity centroid, and b parameter.

    """

    ## defining rest wavelength, oscillator strength, and damping parameter
    if which_line == 'h1':
        lam0s,fs,gammas=1215.67,0.4161,6.26e8
    elif which_line == 'd1':
        lam0s,fs,gammas=1215.3394,0.4161,6.27e8
    elif which_line == 'mg2_h':
        lam0s,fs,gammas=2796.3543,6.155E-01,2.625E+08
    elif which_line == 'mg2_k':
        lam0s,fs,gammas=2803.5315,3.058E-01,2.595E+08
    else:
        raise ValueError("which_line can only equal 'h1' or 'd1'!")

    Ntot=10.**ncols  # column density of H I gas
    nlam=4000       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # fun<D-O>ction of wavelength (one side of transition)
    u_parameter=np.zeros(nlam)  # Voigt "u" parameter
    nu0s=ccgs/(lam0s*1e-8)  # wavelengths of Lyman alpha in frequency
    nuds=nu0s*vdop/c_km    # delta nus based off vdop parameter
    a_parameter = np.abs(gammas/(4.*np.pi*nuds) ) # Voigt "a" parameter -- damping parameter
    xsections_nearlinecenter = np.sqrt(np.pi)*(e**2)*fs*lam0s/(me*ccgs*vdop*1e13)  # cross-sections 
                                                                           # near Lyman line center

    
    wave_edge=lam0s - wave_cut_off # define wavelength cut off - this is important for the brightest lines and should be increased appropriately.
    wave_symmetrical=np.zeros(2*nlam-1) # huge wavelength array centered around a Lyman transition
    wave_onesided = np.zeros(nlam)  # similar to wave_symmetrical, but not centered 
                                    # around a Lyman transition 
    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter

    ## find end point for wave_symmetrical array and create wave_symmetrical array
    num_elements = 2*nlam - 1
    first_point = wave_edge
 
    mid_point = lam0s
    end_point = 2*(mid_point - first_point) + first_point
    wave_symmetrical = np.linspace(first_point,end_point,num=num_elements)
    wave_onesided = np.linspace(lam0s,wave_edge,num=nlam)

    freq_onesided = ccgs / (wave_onesided*1e-8)  ## convert "wave_onesided" array to a frequency array

    u_parameter = (freq_onesided-nu0s)/nuds  ## Voigt "u" parameter -- dimensionless frequency offset

    xsections_onesided=xsections_nearlinecenter*voigt.voigt(a_parameter,u_parameter)  ## cross-sections
                                                                                # single sided
                                                                                ## can't do symmetrical 

    xsections_onesided_flipped = xsections_onesided[::-1]
    
    		
    ## making the cross-sections symmetrical
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all


def mask_spectrum(flux_to_fit,interactive=True,mask_lower_limit=None,mask_upper_limit=None):

    """
    Interactively and iteratively creates a Boolean mask for a spectrum.

    """
    if interactive:
      plt.ion()
      continue_parameter = 'no'
      mask_switch = 'yes'
      while mask_switch == 'yes':
        pixel_array = np.arange(len(flux_to_fit))
        plt.figure()
        plt.step(pixel_array,flux_to_fit)
        mask_lower_limit_string = raw_input("Enter mask lower limit (in pixels): ")
        mask_lower_limit = float(mask_lower_limit_string)
        mask_upper_limit_string = raw_input("Enter mask upper limit (in pixels): ")
        mask_upper_limit = float(mask_upper_limit_string)
        mask = (pixel_array >= mask_lower_limit) & (pixel_array <= mask_upper_limit)
        flux_to_fit_masked = np.ma.masked_where(mask,flux_to_fit)
        plt.step(pixel_array,flux_to_fit_masked)
        continue_parameter = raw_input("Happy? (yes or no]): ")
        plt.close()

        if continue_parameter == 'yes':
          mask_switch = 'no'

    else:
      pixel_array = np.arange(len(flux_to_fit))
      mask = (pixel_array >= mask_lower_limit) & (pixel_array <= mask_upper_limit)

    return mask



def ready_stis_lsf(orig_lsf_wave,orig_lsf,stis_grating_disp,data_wave):


  data_wave_spacing = data_wave[1]-data_wave[0]
  data_wave_length = len(data_wave)

  # define a new LSF wavelength array (lsf_lam) to interpolate orig_lsf onto. lsf_lam has the same wavelength spacing as the data array
  # 
  lsf_lam_min = np.round(np.min(orig_lsf_wave*stis_grating_disp)/data_wave_spacing) * data_wave_spacing # minimum wavelength for lsf_lam
  lsf_lam_onesided = np.arange(lsf_lam_min,0,data_wave_spacing)  
  if len(lsf_lam_onesided) % 2 != 0: ### Make sure it's even and doesn't include zero
    lsf_lam_onesided = lsf_lam_onesided[1::] # get rid of the first element of the array if lsf_lam_onesided is odd

  lsf_lam_flipped = -1*lsf_lam_onesided[::-1] # flip it to make it symmetrical
  lsf_lam_pluszero=np.append(lsf_lam_onesided,np.array([0])) # add zero
  lsf_lam=np.append(lsf_lam_pluszero,lsf_lam_flipped) # add the flipped array - lsf_lam should be odd

  lsf_interp = np.interp(lsf_lam,orig_lsf_wave*stis_grating_disp,orig_lsf/np.sum(orig_lsf))
  lsf_interp_norm = lsf_interp/np.sum(lsf_interp)#np.trapz(lsf_interp,lsf_lam) 

  if data_wave_length < len(lsf_interp_norm):
      lsf_interp_norm = np.delete(lsf_interp_norm,np.where(lsf_interp_norm == 0))
      lsf_interp_norm = np.insert(lsf_interp_norm,0,0)
      lsf_interp_norm = np.append(lsf_interp_norm,0)

  return lsf_interp_norm

## Define priors and likelihoods, where parameters are fixed
def lnprior(theta, minmax, prior_Gauss, prior_list):
    assert len(theta) == len(minmax)

    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    
    priors = 0

    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
        if prior_Gauss[i]: # Gaussian prior, else uniform prior (0 or np.log(h1_b))
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        elif (i == 7) and not prior_Gauss[7]: # h1_b
            priors += np.log(h1_b)
        else:
            priors += 0 # not necessary, just included for clarity/completeness
            
    # ... no parameter was out of range
    
    return priors

def lnlike(theta, x, y, yerr, resolution, singcomp=False, HAW=False):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    #y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
    #                                   h1_b,h1_vel,d2h=d2h,resolution=resolution,
    #                                   single_component_flux=singcomp)/1e14
    intr_profile = lya_intrinsic_profile_func(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,
                   single_component_flux=singcomp)
    tau = total_tau_profile_func(x,h1_col,h1_b,h1_vel,d2h=d2h)
    y_model = damped_lya_profile_shortcut(x,resolution,intr_profile,tau)/1e14

    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)




def lnprob(theta, x, y, yerr, variables):
    order = ['vs_n', 'am_n', 'fw_n', 'vs_b', 'am_b', 'fw_b', 'h1_col', 'h1_b', 'h1_vel', 'd2h']
    theta_all = []
    range_all = []
    prior_Gauss = [] # Boolean list for whether or not the parameter has a Gaussian prior
    prior_list = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
        if variables[p]['Gaussian prior']:
            prior_Gauss.append(True)
            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])
        else:
            prior_Gauss.append(False)
            prior_list.append(0)
                
    assert (i) == len(theta)
    lp = lnprior(theta_all, range_all, prior_Gauss, prior_list)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike(theta_all, x, y, yerr, resolution = variables['d2h']['resolution'],
                                       singcomp = variables['am_b']['single_comp'])
    return lp + ll



def lnprior_g140l(theta, minmax, prior_Gauss, prior_list):
    assert len(theta) == len(minmax)

    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h, vs_haw, am_haw, fw_haw, vs_SiIII, am_SiIII, fw_SiIII = theta
    
    priors = 0
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
        if prior_Gauss[i]: # Gaussian prior, else uniform prior (0 or np.log(h1_b))
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        elif (i == 7) and not prior_Gauss[7]: # h1_b
            priors += np.log(h1_b)
        else:
            priors += 0 # not necessary, just included for clarity/completeness
            
    # ... no parameter was out of range
    
    return priors

###

def lnlike_g140l(theta, x, y, yerr, resolution, singcomp=False, HAW=False, SiIII=False):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h, vs_haw, am_haw, fw_haw, vs_SiIII, am_SiIII, fw_SiIII = theta
    #y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
    #                                   h1_b,h1_vel,d2h=d2h,resolution=resolution,
    #                                   single_component_flux=singcomp)/1e14
    intr_profile = lya_intrinsic_profile_func(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,
                   single_component_flux=singcomp)
    if HAW: #Huge Ass Wings
      haw_profile = lya_intrinsic_profile_func(x,vs_haw,10**am_haw,fw_haw,single_component_flux=True)
      intr_profile += haw_profile
    if SiIII: #Fit SiIII at 1206 Angstroms
      SiIII_profile = lya_intrinsic_profile_func(x,vs_SiIII,10**am_SiIII,fw_SiIII,
                      single_component_flux=True, line_center=1206.50)
      intr_profile += SiIII_profile
    tau = total_tau_profile_func(x,h1_col,h1_b,h1_vel,d2h=d2h)
    y_model = damped_lya_profile_shortcut(x,resolution,intr_profile,tau)/1e14

    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)


def lnprob_g140l(theta, x, y, yerr, variables):
    order = ['vs_n', 'am_n', 'fw_n', 'vs_b', 'am_b', 'fw_b', 'h1_col', 'h1_b', 'h1_vel', 'd2h',
             'vs_haw', 'am_haw', 'fw_haw', 'vs_SiIII', 'am_SiIII', 'fw_SiIII']
    theta_all = []
    range_all = []
    prior_Gauss = [] # Boolean list for whether or not the parameter has a Gaussian prior
    prior_list = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
        if variables[p]['Gaussian prior']:
            prior_Gauss.append(True)
            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])
        else:
            prior_Gauss.append(False)
            prior_list.append(0)
                
    assert (i) == len(theta)
    lp = lnprior_g140l(theta_all, range_all, prior_Gauss, prior_list)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike_g140l(theta_all, x, y, yerr, resolution = variables['d2h']['resolution'],
                                       singcomp = variables['am_b']['single_comp'],
                                       HAW = variables['vs_haw']['HAW'],
                                       SiIII = variables['vs_SiIII']['SiIII'])
    return lp + ll



def lnprior_voigt(theta, minmax, prior_Gauss, prior_list):
    assert len(theta) == len(minmax)

    vs, am, fw_L, fw_G, h1_col, h1_b, h1_vel, d2h, vs_SiIII, am_SiIII, fw_SiIII, ln_f = theta
    
    priors = 0
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
        if prior_Gauss[i]: # Gaussian prior, else uniform prior (0 or np.log(h1_b))
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        elif (i == 7) and not prior_Gauss[7]: # h1_b
            priors += np.log(h1_b)
        else:
            priors += 0 # not necessary, just included for clarity/completeness
            
    # ... no parameter was out of range
    
    return priors

###

def lnlike_voigt(theta, x, y, yerr, resolution, singcomp=False, HAW=False, SiIII=False, Lorentzian_only=False,
                 match_v_HI_SiIII=False,fw_G_fw_L_fixed_ratio=0,fix_stellar_ISM_RV_diff=False):
    vs, am, fw_L, fw_G, h1_col, h1_b, h1_vel, d2h, vs_SiIII, am_SiIII, fw_SiIII, ln_f = theta
    #y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
    #                                   h1_b,h1_vel,d2h=d2h,resolution=resolution,
    #                                   single_component_flux=singcomp)/1e14

    if fix_stellar_ISM_RV_diff:
        h1_vel = h1_vel + vs

    line_center = vs/3e5*1215.67+1215.67
    sigma_G = fw_G/3e5 * 1215.67 #/ 2.3548
    if fw_G_fw_L_fixed_ratio != 0:
        if fw_G_fw_L_fixed_ratio == -1:
            sigma_L = fw_G * fw_L /3e5 * 1215.67 / 2.
        else:
            sigma_L = fw_G * fw_G_fw_L_fixed_ratio /3e5 * 1215.67 / 2.
    else:
        sigma_L = fw_L/3e5 * 1215.67 #/ 2.
        
    if Lorentzian_only:
        voigt_profile_func = Lorentz1D(x_0 = line_center, amplitude = 10**am, fwhm = sigma_L)
    else:
        voigt_profile_func = Voigt1D(x_0 = line_center, amplitude_L = 10**am, fwhm_L = sigma_L, fwhm_G = sigma_G) 
    intr_profile = voigt_profile_func(x)
    if SiIII: #Fit SiIII at 1206 Angstroms
      if match_v_HI_SiIII:
        SiIII_profile = lya_intrinsic_profile_func(x,vs,10**am_SiIII,fw_SiIII,
                      single_component_flux=True, line_center=1206.50)
      else:
        SiIII_profile = lya_intrinsic_profile_func(x,vs_SiIII,10**am_SiIII,fw_SiIII,
                      single_component_flux=True, line_center=1206.50)
      intr_profile += SiIII_profile
    tau = total_tau_profile_func(x,h1_col,h1_b,h1_vel,d2h=d2h)
    y_model = damped_lya_profile_shortcut(x,resolution,intr_profile,tau)/1e14

    sigma2 = yerr**2 + np.exp(2*ln_f) #* y_model**2 

    #return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)
    return -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (y - y_model) ** 2 / sigma2)


def lnprob_voigt(theta, x, y, yerr, variables):
    order = ['vs', 'am', 'fw_L', 'fw_G', 'h1_col', 'h1_b', 'h1_vel', 'd2h',
             'vs_SiIII', 'am_SiIII', 'fw_SiIII','ln_f']
    theta_all = []
    range_all = []
    prior_Gauss = [] # Boolean list for whether or not the parameter has a Gaussian prior
    prior_list = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
        if variables[p]['Gaussian prior']:
            prior_Gauss.append(True)
            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])
        else:
            prior_Gauss.append(False)
            prior_list.append(0)
                
    assert (i) == len(theta)
    lp = lnprior_voigt(theta_all, range_all, prior_Gauss, prior_list)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike_voigt(theta_all, x, y, yerr, resolution = variables['d2h']['resolution'],
                            SiIII = variables['vs_SiIII']['SiIII'],
                            Lorentzian_only = variables['fw_L']['Lorentzian'],
                            match_v_HI_SiIII = variables['vs']['match_v_HI_SiIII'],
                            fix_stellar_ISM_RV_diff = variables['h1_vel']['fix_stellar_ISM_RV_diff'],
                            fw_G_fw_L_fixed_ratio = variables['fw_G']['fw_G_fw_L_fixed_ratio'])
    #print(lp+ll)
    return lp + ll



def lnprior_voigt_cii(theta, minmax, prior_Gauss, prior_list):
    assert len(theta) == len(minmax)

    vs, am, fw, h1_col, h1_b, h1_vel = theta
    
    priors = 0
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
        if prior_Gauss[i]: # Gaussian prior, else uniform prior (0 or np.log(h1_b))
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        elif (i == 7) and not prior_Gauss[7]: # h1_b
            priors += np.log(h1_b)
        else:
            priors += 0 # not necessary, just included for clarity/completeness
            
    # ... no parameter was out of range
    
    return priors

###

def lnlike_voigt_cii(theta, x, y, yerr, resolution):
    vs, am, fw, h1_col, h1_b, h1_vel = theta
    #y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
    #                                   h1_b,h1_vel,d2h=d2h,resolution=resolution,
    #                                   single_component_flux=singcomp)/1e14
    intr_profile = lya_intrinsic_profile_func(x,vs,10**am,fw,
                   single_component_flux=True, line_center=1334.532)
    SiIII_profile = lya_intrinsic_profile_func(x,vs,2.*(10**am),fw,
                      single_component_flux=True, line_center=1335.708)
    intr_profile += SiIII_profile
    tau = total_tau_profile_func_cii(x,h1_col,h1_b,h1_vel)
    y_model = damped_lya_profile_shortcut(x,resolution,intr_profile,tau)/1e14

    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)


def lnprob_voigt_cii(theta, x, y, yerr, variables):
    order = ['vs', 'am', 'fw', 'h1_col', 'h1_b', 'h1_vel']
    theta_all = []
    range_all = []
    prior_Gauss = [] # Boolean list for whether or not the parameter has a Gaussian prior
    prior_list = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
        if variables[p]['Gaussian prior']:
            prior_Gauss.append(True)
            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])
        else:
            prior_Gauss.append(False)
            prior_list.append(0)
                
    assert (i) == len(theta)
    lp = lnprior_voigt_cii(theta_all, range_all, prior_Gauss, prior_list)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike_voigt_cii(theta_all, x, y, yerr, resolution = variables['am']['resolution'])
    return lp + ll



def total_tau_profile_func_cii(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5):

    """
    Given a wavelength array and parameters (H column density, b value, and 
    velocity centroid), computes the Voigt profile of HI and DI Lyman-alpha
    and returns the combined absorption profile.

    """

    ##### ISM absorbers #####

    ## HI ##
   
    hwave_all,htau_all=tau_profile_cii(h1_col,h1_vel,h1_b)
    tauh1=np.interp(wave_to_fit,hwave_all,htau_all)
    clean_htau_all = np.where(np.exp(-htau_all) > 1.0)
    htau_all[clean_htau_all] = 0.0



    ## Adding the optical depths and creating the observed profile ##

    tot_tau = tauh1
    tot_ism = np.exp(-tot_tau)

    return tot_ism

def tau_profile_cii(ncols,vshifts,vdop):

    """ 
    Computes a Lyman-alpha Voigt profile for HI or DI given column density,
    velocity centroid, and b parameter.

    """

    ## defining rest wavelength, oscillator strength, and damping parameter
    lam0s,fs,gammas=1334.532,0.129,5.2e8

    Ntot=10.**ncols  # column density of H I gas
    nlam=1000       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # fun<D-O>ction of wavelength (one side of transition)
    u_parameter=np.zeros(nlam)  # Voigt "u" parameter
    nu0s=ccgs/(lam0s*1e-8)  # wavelengths of Lyman alpha in frequency
    nuds=nu0s*vdop/c_km    # delta nus based off vdop parameter
    a_parameter = np.abs(gammas/(4.*np.pi*nuds) ) # Voigt "a" parameter -- damping parameter
    xsections_nearlinecenter = np.sqrt(np.pi)*(e**2)*fs*lam0s/(me*ccgs*vdop*1e13)  # cross-sections 
                                                                           # near Lyman line center

    wave_edge=1325. # define wavelength cut off
    wave_symmetrical=np.zeros(2*nlam-1) # huge wavelength array centered around a Lyman transition
    wave_onesided = np.zeros(nlam)  # similar to wave_symmetrical, but not centered 
                                    # around a Lyman transition 
    lamshifts=lam0s*vshifts/c_km  # wavelength shifts from vshifts parameter

    ## find end point for wave_symmetrical array and create wave_symmetrical array
    num_elements = 2*nlam - 1
    first_point = wave_edge
 
    mid_point = lam0s
    end_point = 2*(mid_point - first_point) + first_point
    wave_symmetrical = np.linspace(first_point,end_point,num=num_elements)
    wave_onesided = np.linspace(lam0s,wave_edge,num=nlam)

    freq_onesided = ccgs / (wave_onesided*1e-8)  ## convert "wave_onesided" array to a frequency array

    u_parameter = (freq_onesided-nu0s)/nuds  ## Voigt "u" parameter -- dimensionless frequency offset

    xsections_onesided=xsections_nearlinecenter*voigt.voigt(a_parameter,u_parameter)  ## cross-sections
                                                                                # single sided
                                                                                ## can't do symmetrical 

    xsections_onesided_flipped = xsections_onesided[::-1]
    
    		
    ## making the cross-sections symmetrical
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all




def lnprior_voigt_rev(theta, minmax, prior_Gauss, prior_list):
    assert len(theta) == len(minmax)

    vs, am, fw_L, fw_G, h1_col, h1_b, h1_vel, d2h, vs_rev, am_rev, fw_rev = theta
    
    priors = 0
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
        if prior_Gauss[i]: # Gaussian prior, else uniform prior (0 or np.log(h1_b))
            priors += -0.5*((theta[i]-prior_list[i][0])/prior_list[i][1])**2
        elif (i == 7) and not prior_Gauss[7]: # h1_b
            priors += np.log(h1_b)
        else:
            priors += 0 # not necessary, just included for clarity/completeness
            
    # ... no parameter was out of range
    
    return priors

###

def lnlike_voigt_rev(theta, x, y, yerr, resolution, singcomp=False, HAW=False, rev=False, Lorentzian_only=False,
                 fw_G_fw_L_fixed_ratio=0):
    vs, am, fw_L, fw_G, h1_col, h1_b, h1_vel, d2h, vs_rev, am_rev, fw_rev = theta
    #y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
    #                                   h1_b,h1_vel,d2h=d2h,resolution=resolution,
    #                                   single_component_flux=singcomp)/1e14
    line_center = vs/3e5*1215.67+1215.67
    sigma_G = fw_G/3e5 * 1215.67 #/ 2.3548
    if fw_G_fw_L_fixed_ratio != 0:
        if fw_G_fw_L_fixed_ratio == -1:
            sigma_L = fw_G * fw_L /3e5 * 1215.67 #/ 2.
        else:
            sigma_L = fw_G * fw_G_fw_L_fixed_ratio /3e5 * 1215.67 #/ 2.
    else:
        sigma_L = fw_L/3e5 * 1215.67 #/ 2.
        
    if Lorentzian_only:
        voigt_profile_func = Lorentz1D(x_0 = line_center, amplitude = 10**am, fwhm = sigma_L)
    else:
        voigt_profile_func = Voigt1D(x_0 = line_center, amplitude_L = 10**am, fwhm_L = sigma_L, fwhm_G = sigma_G) 
    intr_profile = voigt_profile_func(x)
    if rev: #Fit LyA self reversal
        g_func = Gaussian1D(mean = vs_rev/3e5*1215.67+1215.67, amplitude=am_rev, 
                stddev=fw_rev/3e5*1215.67/2.3548)
        rev_profile = g_func(x)
        intr_profile *= (rev_profile + 1)

    tau = total_tau_profile_func(x,h1_col,h1_b,h1_vel,d2h=d2h)
    y_model = damped_lya_profile_shortcut(x,resolution,intr_profile,tau)/1e14

    return -0.5 * np.sum(np.log(2 * np.pi * yerr**2) + (y - y_model) ** 2 / yerr**2)


def lnprob_voigt_rev(theta, x, y, yerr, variables):
    order = ['vs', 'am', 'fw_L', 'fw_G', 'h1_col', 'h1_b', 'h1_vel', 'd2h',
             'vs_rev', 'am_rev', 'fw_rev']
    theta_all = []
    range_all = []
    prior_Gauss = [] # Boolean list for whether or not the parameter has a Gaussian prior
    prior_list = []
    i = 0
    for p in order:
        range_all.append( [variables[p]['min'],variables[p]['max']] )
        if variables[p]['vary']:
            theta_all.append(theta[i])
            i = i+1
        else:
            theta_all.append(variables[p]['value'])
        if variables[p]['Gaussian prior']:
            prior_Gauss.append(True)
            prior_list.append([variables[p]['prior mean'],variables[p]['prior stddev']])
        else:
            prior_Gauss.append(False)
            prior_list.append(0)
                
    assert (i) == len(theta)
    lp = lnprior_voigt_rev(theta_all, range_all, prior_Gauss, prior_list)
    if not np.isfinite(lp):
        return -np.inf

    #if np.random.uniform() > 0.9995: print "took a step!", theta_all
    ll = lnlike_voigt_rev(theta_all, x, y, yerr, resolution = variables['d2h']['resolution'],
                                       rev = variables['vs_rev']['rev'],
                                       Lorentzian_only = variables['fw_L']['Lorentzian'],
                                       fw_G_fw_L_fixed_ratio = variables['fw_G']['fw_G_fw_L_fixed_ratio'])
    return lp + ll



def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorr_func_1d(x, norm=True):
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= 4*n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1

# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(np.mean(y, axis=0))
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

def autocorr_new(y, c=5.0):
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0*np.cumsum(f)-1.0
    window = auto_window(taus, c)
    return taus[window]

