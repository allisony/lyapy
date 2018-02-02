import numpy as np
import pyspeckit ## only necessary if using MPFIT
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import astropy.table
import voigt ## this is my voigt.py file
import itertools ## only necessary for gridsearch
import multiprocessing ## only necessary for gridsearch
import scipy.stats
import sys

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
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
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

      print "Beginning Geocoronal Subtraction"
      lam_geocorona= raw_input("Wavelength center of geocorona: ")
      lamshft_geocorona = float(lam_geocorona) - lya_rest
      lam_o_geocorona = lya_rest + lamshft_geocorona
      vshft_geocorona = c_km * (lamshft_geocorona/lya_rest)

      ## line amplitude parameters
      peak_geocorona_string = raw_input("Peak of geocorona: ")
      peak_geocorona = float(peak_geocorona_string)
      base_geocorona_string = raw_input("Base of geocorona: ")
      base_geocorona = float(base_geocorona_string)
      dlam_geocorona_string = raw_input("Wavelength FWHM of geocorona: ")
      dlam_geocorona = float(dlam_geocorona_string)
      sig_geocorona =  dlam_geocorona/2.355
      ### create the functional form
      lz0_geocorona =  ( (wave_to_fit - lam_o_geocorona) / sig_geocorona )**2 
      geo_mod = (peak_geocorona * np.exp(-lz0_geocorona/2.0) ) 
  
      flux_mod_temp = flux_to_fit_copy - geo_mod
      plt.plot(wave_to_fit,geo_mod)
      plt.plot(wave_to_fit,flux_mod_temp)

      continue_parameter = raw_input("Happy? (yes or no): ")
      print ("you entered " + continue_parameter)
  
      if continue_parameter == 'yes':
        print "Stopping while loop"
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

    print ('Total EUV luminosity (100-1170 Angstroms) = ' + str( EUV_luminosity) + ' erg/s')
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
                               return_flux=False,single_component_flux=False,return_individual_components=False):

    """
    Given a wavelength array and parameters (velocity centroid, amplitude, and FWHM)
    of 2 Gaussians, computes the resulting intrinsic Lyman-alpha profile, and 
    optionally returns the flux.  Can also choose only 1 Gaussian emission component.

    """

    ## Narrow Ly-alpha ##

    lam_o_n = lya_rest * (1 + (vs_n/c_km))
    sig_n = lya_rest * (fw_n/c_km) / 2.3548

    lz0_n = ( (x - lam_o_n) / sig_n )**2
    lya_profile_narrow = ( am_n * np.exp(-lz0_n/2.0) )

    ## Broad Ly-alpha ##

    lam_o_b = lya_rest * (1 + (vs_b/c_km))
    sig_b = lya_rest * (fw_b/c_km) / 2.3548

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


def total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5):

    """
    Given a wavelength array and parameters (H column density, b value, and 
    velocity centroid), computes the Voigt profile of HI and DI Lyman-alpha
    and returns the combined absorption profile.

    """

    ##### ISM absorbers #####

    ## HI ##
   
    hwave_all,htau_all=tau_profile(h1_col,h1_vel,h1_b,'h1')
    tauh1=np.interp(wave_to_fit,hwave_all,htau_all)
    clean_htau_all = np.where(np.exp(-htau_all) > 1.0)
    htau_all[clean_htau_all] = 0.0

    ## DI ##

    d1_col = np.log10( (10.**h1_col)*d2h )

    dwave_all,dtau_all=tau_profile(d1_col,h1_vel,h1_b,'d1')
    taud1=np.interp(wave_to_fit,dwave_all,dtau_all)
    clean_dtau_all = np.where(np.exp(-dtau_all) > 1.0)
    dtau_all[clean_dtau_all] = 0.0


    ## Adding the optical depths and creating the observed profile ##

    tot_tau = tauh1 + taud1
    tot_ism = np.exp(-tot_tau)

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

def tau_profile(ncols,vshifts,vdop,h1_or_d1):

    """ 
    Computes a Lyman-alpha Voigt profile for HI or DI given column density,
    velocity centroid, and b parameter.

    """

    ## defining rest wavelength, oscillator strength, and damping parameter
    if h1_or_d1 == 'h1':
        lam0s,fs,gammas=1215.67,0.4161,6.26e8
    elif h1_or_d1 == 'd1':
        lam0s,fs,gammas=1215.3394,0.4161,6.27e8
    else:
        raise ValueError("h1_or_d1 can only equal 'h1' or 'd1'!")

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

    wave_edge=1210. # define wavelength cut off
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

  """ 
  Using this function (output aa) will allow the user to use a STIS LSF
  to convolve their model (lya_obs_high), so lyman_fit = np.convolve(lya_obs_high,aa,mode='same')
  to match the resolution/resolving power of their data (lyman_fit)
  orig_lsf_wave is the x array of the STIS LSF
  orig_lsf is the y array of the STIS LSF
  stis_grating_disp is the dispersion of the STIS grating for your chosen LSF (units: Ang/pix)
  data_wave is the wavelength array of your data

  """

  data_wave_spacing = data_wave[1]-data_wave[0]
  data_wave_length = len(data_wave)
  lsf_lam_min = np.round(np.min(orig_lsf_wave*stis_grating_disp)/data_wave_spacing) * data_wave_spacing
  lsf_lam_onesided = np.arange(lsf_lam_min,0,data_wave_spacing)  ### Make sure it's even and doesn't include zero
  if len(lsf_lam_onesided) % 2 != 0:
    lsf_lam_onesided = lsf_lam_onesided[1::] # get rid of the first element of the array

  lsf_lam_flipped = lsf_lam_onesided[::-1]
  lsf_lam_pluszero=np.append(lsf_lam_onesided,np.array([0]))
  lsf_lam=np.append(lsf_lam_pluszero,lsf_lam_flipped) # should be odd

  lsf_interp = np.interp(lsf_lam,orig_lsf_wave*stis_grating_disp,orig_lsf/np.sum(orig_lsf))
  lsf_interp_norm = lsf_interp/np.sum(lsf_interp) # I don't know why I do np.sum() for normalization...

  if data_wave_length < len(lsf_interp_norm):
      lsf_interp_norm = np.delete(lsf_interp_norm,np.where(lsf_interp_norm == 0))
      lsf_interp_norm = np.insert(lsf_interp_norm,0,0)
      lsf_interp_norm = np.append(lsf_interp_norm,0)

  return lsf_interp_norm

## Define priors and likelihoods, where parameters are fixed
def lnprior(theta, minmax):
    assert len(theta) == len(minmax)
    
    for i in range(len(theta)):
        if (theta[i] < minmax[i][0]) or (theta[i] > minmax[i][1]): # a parameter is out of range
            return -np.inf
    # ... no parameter was out of range
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    return np.log(h1_b)

def lnlike(theta, x, y, yerr, resolution, singcomp=False):
    vs_n, am_n, fw_n, vs_b, am_b, fw_b, h1_col, h1_b, h1_vel, d2h = theta
    y_model = damped_lya_profile(x,vs_n,10**am_n,fw_n,vs_b,10**am_b,fw_b,h1_col,
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
    ll = lnlike(theta_all, x, y, yerr, resolution = variables['d2h']['resolution'], 
                                       singcomp = variables['am_b']['single_comp'])
    return lp + ll


## Could include an example for how to change any of these to a Gaussian prior


