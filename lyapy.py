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

    spec = pyfits.getdata(input_filename)
    spec_header = pyfits.getheader(input_filename)

    ## Define wave, flux, error, dq variables ##
    wave = (spec['w0'] + spec['w1']) / 2.
    #wave = spec['wave']
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
    fw = lya_rest/resolution
    aa = make_kernel(grid=wave_to_fit,fwhm=fw)
    lyman_fit = np.convolve(lya_obs_high,aa,mode='same')

    return lyman_fit*1e14


def make_kernel(grid,fwhm):

    """
    Creates a kernel used for convolution to a certain resolution.

    """

    nfwhm = 4.  ## No idea what this parameter is
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
    nlam=1000.       # number of elements in the wavelength grid
    xsections_onesided=np.zeros(nlam)  # absorption cross sections as a 
                                       # function of wavelength (one side of transition)
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

def damped_lya_fitter(multisingle='multi'):

    """
    Generator for Damped LyA fitter class -- this for using MPFIT

    """
    myclass =  pyspeckit.models.model.SpectralModel(damped_lya_profile,11,
            parnames=['vs_n','am_n','fw_n','vs_b','am_b','fw_b','h1_col',
                      'h1_b','h1_vel','d2h','resolution'], 
            parlimited=[(False,False),(True,False),(True,False),(False,False),
                        (True,False),(True,False),(True,True),(True,True),
                        (False,False),(True,True),(True,False)], 
            parlimits=[(14.4,14.6), (1e-16,0), (50.,0),(0,0), (1e-16,0),(50.,0),
                       (17.0,19.5), (5.,20.), (0,0),(1.0e-5,2.5e-5),(10000.,0)])
    myclass.__name__ = "damped_lya"
    
    return myclass

def tau_gridsearch(wave_to_fit,h1_col_range,h1_b_range,h1_vel_range,resolution,
                   mask,save_file=True):

    fw = lya_rest/resolution

    data_to_fit_fromfile = np.loadtxt(data_to_fit_filename)
    wave_to_fit_fromfile = data_to_fit_fromfile[0,:]

    kernel = make_kernel(grid=wave_to_fit_fromfile,fwhm=fw)  

    tau_tot_ism_test = total_tau_profile_func(wave_to_fit,h1_col_range[0],
                                                  h1_b_range[0],h1_vel_range[0])
    tau_tot_ism_test_convolved = np.convolve(tau_tot_ism_test,kernel,mode='same')

    tau_tot_ism_grid = np.zeros([len(h1_col_range),len(h1_b_range),len(h1_vel_range),
                                                 len(tau_tot_ism_test_convolved)])
    for a in range(len(h1_col_range)):
      print '*** a (h1_col) = ' + str(a) + ' of ' + str(len(h1_col_range)-1) + ' ***' 
      for b in range(len(h1_b_range)):
        print 'b (h1_b) = ' + str(b) + ' of ' + str(len(h1_b_range)-1)
        for c in range(len(h1_vel_range)):
      
          tot_ism = total_tau_profile_func(wave_to_fit,h1_col_range[a],
                                               h1_b_range[b],h1_vel_range[c]) 
          tot_ism_convolved = np.convolve(tot_ism,kernel,mode='same')
          if mask.sum() != 0:
            tot_ism_convolved[mask] = 0
          tau_tot_ism_grid[a,b,c,:] = tot_ism_convolved
    
    if save_file:
      hdu = pyfits.PrimaryHDU(data=tau_tot_ism_grid)
      hdu.writeto('tau_tot_ism_grid.fits',clobber=True)

    return

def intrinsic_lya_gridsearch(wave_to_fit,vs_n_range,am_n_range,fw_n_range,vs_b_range,
                             am_b_range,fw_b_range,resolution,mask,
                             single_component_flux=False,save_file=True):

    fw = lya_rest/resolution

    data_to_fit_fromfile = np.loadtxt(data_to_fit_filename)
    wave_to_fit_fromfile = data_to_fit_fromfile[0,:]

    kernel = make_kernel(grid=wave_to_fit_fromfile,fwhm=fw)  

    if not single_component_flux:
      lya_intrinsic_profile_test = lya_intrinsic_profile_func(wave_to_fit,vs_n_range[0],
                                 am_n_range[0],fw_n_range[0],vs_b_range[0],
                                 am_b_range[0],fw_b_range[0],
                                 single_component_flux=single_component_flux)
      lya_intrinsic_profile_test_convolved = np.convolve(lya_intrinsic_profile_test,
                                           kernel,mode='same')

      lya_intrinsic_profile_grid = np.zeros([len(vs_n_range),len(am_n_range),
                                           len(fw_n_range),len(vs_b_range),
                                           len(am_b_range),len(fw_b_range),
                                           len(lya_intrinsic_profile_test_convolved)])
      lya_flux_grid = np.zeros([len(vs_n_range),len(am_n_range),
                                           len(fw_n_range),len(vs_b_range),
                                           len(am_b_range),len(fw_b_range)])

    else:
      lya_intrinsic_profile_test = lya_intrinsic_profile_func(wave_to_fit,vs_n_range[0],
                                 am_n_range[0],fw_n_range[0],
                                 single_component_flux=single_component_flux)
      lya_intrinsic_profile_test_convolved = np.convolve(lya_intrinsic_profile_test,
                                           kernel,mode='same')
      lya_intrinsic_profile_grid = np.zeros([len(vs_n_range),len(am_n_range),
                                           len(fw_n_range),         
                                           len(lya_intrinsic_profile_test_convolved)])
      lya_flux_grid = np.zeros([len(vs_n_range),len(am_n_range),
                                           len(fw_n_range)])


    if not single_component_flux:
      for a in range(len(vs_n_range)):
        print '*** a (vs_n) = ' + str(a) + ' of ' + str(len(vs_n_range)-1) + ' ***' 
        for b in range(len(am_n_range)):
          print 'b (am_n) = ' + str(b) + ' of ' + str(len(am_n_range)-1)
          for c in range(len(fw_n_range)):

            for d in range(len(vs_b_range)):
   
              for e in range(len(am_b_range)):

                for f in range(len(fw_b_range)):
      
                  lya_intrinsic_profile,lya_flux = lya_intrinsic_profile_func(wave_to_fit,
                                            vs_n_range[a],am_n_range[b],fw_n_range[c],
                                            vs_b_range[d],am_b_range[e],fw_b_range[f],
                                            single_component_flux=single_component_flux,return_flux=True)
                  lya_intrinsic_profile_convolved = np.convolve(lya_intrinsic_profile,
                                                                 kernel,mode='same')
                  if mask.sum() != 0:
                    lya_intrinsic_profile_convolved[mask] = 0

                  lya_intrinsic_profile_grid[a,b,c,d,e,f,:] = lya_intrinsic_profile_convolved
                  lya_flux_grid[a,b,c,d,e,f] = lya_flux

    else:
      for a in range(len(vs_n_range)):
        print '*** a (vs_n) = ' + str(a) + ' of ' + str(len(vs_n_range)-1) + ' ***' 
        for b in range(len(am_n_range)):
          print 'b (am_n) = ' + str(b) + ' of ' + str(len(am_n_range)-1)
          for c in range(len(fw_n_range)):
      
                  lya_intrinsic_profile,lya_flux = lya_intrinsic_profile_func(wave_to_fit,
                                            vs_n_range[a],am_n_range[b],fw_n_range[c],
                                            single_component_flux=single_component_flux,return_flux=True)
                  lya_intrinsic_profile_convolved = np.convolve(lya_intrinsic_profile,
                                                                 kernel,mode='same')
                  if mask.sum() != 0:
                    lya_intrinsic_profile_convolved[mask] = 0

                  lya_intrinsic_profile_grid[a,b,c,:] = lya_intrinsic_profile_convolved
                  lya_flux_grid[a,b,c] = lya_flux


    
    if save_file:
      hdu = pyfits.PrimaryHDU(data=lya_intrinsic_profile_grid)
      hdu.writeto('lya_intrinsic_profile_grid.fits',clobber=True)
      hdu2 = pyfits.PrimaryHDU(data=lya_flux_grid)
      hdu2.writeto('lya_flux_grid.fits',clobber=True)


    return

def brute_force_gridsearch(vs_n_range,am_n_range,fw_n_range,vs_b_range,am_b_range,
                           fw_b_range,h1_col_range,h1_b_range,h1_vel_range,mask,
                           single_component_flux=False):

    if not single_component_flux:
      big_chisq_grid = np.zeros([len(vs_n_range),len(am_n_range),len(fw_n_range),
                               len(vs_b_range),len(am_b_range),len(fw_b_range),
                               len(h1_col_range),len(h1_b_range),len(h1_vel_range)])
    else:
      big_chisq_grid = np.zeros([len(vs_n_range),len(am_n_range),len(fw_n_range),
                                 len(h1_col_range),len(h1_b_range),len(h1_vel_range)])


    data_to_fit_fromfile = np.loadtxt(data_to_fit_filename)
    wave_to_fit_fromfile = data_to_fit_fromfile[0,:]
    flux_to_fit_fromfile = data_to_fit_fromfile[1,:]
    error_to_fit_fromfile = data_to_fit_fromfile[2,:]
    tau_tot_ism_grid_fromfile = pyfits.getdata(tau_tot_ism_grid_filename)
    lya_intrinsic_profile_grid_fromfile = pyfits.getdata(lya_intrinsic_profile_grid_filename)

    if not single_component_flux:
      for a in range(len(vs_n_range)):
        print '*** a = ' + str(a) + ' of ' + str(len(vs_n_range)-1) + ' ***' 
        for b in range(len(am_n_range)):
          print 'b = ' + str(b) + ' of ' + str(len(am_n_range)-1)
          for c in range(len(fw_n_range)):
            print 'c = ' + str(c) + ' of ' + str(len(fw_n_range)-1)
            for d in range(len(vs_b_range)):

              for e in range(len(am_b_range)):

                for f in range(len(fw_b_range)):

                  for g in range(len(h1_col_range)):

                    for h in range(len(h1_b_range)):

                      for i in range(len(h1_vel_range)):

                        tau_tot_ism = tau_tot_ism_grid_fromfile[g,h,i,:]

                        lya_total = lya_intrinsic_profile_grid_fromfile[a,b,c,d,e,f,:]
    
                        lyman_fit = lya_total * tau_tot_ism
                        if mask.sum() != 0:
                          chisq = np.sum( ( (flux_to_fit_fromfile[~mask] - lyman_fit[~mask]) / 
                                         error_to_fit_fromfile[~mask] )**2 )
                        else:
                          chisq = np.sum ( ( (flux_to_fit_fromfile - lyman_fit) /
                                         error_to_fit_fromfile )**2 )

                        big_chisq_grid[a,b,c,d,e,f,g,h,i] = chisq

    else:
      for a in range(len(vs_n_range)):
        print '*** a = ' + str(a) + ' of ' + str(len(vs_n_range)-1) + ' ***' 
        for b in range(len(am_n_range)):
          print 'b = ' + str(b) + ' of ' + str(len(am_n_range)-1)
          for c in range(len(fw_n_range)):
                  for g in range(len(h1_col_range)):

                    for h in range(len(h1_b_range)):

                      for i in range(len(h1_vel_range)):

                        tau_tot_ism = tau_tot_ism_grid_fromfile[g,h,i,:]

                        lya_total = lya_intrinsic_profile_grid_fromfile[a,b,c,:]
    
                        lyman_fit = lya_total * tau_tot_ism
                        if mask.sum() != 0:
                          chisq = np.sum( ( (flux_to_fit_fromfile[~mask] - lyman_fit[~mask]) / 
                                         error_to_fit_fromfile[~mask] )**2 )
                        else:
                          chisq = np.sum ( ( (flux_to_fit_fromfile - lyman_fit) /
                                         error_to_fit_fromfile )**2 )

                        big_chisq_grid[a,b,c,g,h,i] = chisq


    return big_chisq_grid
  


def multi_run_wrapper(args):

    return damped_lya_profile_gridsearch(*args)


def delta_chi2(df,probability_of_exceeding):

    """
    Computes the Delta-chi2 above the global minimum chi2 where
    the probability of exceeding that chi2 value equals the
    probability_of_exceeding parameter (in decimals, not percentage)
    given the number of parameters (df).

    """

    grid_size = 5000
    x = np.linspace(0,100,grid_size)
    pdf = scipy.stats.chi2.pdf(x, df, loc=0, scale=1)
    dx = x[1]-x[0]
    prob_sum = 0
    i=grid_size
    while prob_sum < probability_of_exceeding:
      i-=1
      prob_sum += pdf[i]*dx
    delta_chi2_statistic = x[i]/df

    return delta_chi2_statistic


def LyA_fit(input_filename,initial_parameters,save_figure=True):

    ## MPFITTING

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


    mask_switch = raw_input("Mask line center (yes or no)? ")
    print ("you entered " + mask_switch)
    if mask_switch == 'yes':
      mask = mask_spectrum(flux_to_fit)
      flux_to_fit_masked = np.ma.masked_where(mask,flux_to_fit)
      error_to_fit_masked = np.ma.masked_where(mask,error_to_fit)
      spec = pyspeckit.Spectrum(xarr=wave_to_fit,data=flux_to_fit_masked*1e14,
                              error=error_to_fit*1e14,doplot=False,header=spec_header)

    else:
      mask = np.array([False])
      spec = pyspeckit.Spectrum(xarr=wave_to_fit,data=flux_to_fit*1e14,
                              error=error_to_fit*1e14,doplot=False,header=spec_header)


    spec.Registry.add_fitter('damped_lya',damped_lya_fitter(),11)


    ## Fitting the profile - Ly-alpha core included ##
    print "Unpacking Initial Guess Parameters"
    
    ## Define Initial Guess Parameters ##
    vs_n = initial_parameters[0]
    am_n = initial_parameters[1]
    fw_n = initial_parameters[2]

    vs_b = initial_parameters[3]
    am_b = initial_parameters[4]
    fw_b = initial_parameters[5]

    h1_col = initial_parameters[6]
    h1_b   = initial_parameters[7]
    h1_vel = initial_parameters[8]
    d2h = initial_parameters[9]                             

    initial_fit = damped_lya_profile(wave_to_fit,vs_n,am_n,fw_n,vs_b,am_b,
                                        fw_b,h1_col,h1_b,h1_vel,d2h,resolution)/1e14



    if mask.sum() != 0:
      chi2 = np.sum( ( (flux_to_fit[~mask] - initial_fit[~mask]) / error_to_fit[~mask] )**2 )
      dof = len(flux_to_fit[~mask]) - len(initial_parameters)
    else:
      chi2 = np.sum( ( (flux_to_fit - initial_fit) / error_to_fit )**2 )
      dof = len(flux_to_fit) - len(initial_parameters)

    reduced_chi2 = chi2/dof

    plt.ion()
    f = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    ax = f.add_subplot(111)
    ax.step(wave_to_fit,flux_to_fit,'k')
    short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
    error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
    short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
    ax.errorbar(short_wave,short_flux,yerr=error_bars_short,fmt=None,
                ecolor='limegreen',elinewidth=3,capthick=3)
    ax.plot(wave_to_fit,initial_fit,'dodgerblue',linewidth=1.5)
    ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
    ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
    ax.set_title(spec_header['STAR'] + ' spectrum with initial parameters',fontsize=16)

    # defining max of y axis 
    y_max = np.max(flux_to_fit)
    y_min = np.min(flux_to_fit)
    ax.set_ylim([y_min,y_max*1.05])
    ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
    plt.ticklabel_format(useOffset=False)

    # Inserting text
    ax.text(0.03,0.97,'vs\_n = '+str(round(vs_n,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.93,'am\_n = '+"{:.2E}".format(am_n),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.89,'fw\_n = '+str(round(fw_n,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.85,'vs\_b = '+str(round(vs_b,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.81,'am\_b = '+"{:.2E}".format(am_b),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.77,'fw\_b = '+str(round(fw_b,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.73,'h1\_col = '+str(round(h1_col,3)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.69,'h1\_b = '+str(round(h1_b,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.65,'h1\_vel = '+str(round(h1_vel,2)),verticalalignment='top', 
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.61,'d2h = '+"{:.2E}".format(d2h),verticalalignment='top',         
        horizontalalignment='left',transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.97,0.97,r'$\chi^{2}_{\nu}$ = '+str(round(reduced_chi2,2)),verticalalignment='top', 
        horizontalalignment='right',transform=ax.transAxes,fontsize=12., color='black')



    ###

    print "Beginning fitting"


    spec.specfit(fittype='damped_lya',guesses=[vs_n,am_n,fw_n,vs_b,am_b,fw_b,
                  h1_col,h1_b,h1_vel,d2h,resolution],quiet=False,fixed=[False,
                  False,False,False,False,False,False,False,False,True,True])

    print spec.specfit.parinfo

    fit_parameters = spec.specfit.parinfo.values
    fit_parameters_errors = spec.specfit.parinfo.errors
    chi2 = spec.specfit.chi2
    dof = spec.specfit.dof
    reduced_chi2 = chi2/dof

    vs_n_final = fit_parameters[0]
    am_n_final = fit_parameters[1]
    fw_n_final = fit_parameters[2]
    vs_b_final = fit_parameters[3]
    am_b_final = fit_parameters[4]
    fw_b_final = fit_parameters[5]
    h1_col_final = fit_parameters[6]
    h1_b_final = fit_parameters[7]
    h1_vel_final = fit_parameters[8]
    d2h_final = fit_parameters[9]
    resolution_final = fit_parameters[10]

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


    model_best_fit = damped_lya_profile(wave_to_fit,vs_n_final,am_n_final,fw_n_final,
                                        vs_b_final,am_b_final,fw_b_final,h1_col_final,
                                        h1_b_final,h1_vel_final,d2h_final,resolution)/1e14


    ## Creating intrinsic Ly-alpha profile and calculation total intrinsic flux##
    lya_profile_intrinsic,lya_flux_intrinsic = lya_intrinsic_profile_func(wave_to_fit,vs_n_final,
                    am_n_final,fw_n_final,vs_b_final,am_b_final,fw_b_final,return_flux=True)


    ### MAKING FINAL LYA FIT PLOT ############


    plt.ion()
    f = plt.figure()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=14)
    ax = f.add_subplot(111)
    ax.step(wave_to_fit,flux_to_fit,'k')
    short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
    error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
    short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
    ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                fmt=None,ecolor='limegreen',elinewidth=3,capthick=3)
    ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)
    ax.plot(wave_to_fit,lya_profile_intrinsic,'b--',linewidth=1.3)
    if mask.sum() != 0:
      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8)
    ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
    ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
    ax.set_title(spec_header['STAR'] + ' spectrum with best fit parameters',fontsize=16)

    # defining max of y axis 
    y_max = np.max( np.array( [np.max(flux_to_fit),np.max(model_best_fit)] ) )
    y_min = np.min(flux_to_fit)
    ax.set_ylim([y_min,y_max*1.05])
    ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
    plt.ticklabel_format(useOffset=False)

    # Inserting text
    ax.text(0.03,0.97,'vs\_n = '+str(round(vs_n_final,2))+' $\pm$ '+str(round(vs_n_err,2)),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,
        fontsize=12., color='black')
    ax.text(0.03,0.93,'am\_n = '+"{:.2E}".format(am_n_final)+' $\pm$  '+
        "{:.2E}".format(am_n_err),verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.89,'fw\_n = '+str(round(fw_n_final,2))+' $\pm$  '+str(round(fw_n_err,2)),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
    ax.text(0.03,0.85,'vs\_b = '+str(round(vs_b_final,2))+' $\pm$  '+str(round(vs_b_err,2)),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
    ax.text(0.03,0.81,'am\_b = '+"{:.2E}".format(am_b_final)+' $\pm$  '+
        "{:.2E}".format(am_b_final),verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.77,'fw\_b = '+str(round(fw_b_final,2))+' $\pm$  '+str(round(fw_b_err,2)),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
    ax.text(0.03,0.73,'h1\_col = '+str(round(h1_col_final,3))+' $\pm$  '+
        str(round(h1_col_err,3)),verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.69,'h1\_b = '+str(round(h1_b_final,2))+' $\pm$  '+str(round(h1_b_err,2)),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12., 
        color='black')
    ax.text(0.03,0.65,'h1\_vel = '+str(round(h1_vel_final,2))+' $\pm$  '+
        str(round(h1_vel_err,2)),verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.03,0.61,'d2h = '+"{:.2E}".format(d2h_final)+' $\pm$  '+"{:.2E}".format(d2h_err),
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')

    ax.text(0.97,0.97,r'Ly$\alpha$ flux = '+"{:.2E}".format(lya_flux_intrinsic),
        verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,fontsize=12.,
        color='black')
    ax.text(0.97,0.93,'[erg/s/cm2]',verticalalignment='top',horizontalalignment='right',
        transform=ax.transAxes,fontsize=12., color='black')
    ax.text(0.97,0.89,r'$\chi^{2}_{\nu}$ = '+str(round(reduced_chi2,2)),verticalalignment='top', 
        horizontalalignment='right',transform=ax.transAxes,fontsize=12., color='black')



    ###########################################
    if save_figure:
      outfile_str = input_filename.replace('.fits','_finalfit.png')
      plt.savefig(outfile_str)
      deltalam = wave_to_fit[1] - wave_to_fit[0]
      wave_to_fit_extended = np.arange(1208.,1222.,deltalam)
      lya_profile_intrinsic_extended = lya_intrinsic_profile_func(wave_to_fit_extended,vs_n_final,
                    am_n_final,fw_n_final,vs_b_final,am_b_final,fw_b_final)
      outfile_str2 = input_filename.replace('.fits','_intrinsic_lya_profile.txt')
      np.savetxt(outfile_str2,np.transpose([wave_to_fit_extended,lya_profile_intrinsic_extended,
                                             0.25*lya_profile_intrinsic_extended]))

    return


def LyA_gridsearch(input_filename,parameter_range,num_cores,
                   brute_force=False,single_component_flux=False):

    ## Read in fits file ##
    spec_hdu = pyfits.open(input_filename)
    spec = spec_hdu[1].data
    spec_header = spec_hdu[1].header

    ## Define wave, flux, error, dq variables ##
    wave_to_fit = spec['wave']
    flux_to_fit = spec['flux']
    error_to_fit = spec['error']
    resolution = float(spec_header['RES_POW'])

    rms_wing = np.std(flux_to_fit[np.where((wave_to_fit > np.min(wave_to_fit)) & 
                                                        (wave_to_fit < 1214.5))])
    error_to_fit[np.where(error_to_fit < rms_wing)] = rms_wing


    mask_switch = raw_input("Mask line center (yes or no)? ")
    print ("you entered " + mask_switch)
    if mask_switch == 'yes':
      mask = mask_spectrum(flux_to_fit)
    else:
      mask = np.array([False])
      

    print "Unpacking Initial Guess Parameters"
    
    ## Define Initial Guess Parameters ##
    vs_n_range = parameter_range[0]
    am_n_range = parameter_range[1]
    fw_n_range = parameter_range[2]

    vs_b_range = parameter_range[3]
    am_b_range = parameter_range[4]
    fw_b_range = parameter_range[5]

    h1_col_range = parameter_range[6]
    h1_b_range   = parameter_range[7]
    h1_vel_range = parameter_range[8]
 


    tau_gridsearch(wave_to_fit,h1_col_range,h1_b_range,h1_vel_range,resolution,mask)

    intrinsic_lya_gridsearch(wave_to_fit,vs_n_range,am_n_range,fw_n_range,vs_b_range,
                                      am_b_range,fw_b_range,resolution,mask,
                                      single_component_flux=single_component_flux)
 

    ##################################


    ## write a text file with wave_to_fit,flux_norm_to_fit,hwave_all,resolution,mask
    np.savetxt('data_to_fit.savepy',np.array([wave_to_fit,flux_to_fit,
               error_to_fit]))
    np.savetxt('resolution.savepy',np.array([resolution,0]))
    np.savetxt('mask.savepy',mask)

    # Degrees of Freedom = (# bins) - 1 - (# parameters) -- needs to be edited for masking
    if mask.sum() != 0: 
      dof = len(flux_to_fit[~mask]) - len(parameter_range)
    else:
      dof = len(flux_to_fit) - len(parameter_range)


    if brute_force:
      big_reduced_chisq_grid = brute_force_gridsearch(vs_n_range,am_n_range,
                               fw_n_range,vs_b_range,am_b_range,fw_b_range,
                               h1_col_range,h1_b_range,h1_vel_range,mask,
                               single_component_flux=single_component_flux)/dof

    else:
      iter_cycle = np.transpose(zip(*itertools.product(range(len(vs_n_range)),
                                range(len(am_n_range)),range(len(fw_n_range)),
                                range(len(vs_b_range)),range(len(am_b_range)),
                                range(len(fw_b_range)),range(len(h1_col_range)),
                                range(len(h1_b_range)),range(len(h1_vel_range)))))

      pool = multiprocessing.Pool(processes=num_cores)
      print "Beginning Multiprocessing"
      chisq_results = pool.map(multi_run_wrapper,iter_cycle)
      print "Finished Multiprocessing"
      pool.close()

      big_reduced_chisq_grid = np.zeros([len(vs_n_range),len(am_n_range),len(fw_n_range),
                               len(vs_b_range),len(am_b_range),len(fw_b_range),
                               len(h1_col_range),len(h1_b_range),len(h1_vel_range)])
      for i in range(len(iter_cycle)):
        big_reduced_chisq_grid[iter_cycle[i][0],iter_cycle[i][1],iter_cycle[i][2],
                     iter_cycle[i][3],iter_cycle[i][4],iter_cycle[i][5],
                     iter_cycle[i][6],iter_cycle[i][7],
                     iter_cycle[i][8]] = chisq_results[i]/dof

    if mask.sum() != 0:
      return big_reduced_chisq_grid, mask

    else:
      return big_reduced_chisq_grid


def extract_error_bars_from_contours(cs):

    """
    Given a contour (cs) from matplotlib, finds the extrema in the
    x and y orthogonal directions.

    """

    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    x = v[:,0]
    y = v[:,1]

    return [np.min(x),np.max(x),np.min(y),np.max(y)]

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





