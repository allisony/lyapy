import numpy as np
import pyspeckit
import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import astropy.table
import voigt
import itertools
import multiprocessing

pc_in_cm = 3e18
au_in_cm = 1.5e13
lya_rest = 1215.67
c_km = 2.998e5

lya_rest = 1215.67
ccgs = 3e10
e=4.8032e-10            # electron charge in esu
mp=1.6726231e-24        # proton mass in grams
me=mp/1836.             # electron mass in grams

data_to_fit_fromfile = np.loadtxt('data_to_fit.savepy')
wave_to_fit_fromfile = data_to_fit_fromfile[0,:]
flux_norm_to_fit_fromfile = data_to_fit_fromfile[1,:]
error_norm_to_fit_fromfile = data_to_fit_fromfile[2,:]
resolution_array_fromfile = np.loadtxt('resolution.savepy')
resolution_fromfile = resolution_array_fromfile[0]
mask_fromfile = np.loadtxt('mask.savepy')
tau_tot_ism_grid_fromfile = pyfits.getdata('tau_tot_ism_grid.fits')
lya_intrinsic_profile_grid_fromfile = pyfits.getdata('lya_intrinsic_profile_grid.fits')




def geocoronal_subtraction(input_filename,resolution,starname,sub_geo=False):
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
      EUV_luminosity += (waveband_flux * wavebandwidth[i]) * 4*np.pi*(au_in_cm)**2
   
    ## converting EUV_flux_array to distance from target -- still erg/s/cm^2/Angstrom
    EUV_flux_array *= (au_in_cm/(distance_to_target*pc_in_cm))**2
 
    print ('Total EUV luminosity (100-1170 Angstroms) = ' + str( EUV_luminosity) + ' erg/s')
  
    if doplot:
      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(111)
      ax.plot(EUV_wavelength_array,EUV_flux_array,'Navy',linewidth=1.5)
      ax.set_yscale('log')
      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      ax.set_title(starname + ' EUV spectrum',fontsize=16)
      ax.text(0.97,0.97,r'Ly$\alpha$ flux = '+"{:.2E}".format(total_Lya_flux),
              verticalalignment='top', horizontalalignment='right',
              transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.93,'[erg/s/cm2]',verticalalignment='top', 
              horizontalalignment='right',transform=ax.transAxes,fontsize=12., 
              color='black')
      ax.text(0.97,0.89,'EUV luminosity = '+"{:.2E}".format(EUV_luminosity),
              verticalalignment='top',horizontalalignment='right',
              transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.85,'[erg/s]',verticalalignment='top',horizontalalignment='right',
              transform=ax.transAxes,fontsize=12., color='black')

    if savefile:
      ## Writing EUV flux array to txt file
      dat = np.array([EUV_wavelength_array,EUV_flux_array])
      dat_new = np.transpose(dat)
      comment_str = 'Wavelength in Angstroms and EUV flux in erg/s/cm^2/AA'
      outfile_str = starname + '_EUV_spectrum.txt'
      np.savetxt(outfile_str,dat_new,header=comment_str)

    return EUV_wavelength_array,EUV_flux_array,EUV_luminosity


def gaussian_flux(amplitude,sigma):

    flux = np.sqrt(2*np.pi) * sigma * amplitude

    return flux

def lya_intrinsic_profile_func(x,vs_n,am_n,fw_n,vs_b,am_b,fw_b,return_flux=False):

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
      return lya_profile_narrow+lya_profile_broad,lya_flux_narrow+lya_flux_broad
    else:
      return lya_profile_narrow+lya_profile_broad

def total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h=1.5e-5):

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
                       h1_vel,d2h,resolution,return_components=False,
                       return_hyperfine_components=False):
    
    lya_intrinsic_profile = lya_intrinsic_profile_func(wave_to_fit,vs_n,am_n,fw_n,
                                                              vs_b,am_b,fw_b)
    total_tau_profile = total_tau_profile_func(wave_to_fit,h1_col,h1_b,h1_vel,d2h)

    lya_obs_high = lya_intrinsic_profile * total_tau_profile

    ## Convolving the data ##
    fw = lya_rest/resolution
    aa = make_kernel(grid=wave_to_fit,fwhm=fw)
    lyman_fit = np.convolve(lya_obs_high,aa,mode='same')

    return lyman_fit*1e14

def make_kernel(grid,fwhm):

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
    xsections_symmetrical=np.append(xsections_onesided_flipped[0:nlam-1],xsections_onesided,axis=1) 
    deltalam=np.max(wave_symmetrical)-np.min(wave_symmetrical)
    dellam=wave_symmetrical[1]-wave_symmetrical[0] 
    nall=np.round(deltalam/dellam)
    wave_all=deltalam*(np.arange(nall)/(nall-1))+wave_symmetrical[0]

    tau_all = np.interp(wave_all,wave_symmetrical+lamshifts,xsections_symmetrical*Ntot)

    return wave_all,tau_all

def damped_lya_fitter(multisingle='multi'):
    """
    Generator for Damped LyA fitter class
    """
    myclass =  pyspeckit.models.model.SpectralModel(damped_lya_profile,11,
            parnames=['vs_n','am_n','fw_n','vs_b','am_b','fw_b','h1_col',
                      'h1_b','h1_vel','d2h','resolution'], 
            parlimited=[(False,False),(True,False),(True,False),(False,False),
                        (True,False),(True,False),(True,True),(True,True),
                        (False,False),(True,True),(True,False)], 
            parlimits=[(14.4,14.6), (1e-16,0), (50.,0),(0,0), (1e-16,0),(50.,0),
                       (17.0,19.5), (5.,20.), (0,0),(1.3e-5,1.7e-5),(10000.,0)])
    myclass.__name__ = "damped_lya"
    
    return myclass

def tau_gridsearch(wave_to_fit,h1_col_range,h1_b_range,h1_vel_range,save_file=True):

    tau_tot_ism_test = total_tau_profile_func(wave_to_fit,h1_col_range[0],
                                                  h1_b_range[0],h1_vel_range[0])
    tau_tot_ism_grid = np.zeros([len(h1_col_range),len(h1_b_range),len(h1_vel_range),
                                                              len(tau_tot_ism_test)])
    for a in range(len(h1_col_range)):
      print '*** a = ' + str(a) + ' of ' + str(len(h1_col_range)-1) + ' ***' 
      for b in range(len(h1_b_range)):
        print 'b = ' + str(b) + ' of ' + str(len(h1_b_range)-1)
        for c in range(len(h1_vel_range)):
      
          tot_ism = total_tau_profile_func(wave_to_fit,h1_col_range[a],
                                               h1_b_range[b],h1_vel_range[c]) 
          tau_tot_ism_grid[a,b,c,:] = tot_ism
    
    if save_file:
      hdu = pyfits.PrimaryHDU(data=tau_tot_ism_grid)
      hdu.writeto('tau_tot_ism_grid.fits',clobber=True)

    return

def intrinsic_lya_gridsearch(wave_to_fit,vs_n_range,am_n_range,fw_n_range,vs_b_range,
                             am_b_range,fw_b_range,save_file=True):

    lya_intrinsic_profile_test = lya_intrinsic_profile_func(wave_to_fit,vs_n_range[0],
                                           am_n_range[0],fw_n_range[0],vs_b_range[0],
                                           am_b_range[0],fw_b_range[0])

    lya_intrinsic_profile_grid = np.zeros([len(vs_n_range),len(am_n_range),
                                           len(fw_n_range),len(vs_b_range),len(am_b_range),
                                           len(fw_b_range),len(lya_intrinsic_profile_test)])

    for a in range(len(vs_n_range)):
      print '*** a = ' + str(a) + ' of ' + str(len(vs_n_range)-1) + ' ***' 
      for b in range(len(am_n_range)):
        print 'b = ' + str(b) + ' of ' + str(len(am_n_range)-1)
        for c in range(len(fw_n_range)):

          for d in range(len(vs_b_range)):
   
            for e in range(len(am_b_range)):

              for f in range(len(fw_b_range)):
      
                lya_intrinsic_profile = lya_intrinsic_profile_func(wave_to_fit,
                                            vs_n_range[a],am_n_range[b],fw_n_range[c],
                                            vs_b_range[d],am_b_range[e],fw_b_range[f]) 
                lya_intrinsic_profile_grid[a,b,c,d,e,f,:] = lya_intrinsic_profile
    
    if save_file:
      hdu = pyfits.PrimaryHDU(data=lya_intrinsic_profile_grid)
      hdu.writeto('lya_intrinsic_profile_grid.fits',clobber=True)

    return




def multi_run_wrapper(args):

    return damped_lya_profile_gridsearch(*args)

def damped_lya_profile_gridsearch(a,b,c,d,e,f,g,h,i):
    ## Narrow + Broad Ly-alpha ##
    
    ###################################

    tau_tot_ism = tau_tot_ism_grid_fromfile[g,h,i,:]

    lya_total = lya_intrinsic_profile_grid_fromfile[a,b,c,d,e,f,:]
    
    lya_obs_high = lya_total * tau_tot_ism

    ## Convolving the data to the proper resolution ##

    stis_respow = resolution_fromfile  ## STIS resolution power

    fw = lya_rest/stis_respow

    aa = make_kernel(grid=wave_to_fit_fromfile,fwhm=fw)  

    ## HERE IS THE MODEL TO COMPARE WITH THE DATA ##   
    lyman_fit = np.convolve(lya_obs_high,aa,mode='same')

    if mask_fromfile.sum() != 0:
      chisq = np.sum( ( (flux_norm_to_fit_fromfile[~mask_fromfile] - 
                       lyman_fit[~mask_fromfile]) / error_norm_to_fit_fromfile[~mask_fromfile] )**2 )
    else:
      chisq = np.sum( ( (flux_norm_to_fit_fromfile - lyman_fit) / error_norm_to_fit_fromfile )**2 )

    return chisq



def LyA_fit(input_filename,initial_parameters,save_figure=True):

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


    plt.ion()
    ## MASK LOOP ## 
    mask_switch = raw_input("Mask line center (yes or no)? ")
    print ("you entered " + mask_switch)
    continue_parameter = 'no'
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

    ## END MASK LOOP ##

    spec = pyspeckit.Spectrum(xarr=wave_to_fit,data=flux_to_fit_masked*1e14,
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


    if continue_parameter == 'yes':
      chi2 = np.sum( ( (flux_to_fit[~mask] - initial_fit[~mask]) / error_to_fit[~mask] )**2 )
      dof = len(flux_to_fit[~mask]) - 9.
    else:
      chi2 = np.sum( ( (flux_to_fit - initial_fit) / error_to_fit )**2 )
      dof = len(flux_to_fit) - 9.

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
    ax.set_ylim([y_min,y_max])
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

    from matplotlib import rc
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)

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
    if continue_parameter == 'yes':
      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8)
    ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
    ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
    ax.set_title(spec_header['STAR'] + ' spectrum with best fit parameters',fontsize=16)

    # defining max of y axis 
    y_max = np.max( np.array( [np.max(flux_to_fit),np.max(model_best_fit)] ) )
    y_min = np.min(flux_to_fit)
    ax.set_ylim([y_min,y_max])
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

    return


def LyA_gridsearch(input_filename,parameter_range,num_cores,do_plot=True):

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


    norm_const = np.max(flux_to_fit)
    flux_norm_to_fit = flux_to_fit/norm_const


    plt.ion()
    ## MASK LOOP ## 
    mask_switch = raw_input("Mask line center (yes or no)? ")
    print ("you entered " + mask_switch)
    continue_parameter = 'no'
    mask = np.array([False])
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

    ## END MASK LOOP ##

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


    tau_gridsearch(wave_to_fit,h1_col_range,h1_b_range,h1_vel_range)


    intrinsic_lya_gridsearch(wave_to_fit,vs_n_range,am_n_range,fw_n_range,vs_b_range,
                                                               am_b_range,fw_b_range,)
    ##################################


    ## write a text file with wave_to_fit,flux_norm_to_fit,hwave_all,resolution,mask
    np.savetxt('data_to_fit.savepy',np.array([wave_to_fit,flux_norm_to_fit,error_to_fit/norm_const]))
    np.savetxt('resolution.savepy',np.array([resolution,0]))
    np.savetxt('mask.savepy',mask)

    ## parallelizing!  
    #a,b,c,d,e,f,g,h,i = zip(*itertools.product(range(len(vs_n_range)),range(len(am_n_range)),range(len(fw_n_range)),
    #                                           range(len(vs_b_range)),range(len(am_b_range)),range(len(fw_b_range)),
    #                                           range(len(h1_col_range)),range(len(h1_b_range)),range(len(h1_vel_range))))

    iter_cycle = np.transpose(zip(*itertools.product(range(len(vs_n_range)),range(len(am_n_range)),range(len(fw_n_range)),
                                           range(len(vs_b_range)),range(len(am_b_range)),range(len(fw_b_range)),
                                           range(len(h1_col_range)),range(len(h1_b_range)),range(len(h1_vel_range)))))


    pool = multiprocessing.Pool(processes=num_cores)
    chisq_results = pool.map(multi_run_wrapper,iter_cycle)
    pool.close()

    ##### chisq_results is a list of length num_jobs.  The corresponding 9 parameters for each chi-square found in
    ##### chisq_results are found in the num_jobs x 9 iter_cycle array.


    ###


    # Degrees of Freedom = (# bins) - 1 - (# parameters) -- needs to be edited for masking
    if mask.sum() != 0: 
      dof = len(flux_norm_to_fit[~mask]) - 9
    else:
      dof = len(flux_norm_to_fit) - 9

    reduced_chisq_minimum = np.min(chisq_results)/dof

    ## Best fit parameters
    chisq_min_indices = np.where( chisq_results == np.min(chisq_results) )
    best_fit_indices = iter_cycle[chisq_min_indices[0][0],:]
    vs_n_final = vs_n_range[best_fit_indices[0]]
    am_n_final = am_n_range[best_fit_indices[1]]*norm_const
    fw_n_final = fw_n_range[best_fit_indices[2]]
    vs_b_final = vs_b_range[best_fit_indices[3]]
    am_b_final = am_b_range[best_fit_indices[4]]*norm_const
    fw_b_final = fw_b_range[best_fit_indices[5]]
    h1_col_final = h1_col_range[best_fit_indices[6]]
    h1_b_final = h1_b_range[best_fit_indices[7]]
    h1_vel_final = h1_vel_range[best_fit_indices[8]]

    ## Reconstructing intrinsic LyA flux
    model_best_fit = damped_lya_profile(wave_to_fit,vs_n_final,am_n_final,fw_n_final,
                                          vs_b_final,am_b_final,fw_b_final,h1_col_final,
                                          h1_b_final,h1_vel_final,1.5e-5,resolution)/1e14

    lya_intrinsic_profile,lya_intrinsic_flux = lya_intrinsic_profile_func(wave_to_fit,
         vs_n_final,am_n_final,fw_n_final,vs_b_final,am_b_final,fw_b_final,return_flux=True)
    ##########################################



    ## Print best fit parameters
    print ' '
    print 'BEST FIT PARAMETERS'
    print 'Reduced Chi-square = ' + str(reduced_chisq_minimum)
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



    #### Confidence Intervals
    big_chisq_grid = np.zeros([len(vs_n_range),len(am_n_range),len(fw_n_range),
                               len(vs_b_range),len(am_b_range),len(fw_b_range),
                               len(h1_col_range),len(h1_b_range),len(h1_vel_range)])
    for i in range(len(iter_cycle)):
      big_chisq_grid[iter_cycle[i][0],iter_cycle[i][1],iter_cycle[i][2],iter_cycle[i][3],
                     iter_cycle[i][4],iter_cycle[i][5],iter_cycle[i][6],iter_cycle[i][7],
                     iter_cycle[i][8]] = chisq_results[i]


    if do_plot:
      ### MAKING FINAL LYA FIT PLOT ############

      #from matplotlib import rc
      #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
      #rc('text', usetex=True)

      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(111)
      ax.step(wave_to_fit,flux_norm_to_fit*norm_const,'k')
      short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      short_flux = np.interp(short_wave,wave_to_fit,flux_norm_to_fit*norm_const)
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,fmt=None,ecolor='limegreen',elinewidth=3,capthick=3)
      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)
      ax.plot(wave_to_fit,lya_intrinsic_profile,'b--',linewidth=1.3)
      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      ax.set_title(spec_header['STAR'] + ' spectrum with best fit parameters',fontsize=16)

      # defining max of y axis 
      y_max = np.max( np.array( [np.max(flux_norm_to_fit*norm_const),np.max(model_best_fit)] ) )
      y_min = 0.0
      ax.set_ylim([y_min,y_max])
      ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
      plt.ticklabel_format(useOffset=False)

      vs_n_err,am_n_err,fw_n_err,vs_b_err,am_b_err,fw_b_err,h1_col_err,h1_b_err,h1_vel_err=0,0,0,0,0,0,0,0,0

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
      #ax.text(0.03,0.61,'d2h = '+"{:.2E}".format(d2h_final)+' $\pm$  '+"{:.2E}".format(d2h_err),
  #      verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
  #      color='black')

      ax.text(0.97,0.97,r'Ly$\alpha$ flux = '+"{:.2E}".format(lya_intrinsic_flux),
        verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.97,0.93,'[erg/s/cm2]',verticalalignment='top',horizontalalignment='right',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.89,r'$\chi^{2}_{\nu}$ = '+str(round(reduced_chisq_minimum,2)),verticalalignment='top', 
        horizontalalignment='right',transform=ax.transAxes,fontsize=12., color='black')

    return big_chisq_grid






