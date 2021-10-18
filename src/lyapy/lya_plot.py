import numpy as np
import matplotlib.pyplot as plt
from lyapy import lyapy
import corner as triangle
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter, MaxNLocator
import time
from astropy.modeling.models import Voigt1D, Lorentz1D, Gaussian1D
from matplotlib import gridspec


plt.ion()


def walkers(sampler_chain, variables, param_order, burnin, subset=False):
    
    ndim = sampler_chain[0, 0, :].size
    
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, ndim))

    ## this is for long chains, to plot only 1000 evenly sampled points
    if subset:
        toplot = np.array(np.linspace(0,len(sampler_chain[0,:,0])-1,1000), dtype=int)
    else:
        toplot = np.ones_like(sampler_chain[0,:,0], dtype=bool)

    i = 0
    for p in param_order:
        if variables[p]['vary']:
            axes[i].plot(sampler_chain[:, toplot, i].T, color="k", alpha=0.4)
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            axes[i].set_ylabel(variables[p]['texname'])
            ymin = variables[p]['value']-variables[p]['scale']
            ymax = variables[p]['value']+variables[p]['scale']
            axes[i].vlines(burnin,ymin,ymax,color='r')
            i = i + 1

    if subset:
        plt.xlabel("Coursely sampled step number")
    else:
        plt.xlabel("Step number")

    #outfile_str = spec_header['STAR'] + descrip + '_walkers.png'
    #plt.savefig(outfile_str)



def corner(samples, variables, param_order, quantiles=[0.16,0.5,0.84], truths=None, nbins=20,range=None):
      # Make the triangle plot. 

      variable_names = []
      for p in param_order:
        if variables[p]['vary']:
              variable_names.append(variables[p]['texname'])

      ndim = len(variable_names)

      fig, axes = plt.subplots(ndim, ndim, figsize=(12.5,9))
      triangle.corner(samples, bins=nbins, labels=variable_names,
                      max_n_ticks=3,plot_contours=True,quantiles=quantiles,fig=fig,
                      show_titles=True,verbose=True,truths=truths,range=range)


     #outfile_str = spec_header['STAR'] + descrip + '_cornerplot.png'
     #plt.savefig(outfile_str)

# End triangle plot



def profile(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_best_fit, lya_intrinsic_profile_mcmc, variables, param_order, samples = None, 
            perform_error=True, Voigt=False, Lorentzian=False, nbins=100, thin_out = 1.0,
            fix_stellar_ISM_RV_diff=False):


      f = plt.figure(figsize=(8,9))
      plt.rc('text', usetex=True)
      plt.rc('font', family='sans-serif', size=14)
      gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1]) 
      ax = plt.subplot(gs[0])
      axx = plt.subplot(gs[1])
      axxx = plt.subplot(gs[2])


      if samples is not None:
          ndim = len(samples[0])
          #print ndim

          if perform_error:
              
            array_length = int(len(samples)/thin_out)
            model_fits = np.zeros((array_length,len(wave_to_fit)))
            intrinsic_profs = np.zeros((array_length,len(wave_to_fit)))
            SiIII_profs = np.zeros((array_length,len(wave_to_fit)))
            lnprobs = np.zeros(array_length)

            #for i, sample in enumerate(samples[np.random.randint(len(samples), size=10)]):
            i=-1
            for ll, sample in enumerate(samples):
              if (ll % thin_out) == 0:                  
                  i += 1
                  theta_all = []
                  theta = []
                  j = 0
                  for p in param_order:
                      if variables[p]['vary']:
                          theta_all.append(sample[j])
                          theta.append(sample[j])
                          j = j+1
                      else:
                          theta_all.append(variables[p]['value'])

                  if not Voigt:
                    vs_n_i, am_n_i, fw_n_i, vs_b_i, am_b_i, fw_b_i, h1_col_i, h1_b_i, \
                                h1_vel_i, d2h_i, vs_haw_i, am_haw_i, fw_haw_i, \
                                vs_SiIII_i, am_SiIII_i, fw_SiIII_i, ln_f_i = theta_all


                    singcomp = variables['am_b']['single_comp']
                    #model_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_i,10**am_n_i,fw_n_i,
                    #                                    vs_b_i,10**am_b_i,fw_b_i,h1_col_i,
                    #                                    h1_b_i,h1_vel_i,d2h_i,resolution,
                    #                                    single_component_flux=singcomp)/1e14
                    lya_intrinsic_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   vs_n_i,10**am_n_i,fw_n_i,
                                                   vs_b_i,10**am_b_i,fw_b_i,
                                                   single_component_flux=singcomp)

                    if variables['vs_haw']['HAW']:
                      haw_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                          vs_haw_i,
                                          10**am_haw_i,
                                          fw_haw_i,
                                          single_component_flux=True)
                      lya_intrinsic_profile += haw_profile

                    intrinsic_profs[i,:] = lya_intrinsic_profile
                  else:
                      vs_i, am_i, fw_L_i, fw_G_i, h1_col_i, h1_b_i, h1_vel_i, d2h_i, vs_SiIII_i, am_SiIII_i, fw_SiIII_i,ln_f_i = theta_all

                      if fix_stellar_ISM_RV_diff:
                         h1_vel_i = h1_vel_i + vs_i


                      line_center_i = vs_i/3e5*1215.67+1215.67

                      sigma_G_i = fw_G_i/3e5 * 1215.67 #/ 2.3548


                      if variables['fw_G']['fw_G_fw_L_fixed_ratio'] != 0:
                          if variables['fw_G']['fw_G_fw_L_fixed_ratio'] == -1:
                              sigma_L_i = fw_G_i * fw_L_i /3e5 * 1215.67 / 2.
                          else:
                              sigma_L_i = fw_G_i * variables['fw_G']['fw_G_fw_L_fixed_ratio'] /3e5 * 1215.67 / 2.
                      else:
                          sigma_L_i = fw_L_i/3e5 * 1215.67 #/ 2.
                                           

                      if Lorentzian:
                          voigt_profile_func_i = Lorentz1D(x_0 = line_center_i, amplitude = 10**am_i, 
                                                fwhm = sigma_L_i)
                      else:
                          voigt_profile_func_i = Voigt1D(x_0 = line_center_i, amplitude_L = 10**am_i, 
                                                fwhm_L = sigma_L_i, fwhm_G = sigma_G_i)

                      lya_intrinsic_profile = voigt_profile_func_i(wave_to_fit)

                  intrinsic_profs[i,:] = np.convolve(lya_intrinsic_profile,resolution,mode='same')


                  if variables['vs_SiIII']['SiIII']:
                    if variables['vs']['match_v_HI_SiIII']:
                      SiIII_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                          vs_i,
                                          10**am_SiIII_i,
                                          fw_SiIII_i,
                                          single_component_flux=True, line_center=1206.50)
                    else:
                      SiIII_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                          vs_SiIII_i,
                                          10**am_SiIII_i,
                                          fw_SiIII_i,
                                          single_component_flux=True, line_center=1206.50)
                    lya_intrinsic_profile += SiIII_profile

                    SiIII_profs[i,:] = np.convolve(SiIII_profile,resolution,mode='same')

                  total_attenuation = lyapy.total_tau_profile_func(wave_to_fit,
                                           h1_col_i,h1_b_i,h1_vel_i,d2h_i)
                  model_fit = lyapy.damped_lya_profile_shortcut(wave_to_fit,resolution,
                                        lya_intrinsic_profile, total_attenuation)/1e14
                  
                  model_fits[i,:] = model_fit
                  lnprobs[i] = lyapy.lnprob_voigt(theta,wave_to_fit,flux_to_fit,error_to_fit,variables)

            max_lnprob_index = np.where(lnprobs == np.max(lnprobs))
            
                  #plt.plot(wave_to_fit,model_fit,'deeppink',linewidth=1., alpha=0.1)
            sig2_low = np.zeros_like(wave_to_fit)
            sig1_low = np.zeros_like(wave_to_fit)
            median = np.zeros_like(wave_to_fit)
            sig1_high = np.zeros_like(wave_to_fit)
            sig2_high = np.zeros_like(wave_to_fit)

            sig2_low_intr_prof = np.zeros_like(wave_to_fit)
            sig1_low_intr_prof = np.zeros_like(wave_to_fit)
            median_intr_prof = np.zeros_like(wave_to_fit)
            sig1_high_intr_prof = np.zeros_like(wave_to_fit)
            sig2_high_intr_prof = np.zeros_like(wave_to_fit)


            sig2_low_SiIII_prof = np.zeros_like(wave_to_fit)
            sig1_low_SiIII_prof = np.zeros_like(wave_to_fit)
            median_SiIII_prof = np.zeros_like(wave_to_fit)
            sig1_high_SiIII_prof = np.zeros_like(wave_to_fit)
            sig2_high_SiIII_prof = np.zeros_like(wave_to_fit)


            #sig2_low_intr = np.zeros_like(wave_to_fit)
            #sig1_low_intr = np.zeros_like(wave_to_fit)
            #median_intr = np.zeros_like(wave_to_fit)
            #sig1_high_intr = np.zeros_like(wave_to_fit)
            #sig2_high_intr = np.zeros_like(wave_to_fit)

            #sig2_low_SiIII = np.zeros_like(wave_to_fit)
            #sig1_low_SiIII = np.zeros_like(wave_to_fit)
            #median_SiIII = np.zeros_like(wave_to_fit)
            #sig1_high_SiIII = np.zeros_like(wave_to_fit)
            #sig2_high_SiIII = np.zeros_like(wave_to_fit)


            for i in np.arange(len(wave_to_fit)):
                sig2_low[i], sig1_low[i], median[i], sig1_high[i], sig2_high[i] = \
                                       np.percentile(model_fits[:,i], [2.5,15.9,50,84.1,97.5])
                sig2_low_intr_prof[i], sig1_low_intr_prof[i], median_intr_prof[i], sig1_high_intr_prof[i], sig2_high_intr_prof[i] = \
                                       np.percentile(intrinsic_profs[:,i], [2.5,15.9,50,84.1,97.5])

                sig2_low_SiIII_prof[i], sig1_low_SiIII_prof[i], median_SiIII_prof[i], sig1_high_SiIII_prof[i], sig2_high_SiIII_prof[i] = \
                                       np.percentile(SiIII_profs[:,i], [2.5,15.9,50,84.1,97.5])


                #low_intr[i], mid_intr[i], high_intr[i] = np.percentile(intrinsic_profs[:,i], [2.5,50,97.5])
                #low_SiIII[i], mid_SiIII[i], high_SiIII[i] = np.percentile(SiIII_profs[:,i], [2.5, 50, 97.5])
            ax.fill_between(wave_to_fit, sig2_low, sig2_high, color='lightgrey')
            axx.fill_between(wave_to_fit, sig2_low, sig2_high, color='lightgrey')
            ax.fill_between(wave_to_fit, sig1_low, sig1_high, color='grey')
            axx.fill_between(wave_to_fit, sig1_low, sig1_high, color='grey')

            ax.fill_between(wave_to_fit, sig2_low_intr_prof, sig2_high_intr_prof, color='lavenderblush')
            axx.fill_between(wave_to_fit, sig2_low_intr_prof, sig2_high_intr_prof, color='lavenderblush')
            ax.fill_between(wave_to_fit, sig1_low_intr_prof, sig1_high_intr_prof, color='lightpink')
            axx.fill_between(wave_to_fit, sig1_low_intr_prof, sig1_high_intr_prof, color='lightpink')


            reconstructed_flux = np.zeros(len(intrinsic_profs))
            SiIII_flux = np.zeros(len(intrinsic_profs))
            for i,prof in enumerate(intrinsic_profs):
                reconstructed_flux[i] = np.trapz(prof,wave_to_fit)
                SiIII_flux[i] = np.trapz(SiIII_profs[i],wave_to_fit)
                #plt.histogram
            sig2_low_intr, sig1_low_intr, median_intr, sig1_high_intr, sig2_high_intr = \
                                    np.percentile(reconstructed_flux, [2.5,15.9,50,84.1,97.5])
            sig2_low_SiIII, sig1_low_SiIII, median_SiIII, sig1_high_SiIII, sig2_high_SiIII = \
                                    np.percentile(SiIII_flux, [2.5,15.9,50,84.1,97.5])
            #import pdb; pdb.set_trace()
            #for i in range(len(max_lnprob_index[0])):
              #ax.plot(wave_to_fit,model_fits[max_lnprob_index[0][i],:],color='orange')
              #axx.plot(wave_to_fit,model_fits[max_lnprob_index[0][i],:],color='orange')


            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            n,bins,patches=ax2.hist(reconstructed_flux,bins=nbins)
            ax2.vlines(sig2_low_intr,0,1.1*np.max(n),color='k',linestyle='--')
            ax2.vlines(sig1_low_intr,0,1.1*np.max(n),linestyle='--',color='grey')
            #ax2.vlines(mid_flux,0,1.1*np.max(n),color='k')
            ax2.vlines(sig2_high_intr,0,1.1*np.max(n),color='k',linestyle='--')
            ax2.vlines(sig1_high_intr,0,1.1*np.max(n),linestyle='--',color='grey')
            ax2.set_xlabel('Ly$\\alpha$ flux (erg cm$^{-2}$ s$^{-1}$)')
            ax2.minorticks_on()
            best_fit_flux = np.trapz(lya_intrinsic_profile_mcmc,wave_to_fit)
            ax2.vlines(best_fit_flux,0,1.1*np.max(n),color='r')
            #ax2.vlines(np.trapz(intrinsic_profs[max_lnprob_index,:],wave_to_fit),0,1.1*np.max(n),color='orange')
            lya_intrinsic_flux_argument = float(("%e" % best_fit_flux).split('e')[0])
            lya_intrinsic_flux_exponent = float(("%e" % best_fit_flux).split('e')[1])
            ax2.text(0.55,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + \
                  '$^{+' + str(round((sig2_high_intr-best_fit_flux)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}_{-' + str(round((best_fit_flux-sig2_low_intr)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
                   verticalalignment='top',horizontalalignment='left',
                   transform=ax2.transAxes,fontsize=12., color='black')
            ax2.text(0.97,0.93,r'erg cm$^{-2}$ s$^{-1}$',verticalalignment='top',horizontalalignment='right',
                       transform=ax2.transAxes,fontsize=12., color='black')

            f3 = plt.figure()
            ax3 = f3.add_subplot(1,1,1)
            n,bins,patches=ax3.hist(SiIII_flux,bins=nbins)
            ax3.vlines(sig2_low_SiIII,0,1.1*np.max(n),color='k',linestyle='--')
            #ax2.vlines(mid_flux,0,1.1*np.max(n),color='k')
            ax3.vlines(sig2_high_SiIII,0,1.1*np.max(n),color='k',linestyle='--')
            ax3.set_xlabel('Si III flux (erg cm$^{-2}$ s$^{-1}$)')
            ax3.minorticks_on()
            ax3.set_title(str(sig2_low_SiIII) + ', ' + str(sig1_low_SiIII) + ', ' + str(median_SiIII) + ', ' + str(sig1_high_SiIII) + ', ' + str(sig2_high_SiIII))
            #best_fit_flux = np.trapz(lya_intrinsic_profile_mcmc,wave_to_fit)
            #ax2.vlines(best_fit_flux,0,1.1*np.max(n),color='r')
            #lya_intrinsic_flux_argument = float(("%e" % best_fit_flux).split('e')[0])
            #lya_intrinsic_flux_exponent = float(("%e" % best_fit_flux).split('e')[1])
            #ax2.text(0.55,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + \
            #      '$^{+' + str(round((high_flux-best_fit_flux)/10**lya_intrinsic_flux_exponent,2)) + \
            #      '}_{-' + str(round((best_fit_flux-low_flux)/10**lya_intrinsic_flux_exponent,2)) + \
            #      '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
            #       verticalalignment='top',horizontalalignment='left',
            #       transform=ax2.transAxes,fontsize=12., color='black')
            #ax2.text(0.97,0.93,r'erg cm$^{-2}$ s$^{-1}$',verticalalignment='top',horizontalalignment='right',
            #           transform=ax2.transAxes,fontsize=12., color='black')

            

 

      ax.step(wave_to_fit,flux_to_fit,'k',where='mid')
      axx.step(wave_to_fit,flux_to_fit,'k',where='mid')

      mask=[]
      for i in range(len(wave_to_fit)):
        if (i%5 == 0):
          mask.append(i)

      #short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      #error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      #short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      short_wave = wave_to_fit[mask]
      error_bars_short = error_to_fit[mask]
      short_flux = flux_to_fit[mask]
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)

      axx.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      axx.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)


      #plot the intrinsic profile + components

      if not Voigt:
        narrow_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_n']['best'][0],
                                                   10**variables['am_n']['best'][0],
                                                   variables['fw_n']['best'][0],
                                                    single_component_flux=True)
        if not variables['am_b']['single_comp']: 
          broad_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_b']['best'][0],
                                                   10**variables['am_b']['best'][0],
                                                   variables['fw_b']['best'][0],
                                                    single_component_flux=True)
          lya_intrinsic_profile_bestfit = narrow_component + broad_component
        else:
          lya_intrinsic_profile_bestfit = narrow_component
        

        if variables['vs_haw']['HAW']:
          haw_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_haw']['best'][0],
                                                   10**variables['am_haw']['best'][0],
                                                   variables['fw_haw']['best'][0],
                                                    single_component_flux=True)
          lya_intrinsic_profile_bestfit += haw_component

        narrow_component_convolved = np.convolve(narrow_component,resolution,mode='same')
        ax.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)
        axx.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)

        if not variables['am_b']['single_comp']:
          broad_component_convolved = np.convolve(broad_component,resolution,mode='same')
          ax.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)
          axx.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)

        if variables['vs_haw']['HAW']:
          haw_component_convolved = np.convolve(haw_component,resolution,mode='same')
          ax.plot(wave_to_fit,haw_component_convolved,'c:',linewidth=0.7)
          axx.plot(wave_to_fit,haw_component_convolved,'c:',linewidth=0.7)

      else:
   
          line_center_mle = variables['vs']['mle_best']/3e5*1215.67+1215.67
          sigma_G_mle = variables['fw_G']['mle_best']/3e5 * 1215.67 #/ 2.3548

          if variables['fw_G']['fw_G_fw_L_fixed_ratio'] != 0:
              if variables['fw_G']['fw_G_fw_L_fixed_ratio'] == -1:
                  sigma_L_mle = variables['fw_G']['mle_best'] * variables['fw_L']['mle_best'] /3e5 * 1215.67 / 2.
              else:
                  sigma_L_mle = variables['fw_G']['mle_best'] * variables['fw_G']['fw_G_fw_L_fixed_ratio'] /3e5 * 1215.67 / 2.
          else:
              sigma_L_mle = variables['fw_L']['mle_best']/3e5 * 1215.67 #/ 2.

          if Lorentzian:
              voigt_profile_func_mle = Lorentz1D(x_0 = line_center_mle, amplitude = 10**variables['am']['mle_best'], 
                                            fwhm = sigma_L_mle)
          else:
              voigt_profile_func_mle = Voigt1D(x_0 = line_center_mle, amplitude_L = 10**variables['am']['mle_best'], 
                                            fwhm_L = sigma_L_mle, fwhm_G = sigma_G_mle)

          lya_intrinsic_profile_bestfit = voigt_profile_func_mle(wave_to_fit)

      try:
       if variables['vs_SiIII']['SiIII']:
        if variables['vs']['match_v_HI_SiIII']:
          SiIII_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs']['mle_best'],
                                                   10**variables['am_SiIII']['mle_best'],
                                                   variables['fw_SiIII']['mle_best'],
                                                    single_component_flux=True, line_center=1206.50)
        else:
          SiIII_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_SiIII']['mle_best'],
                                                   10**variables['am_SiIII']['mle_best'],
                                                   variables['fw_SiIII']['mle_best'],
                                                    single_component_flux=True, line_center=1206.50)
      except:
          pass

      
      lya_intrinsic_profile_bestfit_convolved = np.convolve(lya_intrinsic_profile_bestfit,resolution,mode='same')
      
      
      

      ax.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      axx.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      
      try:
       if variables['vs_SiIII']['SiIII']:
          SiIII_component_convolved = np.convolve(SiIII_component,resolution,mode='same')
          ax.plot(wave_to_fit,SiIII_component_convolved,'k:',linewidth=0.7)
          axx.plot(wave_to_fit,SiIII_component_convolved,'k:',linewidth=0.7)
      except:
          pass


#
#      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8) ## plotting "masked" region

      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.minorticks_on()
      axx.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      axx.minorticks_on()

      ax.set_ylim([-np.max(flux_to_fit)/10.,np.max(lya_intrinsic_profile_bestfit_convolved)*1.1])
      axx.set_ylim([-np.max(flux_to_fit)/100.,np.max(flux_to_fit)*0.1])

      plt.ticklabel_format(useOffset=False)

      residuals = (flux_to_fit - model_best_fit) / error_to_fit
      chi2 = np.sum( residuals**2 )
      dof = len(flux_to_fit) - ndim - 1

      axxx.plot(wave_to_fit,residuals,'ko')
      axxx.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      axxx.set_ylabel('Residuals',fontsize=14)
      axxx.minorticks_on()
      axxx.hlines(0,np.min(wave_to_fit),np.max(wave_to_fit),linestyle='--',color='grey')
      axxx.hlines(1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')
      axxx.hlines(-1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')

      axxx.text(0.03,0.2,'$\\chi_{\\nu}^2$ = ' + str(round(chi2/dof,2)),verticalalignment='top',
              horizontalalignment='left',transform=axxx.transAxes,fontsize=12., color='black')

      if perform_error:
        LyA_intr_fluxes = np.transpose(np.array([sig2_low_intr, sig1_low_intr, median_intr, sig1_high_intr, sig2_high_intr]))
        LyA_bestfit_models = np.transpose(np.array([sig2_low, sig1_low, median, sig1_high, sig2_high]))
        SiIII_fluxes = np.transpose([sig2_low_SiIII, sig1_low_SiIII, median_SiIII, sig1_high_SiIII, sig2_high_SiIII])
        LyA_intr_profiles = np.transpose(np.array([sig2_low_intr_prof, sig1_low_intr_prof, median_intr_prof, sig1_high_intr_prof, sig2_high_intr_prof]))
        SiIII_profiles = np.transpose(np.array([sig2_low_SiIII_prof, sig1_low_SiIII_prof, median_SiIII_prof, sig1_high_SiIII_prof, sig2_high_SiIII_prof]))
        return LyA_intr_fluxes, LyA_bestfit_models, SiIII_fluxes, LyA_intr_profiles, SiIII_profiles

#      am_n_mcmc_float_str = "{0:.2g}".format(10**am_n_mcmc[0])
#      base, exponent = am_n_mcmc_float_str.split("e")
#      am_n_exponent = float('1e'+exponent)


#      # Inserting text
#      ax.text(0.03,0.97,'V$_n$ = ' + str(round(vs_n_mcmc[0],1)) + '$^{+' + str(round(vs_n_mcmc[1],1)) + '}_{-' + str(round(vs_n_mcmc[2],1)) + '}$',
#        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,
#        fontsize=12., color='black')
#      am_n_p = (10**(am_n_mcmc[0] + am_n_mcmc[1])-10**am_n_mcmc[0])/am_n_exponent
#      am_n_m = (10**am_n_mcmc[0]-10**(am_n_mcmc[0] - am_n_mcmc[2]))/am_n_exponent
#      ax.text(0.03,0.91,'A$_n$ = ('+ str(round(10**am_n_mcmc[0]/am_n_exponent,1)) + '$^{+' + str(round(am_n_p,1)) + '}_{-' + str(round(am_n_m,1)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
#verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#
#
#      ax.text(0.03,0.85,'FW$_n$ = '+ str(round(fw_n_mcmc[0],1)) + '$^{+' + str(round(fw_n_mcmc[1],1)) + '}_{-' + str(round(fw_n_mcmc[2],1)) + '}$',
#        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
#        color='black')
#      ax.text(0.03,0.79,'V$_b$ = '+ str(round(vs_b_mcmc[0],1)) + '$^{+' + str(round(vs_b_mcmc[1],1)) + '}_{-' + str(round(vs_b_mcmc[2],1)) + '}$',
#        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
#        color='black')
#
#      am_b_p = (10**(am_b_mcmc[0] + am_b_mcmc[1])-10**am_b_mcmc[0])/am_n_exponent
#      am_b_m = (10**am_b_mcmc[0]-10**(am_b_mcmc[0] - am_b_mcmc[2]))/am_n_exponent
#      ax.text(0.03,0.73,'A$_b$ = ('+ str(round(10**am_b_mcmc[0]/am_n_exponent,2)) + '$^{+' + str(round(am_b_p,2)) + '}_{-' + str(round(am_b_m,2)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
#verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#
#      ax.text(0.03,0.67,'FW$_b$ = '+ str(round(fw_b_mcmc[0],1)) + '$^{+' + str(round(fw_b_mcmc[1],0)) + '}_{-' + str(round(fw_b_mcmc[2],0)) + '}$',
#        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
#        color='black')
#      ax.text(0.03,0.61,'log N(HI) = '+ str(round(h1_col_mcmc[0],2)) + '$^{+' + str(round(h1_col_mcmc[1],2)) + '}_{-' + str(round(h1_col_mcmc[2],2)) + '}$',
#        verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#      ax.text(0.03,0.55,'b = '+ str(round(h1_b_mcmc[0],1)) + '$^{+' + str(round(h1_b_mcmc[1],1)) + '}_{-' + str(round(h1_b_mcmc[2],1)) + '}$',
#        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12., 
#        color='black')
#      ax.text(0.03,0.49,'V$_{HI}$ = '+ str(round(h1_vel_mcmc[0],1)) + '$^{+' + str(round(h1_vel_mcmc[1],1)) + '}_{-' + str(round(h1_vel_mcmc[2],1)) + '}$',verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#      ax.text(0.03,0.43,r'D/H = 1.5$\times$10$^{-5}$',verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#
#
#      lya_intrinsic_flux_argument = float(("%e" % lya_intrinsic_flux_mcmc).split('e')[0])
#      lya_intrinsic_flux_exponent = float(("%e" % lya_intrinsic_flux_mcmc).split('e')[1])
#      ax.text(0.65,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + '$^{+' + str(round(lya_intrinsic_flux_max_error/10**lya_intrinsic_flux_exponent,2)) + '}_{-' + str(round(lya_intrinsic_flux_min_error/10**lya_intrinsic_flux_exponent,2)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
#        verticalalignment='top',horizontalalignment='left',
#        transform=ax.transAxes,fontsize=12., color='black')
#      ax.text(0.97,0.93,r'erg s$^{-1}$ cm$^{-2}$',verticalalignment='top',horizontalalignment='right',
#        transform=ax.transAxes,fontsize=12., color='black')
#      ax.text(0.97,0.88,r'$\chi^{2}_{\nu}$ = ' + str(round(chi2_mcmc/dof_mcmc,1)),verticalalignment='top',horizontalalignment='right',
#        transform=ax.transAxes,fontsize=12., color='black') 
#
#      outfile_str = spec_header['STAR'] + descrip + '_bestfit.png'
#      plt.savefig(outfile_str)



def profile_cii(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_best_fit, lya_intrinsic_profile_mcmc, variables, param_order, samples = None, 
            perform_error=True, nbins=100):


      f = plt.figure(figsize=(8,9))
      plt.rc('text', usetex=True)
      plt.rc('font', family='sans-serif', size=14)
      gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1]) 
      ax = plt.subplot(gs[0])
      axx = plt.subplot(gs[1])
      axxx = plt.subplot(gs[2])


      if samples is not None:
          ndim = len(samples[0])
          #print ndim

          if perform_error:
              
            model_fits = np.zeros((len(samples),len(wave_to_fit)))
            intrinsic_profs = np.zeros((len(samples),len(wave_to_fit)))

            #for i, sample in enumerate(samples[np.random.randint(len(samples), size=10)]):
            for i, sample in enumerate(samples):
                  theta_all = []
                  j = 0
                  for p in param_order:
                      if variables[p]['vary']:
                          theta_all.append(sample[j])
                          j = j+1
                      else:
                          theta_all.append(variables[p]['value'])

                  vs_i, am_i, fw_i, h1_col_i, h1_b_i, h1_vel_i = theta_all

                  lya_intrinsic_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   vs_i,10**am_i,fw_i,
                                                   single_component_flux=True,line_center=1334.532)
                  other_intrinsic_profile = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   vs_i,2.*(10**am_i),fw_i,
                                                   single_component_flux=True,line_center=1335.708)


                  lya_intrinsic_profile += other_intrinsic_profile
                  intrinsic_profs[i,:] = lya_intrinsic_profile



                  total_attenuation = lyapy.total_tau_profile_func_cii(wave_to_fit,
                                           h1_col_i,h1_b_i,h1_vel_i)
                  model_fit = lyapy.damped_lya_profile_shortcut(wave_to_fit,resolution,
                                        lya_intrinsic_profile, total_attenuation)/1e14
                  
                  model_fits[i,:] = model_fit

                  
                  #plt.plot(wave_to_fit,model_fit,'deeppink',linewidth=1., alpha=0.1)
            low = np.zeros_like(wave_to_fit)
            mid = np.zeros_like(wave_to_fit)
            high = np.zeros_like(wave_to_fit)

            low_intr = np.zeros_like(wave_to_fit)
            mid_intr = np.zeros_like(wave_to_fit)
            high_intr = np.zeros_like(wave_to_fit)

            for i in np.arange(len(wave_to_fit)):
                low[i], mid[i], high[i] = np.percentile(model_fits[:,i], [2.5,50,97.5])
                low_intr[i], mid_intr[i], high_intr[i] = np.percentile(intrinsic_profs[:,i], [2.5,50,97.5])
            ax.fill_between(wave_to_fit, low, high, color='grey')
            axx.fill_between(wave_to_fit, low, high, color='grey')

            reconstructed_flux = np.zeros(len(intrinsic_profs))
            for i,prof in enumerate(intrinsic_profs):
                reconstructed_flux[i] = np.trapz(prof,wave_to_fit)
                #plt.histogram
            low_flux,mid_flux, high_flux = np.percentile(reconstructed_flux, [2.5,50,97.5])
            
            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            n,bins,patches=ax2.hist(reconstructed_flux,bins=nbins)
            ax2.vlines(low_flux,0,1.1*np.max(n),color='k',linestyle='--')
            #ax2.vlines(mid_flux,0,1.1*np.max(n),color='k')
            ax2.vlines(high_flux,0,1.1*np.max(n),color='k',linestyle='--')
            ax2.set_xlabel('Ly$\\alpha$ flux (erg cm$^{-2}$ s$^{-1}$)')
            ax2.minorticks_on()
            best_fit_flux = np.trapz(lya_intrinsic_profile_mcmc,wave_to_fit)
            ax2.vlines(best_fit_flux,0,1.1*np.max(n),color='r')
            lya_intrinsic_flux_argument = float(("%e" % best_fit_flux).split('e')[0])
            lya_intrinsic_flux_exponent = float(("%e" % best_fit_flux).split('e')[1])
            ax2.text(0.55,0.98,r'C II flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + \
                  '$^{+' + str(round((high_flux-best_fit_flux)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}_{-' + str(round((best_fit_flux-low_flux)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
                   verticalalignment='top',horizontalalignment='left',
                   transform=ax2.transAxes,fontsize=12., color='black')
            ax2.text(0.97,0.93,r'erg cm$^{-2}$ s$^{-1}$',verticalalignment='top',horizontalalignment='right',
                       transform=ax2.transAxes,fontsize=12., color='black')

            

 

      ax.step(wave_to_fit,flux_to_fit,'k',where='mid')
      axx.step(wave_to_fit,flux_to_fit,'k',where='mid')

      mask=[]
      for i in range(len(wave_to_fit)):
        if (i%5 == 0):
          mask.append(i)

      #short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      #error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      #short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      short_wave = wave_to_fit[mask]
      error_bars_short = error_to_fit[mask]
      short_flux = flux_to_fit[mask]
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)

      axx.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      axx.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)


      #plot the intrinsic profile + components

      #plot the intrinsic profile + components

      narrow_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs']['best'][0],
                                                   10**variables['am']['best'][0],
                                                   variables['fw']['best'][0],
                                                    single_component_flux=True,line_center=1334.532)

      broad_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs']['best'][0],
                                                   2.*(10**variables['am']['best'][0]),
                                                   variables['fw']['best'][0],
                                                    single_component_flux=True,line_center=1335.708)
      lya_intrinsic_profile_bestfit = narrow_component + broad_component
 
      narrow_component_convolved = np.convolve(narrow_component,resolution,mode='same')
      ax.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)
      axx.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)

      broad_component_convolved = np.convolve(broad_component,resolution,mode='same')
      ax.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)
      axx.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)



      
      lya_intrinsic_profile_bestfit_convolved = np.convolve(lya_intrinsic_profile_bestfit,resolution,mode='same')
      
      
      

      ax.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      axx.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      


#
#      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8) ## plotting "masked" region

      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.minorticks_on()
      axx.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      axx.minorticks_on()

      ax.set_ylim([-np.max(flux_to_fit)/10.,np.max(lya_intrinsic_profile_bestfit_convolved)*1.1])
      axx.set_ylim([-np.max(flux_to_fit)/100.,np.max(flux_to_fit)*0.1])

      plt.ticklabel_format(useOffset=False)

      residuals = (flux_to_fit - model_best_fit) / error_to_fit
      chi2 = np.sum( residuals**2 )
      dof = len(flux_to_fit) - ndim - 1

      axxx.plot(wave_to_fit,residuals,'ko')
      axxx.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      axxx.set_ylabel('Residuals',fontsize=14)
      axxx.minorticks_on()
      axxx.hlines(0,np.min(wave_to_fit),np.max(wave_to_fit),linestyle='--',color='grey')
      axxx.hlines(1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')
      axxx.hlines(-1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')

      axxx.text(0.03,0.2,'$\\chi_{\\nu}^2$ = ' + str(round(chi2/dof,2)),verticalalignment='top',
              horizontalalignment='left',transform=axxx.transAxes,fontsize=12., color='black')

      if perform_error:
        return low_flux, mid_flux, high_flux



######
######

def profile_rev(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_best_fit, lya_intrinsic_profile_mcmc, variables, param_order, samples = None, 
            perform_error=True, Voigt=False, Lorentzian=False, nbins=100, thin_out = 1.0):


      f = plt.figure(figsize=(8,9))
      plt.rc('text', usetex=True)
      plt.rc('font', family='sans-serif', size=14)
      gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1]) 
      ax = plt.subplot(gs[0])
      axx = plt.subplot(gs[1])
      axxx = plt.subplot(gs[2])

      fig_revs = plt.figure()
      ax_rev = fig_revs.add_subplot(111)


      if samples is not None:
          ndim = len(samples[0])
          #print ndim

          if perform_error:
              
            array_length = int(len(samples)/thin_out)
            model_fits = np.zeros((array_length,len(wave_to_fit)))
            intrinsic_profs = np.zeros((array_length,len(wave_to_fit)))
            #SiIII_profs = np.zeros((array_length,len(wave_to_fit)))
            rev_profs = np.zeros((array_length,len(wave_to_fit)))
            lnprobs = np.zeros(array_length)

            #for i, sample in enumerate(samples[np.random.randint(len(samples), size=10)]):
            i=-1
            for ll, sample in enumerate(samples):
              if (ll % thin_out) == 0:                  
                  i += 1
                  theta_all = []
                  theta = []
                  j = 0
                  for p in param_order:
                      if variables[p]['vary']:
                          theta_all.append(sample[j])
                          theta.append(sample[j])
                          j = j+1
                      else:
                          theta_all.append(variables[p]['value'])

                  if True:
                      vs_i, am_i, fw_L_i, fw_G_i, h1_col_i, h1_b_i, h1_vel_i, d2h_i, vs_rev_i, am_rev_i, \
                       fw_rev_i = theta_all

                      line_center_i = vs_i/3e5*1215.67+1215.67

                      sigma_G_i = fw_G_i/3e5 * 1215.67 #/ 2.3548


                      if variables['fw_G']['fw_G_fw_L_fixed_ratio'] != 0:
                          if variables['fw_G']['fw_G_fw_L_fixed_ratio'] == -1:
                              sigma_L_i = fw_G_i * fw_L_i /3e5 * 1215.67 / 2.
                          else:
                              sigma_L_i = fw_G_i * variables['fw_G']['fw_G_fw_L_fixed_ratio'] /3e5 * 1215.67 / 2.
                      else:
                          sigma_L_i = fw_L_i/3e5 * 1215.67 #/ 2.
                                           

                      if Lorentzian:
                          voigt_profile_func_i = Lorentz1D(x_0 = line_center_i, amplitude = 10**am_i, 
                                                fwhm = sigma_L_i)
                      else:
                          voigt_profile_func_i = Voigt1D(x_0 = line_center_i, amplitude_L = 10**am_i, 
                                                fwhm_L = sigma_L_i, fwhm_G = sigma_G_i)

                      lya_intrinsic_profile = voigt_profile_func_i(wave_to_fit)

                  

                  



                  g_func = Gaussian1D(mean = vs_rev_i/3e5*1215.67+1215.67, amplitude=am_rev_i,stddev=fw_rev_i/3e5*1215.67/2.3548)
                  rev_profile = g_func(wave_to_fit)
                  lya_intrinsic_profile *= (rev_profile + 1.)
                  rev_profs[i,:] = rev_profile + 1.

                  intrinsic_profs[i,:] = np.convolve(lya_intrinsic_profile,resolution,mode='same')

                  total_attenuation = lyapy.total_tau_profile_func(wave_to_fit,
                                           h1_col_i,h1_b_i,h1_vel_i,d2h_i)
                  model_fit = lyapy.damped_lya_profile_shortcut(wave_to_fit,resolution,
                                        lya_intrinsic_profile, total_attenuation)/1e14
                  
                  model_fits[i,:] = model_fit
                  #lnprobs[i] = lyapy.lnprob_voigt_rev(theta,wave_to_fit,flux_to_fit,error_to_fit,variables)

            #max_lnprob_index = np.where(lnprobs == np.max(lnprobs))
            
                  #plt.plot(wave_to_fit,model_fit,'deeppink',linewidth=1., alpha=0.1)

                  if i%20 == 0:
                      ax.plot(wave_to_fit,model_fit,color='k',alpha=0.3)
                      ax_rev.plot(wave_to_fit,rev_profile + 1.,color='k',alpha=0.3)
                
            sig2_low = np.zeros_like(wave_to_fit)
            sig1_low = np.zeros_like(wave_to_fit)
            median = np.zeros_like(wave_to_fit)
            sig1_high = np.zeros_like(wave_to_fit)
            sig2_high = np.zeros_like(wave_to_fit)

            sig2_low_intr_prof = np.zeros_like(wave_to_fit)
            sig1_low_intr_prof = np.zeros_like(wave_to_fit)
            median_intr_prof = np.zeros_like(wave_to_fit)
            sig1_high_intr_prof = np.zeros_like(wave_to_fit)
            sig2_high_intr_prof = np.zeros_like(wave_to_fit)



            sig2_low_rev_prof = np.zeros_like(wave_to_fit)
            sig1_low_rev_prof = np.zeros_like(wave_to_fit)
            median_rev_prof = np.zeros_like(wave_to_fit)
            sig1_high_rev_prof = np.zeros_like(wave_to_fit)
            sig2_high_rev_prof = np.zeros_like(wave_to_fit)

            #sig2_low_intr = np.zeros_like(wave_to_fit)
            #sig1_low_intr = np.zeros_like(wave_to_fit)
            #median_intr = np.zeros_like(wave_to_fit)
            #sig1_high_intr = np.zeros_like(wave_to_fit)
            #sig2_high_intr = np.zeros_like(wave_to_fit)

            #sig2_low_SiIII = np.zeros_like(wave_to_fit)
            #sig1_low_SiIII = np.zeros_like(wave_to_fit)
            #median_SiIII = np.zeros_like(wave_to_fit)
            #sig1_high_SiIII = np.zeros_like(wave_to_fit)
            #sig2_high_SiIII = np.zeros_like(wave_to_fit)


            for i in np.arange(len(wave_to_fit)):
                sig2_low[i], sig1_low[i], median[i], sig1_high[i], sig2_high[i] = \
                                       np.percentile(model_fits[:,i], [2.5,15.9,50,84.1,97.5])
                sig2_low_intr_prof[i], sig1_low_intr_prof[i], median_intr_prof[i], sig1_high_intr_prof[i], sig2_high_intr_prof[i] = \
                                       np.percentile(intrinsic_profs[:,i], [2.5,15.9,50,84.1,97.5])

                #sig2_low_SiIII_prof[i], sig1_low_SiIII_prof[i], median_SiIII_prof[i], sig1_high_SiIII_prof[i], sig2_high_SiIII_prof[i] = \
               #                        np.percentile(SiIII_profs[:,i], [2.5,15.9,50,84.1,97.5])
                sig2_low_rev_prof[i], sig1_low_rev_prof[i], median_rev_prof[i], sig1_high_rev_prof[i], sig2_high_rev_prof[i] = \
                                       np.percentile(rev_profs[:,i], [2.5,15.9,50,84.1,97.5])


                #low_intr[i], mid_intr[i], high_intr[i] = np.percentile(intrinsic_profs[:,i], [2.5,50,97.5])
                #low_SiIII[i], mid_SiIII[i], high_SiIII[i] = np.percentile(SiIII_profs[:,i], [2.5, 50, 97.5])
            ax.fill_between(wave_to_fit, sig2_low, sig2_high, color='lightgrey')
            axx.fill_between(wave_to_fit, sig2_low, sig2_high, color='lightgrey')
            ax.fill_between(wave_to_fit, sig1_low, sig1_high, color='grey')
            axx.fill_between(wave_to_fit, sig1_low, sig1_high, color='grey')

            ax.fill_between(wave_to_fit, sig2_low_intr_prof, sig2_high_intr_prof, color='lavenderblush')
            axx.fill_between(wave_to_fit, sig2_low_intr_prof, sig2_high_intr_prof, color='lavenderblush')
            ax.fill_between(wave_to_fit, sig1_low_intr_prof, sig1_high_intr_prof, color='lightpink')
            axx.fill_between(wave_to_fit, sig1_low_intr_prof, sig1_high_intr_prof, color='lightpink')


            reconstructed_flux = np.zeros(len(intrinsic_profs))
            #SiIII_flux = np.zeros(len(intrinsic_profs))
            for i,prof in enumerate(intrinsic_profs):
                reconstructed_flux[i] = np.trapz(prof,wave_to_fit)
                #SiIII_flux[i] = np.trapz(SiIII_profs[i],wave_to_fit)
                #plt.histogram
            sig2_low_intr, sig1_low_intr, median_intr, sig1_high_intr, sig2_high_intr = \
                                    np.percentile(reconstructed_flux, [2.5,15.9,50,84.1,97.5])
            #sig2_low_SiIII, sig1_low_SiIII, median_SiIII, sig1_high_SiIII, sig2_high_SiIII = \
            #                        np.percentile(SiIII_flux, [2.5,15.9,50,84.1,97.5])
            #import pdb; pdb.set_trace()
            #for i in range(len(max_lnprob_index[0])):
              #ax.plot(wave_to_fit,model_fits[max_lnprob_index[0][i],:],color='orange')
              #axx.plot(wave_to_fit,model_fits[max_lnprob_index[0][i],:],color='orange')


            f2 = plt.figure()
            ax2 = f2.add_subplot(1,1,1)
            n,bins,patches=ax2.hist(reconstructed_flux,bins=nbins)
            ax2.vlines(sig2_low_intr,0,1.1*np.max(n),color='k',linestyle='--')
            ax2.vlines(sig1_low_intr,0,1.1*np.max(n),linestyle='--',color='grey')
            #ax2.vlines(mid_flux,0,1.1*np.max(n),color='k')
            ax2.vlines(sig2_high_intr,0,1.1*np.max(n),color='k',linestyle='--')
            ax2.vlines(sig1_high_intr,0,1.1*np.max(n),linestyle='--',color='grey')
            ax2.set_xlabel('Ly$\\alpha$ flux (erg cm$^{-2}$ s$^{-1}$)')
            ax2.minorticks_on()
            best_fit_flux = np.trapz(lya_intrinsic_profile_mcmc,wave_to_fit)
            ax2.vlines(best_fit_flux,0,1.1*np.max(n),color='r')
            #ax2.vlines(np.trapz(intrinsic_profs[max_lnprob_index,:],wave_to_fit),0,1.1*np.max(n),color='orange')
            lya_intrinsic_flux_argument = float(("%e" % best_fit_flux).split('e')[0])
            lya_intrinsic_flux_exponent = float(("%e" % best_fit_flux).split('e')[1])
            ax2.text(0.55,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + \
                  '$^{+' + str(round((sig2_high_intr-best_fit_flux)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}_{-' + str(round((best_fit_flux-sig2_low_intr)/10**lya_intrinsic_flux_exponent,2)) + \
                  '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
                   verticalalignment='top',horizontalalignment='left',
                   transform=ax2.transAxes,fontsize=12., color='black')
            ax2.text(0.97,0.93,r'erg cm$^{-2}$ s$^{-1}$',verticalalignment='top',horizontalalignment='right',
                       transform=ax2.transAxes,fontsize=12., color='black')

            #f3 = plt.figure()
            #ax3 = f3.add_subplot(1,1,1)
            #n,bins,patches=ax3.hist(SiIII_flux,bins=nbins)
            #ax3.vlines(sig2_low_SiIII,0,1.1*np.max(n),color='k',linestyle='--')
            #ax2.vlines(mid_flux,0,1.1*np.max(n),color='k')
            #ax3.vlines(sig2_high_SiIII,0,1.1*np.max(n),color='k',linestyle='--')
            #ax3.set_xlabel('Si III flux (erg cm$^{-2}$ s$^{-1}$)')
            #ax3.minorticks_on()
            #ax3.set_title(str(sig2_low_SiIII) + ', ' + str(sig1_low_SiIII) + ', ' + str(median_SiIII) + ', ' + str(sig1_high_SiIII) + ', ' + str(sig2_high_SiIII))
            #best_fit_flux = np.trapz(lya_intrinsic_profile_mcmc,wave_to_fit)
            #ax2.vlines(best_fit_flux,0,1.1*np.max(n),color='r')
            #lya_intrinsic_flux_argument = float(("%e" % best_fit_flux).split('e')[0])
            #lya_intrinsic_flux_exponent = float(("%e" % best_fit_flux).split('e')[1])
            #ax2.text(0.55,0.98,r'Ly$\alpha$ flux= ('+ str(round(lya_intrinsic_flux_argument,2)) + \
            #      '$^{+' + str(round((high_flux-best_fit_flux)/10**lya_intrinsic_flux_exponent,2)) + \
            #      '}_{-' + str(round((best_fit_flux-low_flux)/10**lya_intrinsic_flux_exponent,2)) + \
            #      '}$) ' + r'$\times$'+ ' 10$^{' + str(int(lya_intrinsic_flux_exponent)) + '}$',
            #       verticalalignment='top',horizontalalignment='left',
            #       transform=ax2.transAxes,fontsize=12., color='black')
            #ax2.text(0.97,0.93,r'erg cm$^{-2}$ s$^{-1}$',verticalalignment='top',horizontalalignment='right',
            #           transform=ax2.transAxes,fontsize=12., color='black')

            

 

      ax.step(wave_to_fit,flux_to_fit,'k',where='mid')
      axx.step(wave_to_fit,flux_to_fit,'k',where='mid')

      mask=[]
      for i in range(len(wave_to_fit)):
        if (i%5 == 0):
          mask.append(i)

      #short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      #error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      #short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      short_wave = wave_to_fit[mask]
      error_bars_short = error_to_fit[mask]
      short_flux = flux_to_fit[mask]
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)

      axx.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      axx.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)


      #plot the intrinsic profile + components

      if not Voigt:
        narrow_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_n']['best'][0],
                                                   10**variables['am_n']['best'][0],
                                                   variables['fw_n']['best'][0],
                                                    single_component_flux=True)
        if not variables['am_b']['single_comp']: 
          broad_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_b']['best'][0],
                                                   10**variables['am_b']['best'][0],
                                                   variables['fw_b']['best'][0],
                                                    single_component_flux=True)
          lya_intrinsic_profile_bestfit = narrow_component + broad_component
        else:
          lya_intrinsic_profile_bestfit = narrow_component
        

        if variables['vs_haw']['HAW']:
          haw_component = lyapy.lya_intrinsic_profile_func(wave_to_fit,
                                                   variables['vs_haw']['best'][0],
                                                   10**variables['am_haw']['best'][0],
                                                   variables['fw_haw']['best'][0],
                                                    single_component_flux=True)
          lya_intrinsic_profile_bestfit += haw_component

        narrow_component_convolved = np.convolve(narrow_component,resolution,mode='same')
        ax.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)
        axx.plot(wave_to_fit,narrow_component_convolved,'r:',linewidth=0.7)

        if not variables['am_b']['single_comp']:
          broad_component_convolved = np.convolve(broad_component,resolution,mode='same')
          ax.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)
          axx.plot(wave_to_fit,broad_component_convolved,'g:',linewidth=0.7)

        if variables['vs_haw']['HAW']:
          haw_component_convolved = np.convolve(haw_component,resolution,mode='same')
          ax.plot(wave_to_fit,haw_component_convolved,'c:',linewidth=0.7)
          axx.plot(wave_to_fit,haw_component_convolved,'c:',linewidth=0.7)

      else:
   
          line_center_mle = variables['vs']['mle_best']/3e5*1215.67+1215.67
          sigma_G_mle = variables['fw_G']['mle_best']/3e5 * 1215.67 #/ 2.3548

          if variables['fw_G']['fw_G_fw_L_fixed_ratio'] != 0:
              if variables['fw_G']['fw_G_fw_L_fixed_ratio'] == -1:
                  sigma_L_mle = variables['fw_G']['mle_best'] * variables['fw_L']['mle_best'] /3e5 * 1215.67 #/ 2.
              else:
                  sigma_L_mle = variables['fw_G']['mle_best'] * variables['fw_G']['fw_G_fw_L_fixed_ratio'] /3e5 * 1215.67 #/ 2.
          else:
              sigma_L_mle = variables['fw_L']['mle_best']/3e5 * 1215.67 #/ 2.

          if Lorentzian:
              voigt_profile_func_mle = Lorentz1D(x_0 = line_center_mle, amplitude = 10**variables['am']['mle_best'], 
                                            fwhm = sigma_L_mle)
          else:
              voigt_profile_func_mle = Voigt1D(x_0 = line_center_mle, amplitude_L = 10**variables['am']['mle_best'], 
                                            fwhm_L = sigma_L_mle, fwhm_G = sigma_G_mle)

          lya_intrinsic_profile_bestfit = voigt_profile_func_mle(wave_to_fit)




      g_func = Gaussian1D(mean = variables['vs_rev']['mle_best']/3e5*1215.67+1215.67, 
              amplitude=variables['am_rev']['mle_best'], stddev=variables['fw_rev']['mle_best']/3e5*1215.67/2.3548)
      rev_profile = g_func(wave_to_fit)
      lya_intrinsic_profile_bestfit *= (rev_profile + 1.)
      
      
      lya_intrinsic_profile_bestfit_convolved = np.convolve(lya_intrinsic_profile_bestfit,resolution,mode='same')
      
      
      

      ax.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      axx.plot(wave_to_fit,lya_intrinsic_profile_bestfit_convolved,'b--',linewidth=1.3)
      

#
#      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8) ## plotting "masked" region

      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.minorticks_on()
      axx.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      axx.minorticks_on()

      ax.set_ylim([-np.max(flux_to_fit)/10.,np.max(lya_intrinsic_profile_bestfit_convolved)*1.1])
      axx.set_ylim([-np.max(flux_to_fit)/100.,np.max(flux_to_fit)*0.1])

      plt.ticklabel_format(useOffset=False)

      residuals = (flux_to_fit - model_best_fit) / error_to_fit
      chi2 = np.sum( residuals**2 )
      dof = len(flux_to_fit) - ndim - 1

      axxx.plot(wave_to_fit,residuals,'ko')
      axxx.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      axxx.set_ylabel('Residuals',fontsize=14)
      axxx.minorticks_on()
      axxx.hlines(0,np.min(wave_to_fit),np.max(wave_to_fit),linestyle='--',color='grey')
      axxx.hlines(1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')
      axxx.hlines(-1,np.min(wave_to_fit),np.max(wave_to_fit),linestyle=':',color='grey')

      axxx.text(0.03,0.2,'$\\chi_{\\nu}^2$ = ' + str(round(chi2/dof,2)),verticalalignment='top',
              horizontalalignment='left',transform=axxx.transAxes,fontsize=12., color='black')

      if perform_error:
        LyA_intr_fluxes = np.transpose(np.array([sig2_low_intr, sig1_low_intr, median_intr, sig1_high_intr, sig2_high_intr]))
        LyA_bestfit_models = np.transpose(np.array([sig2_low, sig1_low, median, sig1_high, sig2_high]))
        #SiIII_fluxes = np.transpose([sig2_low_SiIII, sig1_low_SiIII, median_SiIII, sig1_high_SiIII, sig2_high_SiIII])
        LyA_intr_profiles = np.transpose(np.array([sig2_low_intr_prof, sig1_low_intr_prof, median_intr_prof, sig1_high_intr_prof, sig2_high_intr_prof]))
        #SiIII_profiles = np.transpose(np.array([sig2_low_SiIII_prof, sig1_low_SiIII_prof, median_SiIII_prof, sig1_high_SiIII_prof, sig2_high_SiIII_prof]))
        rev_profiles = np.transpose(np.array([sig2_low_rev_prof, sig1_low_rev_prof, median_rev_prof, sig1_high_rev_prof, sig2_high_rev_prof]))
        return LyA_intr_fluxes, LyA_bestfit_models, LyA_intr_profiles, rev_profiles

