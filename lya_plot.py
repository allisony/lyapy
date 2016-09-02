import numpy as np
import matplotlib.pyplot as plt
import lyapy
import corner as triangle
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter, MaxNLocator
import time

plt.ion()

def walkers(sampler):

    # Plot the walkers
    fig, axes = plt.subplots(9, 1, sharex=True, figsize=(8, 9))
    axes[0].plot(sampler.chain[:, :, 0].T, color="k", alpha=0.4)
    axes[0].yaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel("vsn")

    axes[1].plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
    axes[1].yaxis.set_major_locator(MaxNLocator(5))
    axes[1].set_ylabel("amn")

    axes[2].plot(sampler.chain[:, :, 2].T, color="k", alpha=0.4)
    axes[2].yaxis.set_major_locator(MaxNLocator(5))
    axes[2].set_ylabel("fwn")

    axes[3].plot(sampler.chain[:, :, 3].T, color="k", alpha=0.4)
    axes[3].yaxis.set_major_locator(MaxNLocator(5))
    axes[3].set_ylabel("vsb")

    axes[4].plot(sampler.chain[:, :, 4].T, color="k", alpha=0.4)
    axes[4].yaxis.set_major_locator(MaxNLocator(5))
    axes[4].set_ylabel("amb")

    axes[5].plot(sampler.chain[:, :, 5].T, color="k", alpha=0.4)
    axes[5].yaxis.set_major_locator(MaxNLocator(5))
    axes[5].set_ylabel("fwb")

    axes[6].plot(sampler.chain[:, :, 6].T, color="k", alpha=0.4)
    axes[6].yaxis.set_major_locator(MaxNLocator(5))
    axes[6].set_ylabel("h1col")

    axes[7].plot(sampler.chain[:, :, 7].T, color="k", alpha=0.4)
    axes[7].yaxis.set_major_locator(MaxNLocator(5))
    axes[7].set_ylabel("h1b")

    axes[8].plot(sampler.chain[:, :, 8].T, color="k", alpha=0.4)
    axes[8].yaxis.set_major_locator(MaxNLocator(5))
    axes[8].set_ylabel("h1vel")
    axes[8].set_xlabel("step number")

    #outfile_str = spec_header['STAR'] + descrip + '_walkers.png'
    #plt.savefig(outfile_str)

# End walker plot


def corner(samples, variable_names):
      # Make the triangle plot. 
      ndim = len(variable_names)
      
  
      fig, axes = plt.subplots(ndim, ndim, figsize=(12.5,9))
      triangle.corner(samples, bins=20, labels=variable_names,
                      max_n_ticks=3,plot_contours=True,quantiles=[0.16,0.5,0.84],fig=fig,
                      show_titles=True,verbose=True)


     #outfile_str = spec_header['STAR'] + descrip + '_cornerplot.png'
     #plt.savefig(outfile_str)

# End triangle plot



def profile(wave_to_fit, flux_to_fit, error_to_fit, resolution, 
            model_best_fit, lya_intrinsic_profile_mcmc, samples = None,
            d2h_true = 1.5e-5):


      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(1,1,1)

      ## plots 1 sigma contours of the Lya profile
      if samples is not None:
          ndim = len(samples[0])
          print ndim
          if ndim == 9:
              singcomp = False
              vs_n_mcmc, am_n_mcmc, fw_n_mcmc, vs_b_mcmc, am_b_mcmc, fw_b_mcmc, h1_col_mcmc, h1_b_mcmc, \
                                        h1_vel_mcmc  = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                                        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
          else:
              singcomp = True
              vs_n_mcmc, am_n_mcmc, fw_n_mcmc, h1_col_mcmc, h1_b_mcmc, \
                                        h1_vel_mcmc  = map(lambda v: [v[1], v[2]-v[1], v[1]-v[0]], \
                                        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
              vs_b_mcmc, am_b_mcmc, fw_b_mcmc = 0., 0., 0
               
          model_fits = np.zeros((len(samples),len(wave_to_fit)))
          #for i, sample in enumerate(samples[np.random.randint(len(samples), size=10)]):
          for i, sample in enumerate(samples):
              if ndim == 9:
                 vs_n_i, am_n_i, fw_n_i, vs_b_i, am_b_i, fw_b_i, h1_col_i, h1_b_i, \
                                            h1_vel_i = sample
              else:
                 vs_n_i, am_n_i, fw_n_i, h1_col_i, h1_b_i, \
                                            h1_vel_i = sample
                 vs_b_i, am_b_i, fw_b_i = 0., 0., 0.

              model_fit = lyapy.damped_lya_profile(wave_to_fit,vs_n_i,10**am_n_i,fw_n_i,
                                                    vs_b_i,10**am_b_i,fw_b_i,h1_col_i,
                                                    h1_b_i,h1_vel_i,d2h_true,resolution,
                                                    single_component_flux=singcomp)/1e14
              model_fits[i,:] = model_fit
              #plt.plot(wave_to_fit,model_fit,'deeppink',linewidth=1., alpha=0.1)
          low = np.zeros_like(wave_to_fit)
          mid = np.zeros_like(wave_to_fit)
          high = np.zeros_like(wave_to_fit)
          for i in np.arange(len(wave_to_fit)):
              low[i], mid[i], high[i] = np.percentile(model_fits[:,i], [16,50,84])
          plt.fill_between(wave_to_fit, low, high)

      # end 1 sigma contours plotting

      ax.step(wave_to_fit,flux_to_fit,'k')

      short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt="none",ecolor='limegreen',elinewidth=3,capthick=3)

      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)
      ax.plot(wave_to_fit,lya_intrinsic_profile_mcmc,'b--',linewidth=1.3)

#      chi2_mcmc = np.sum( ( (flux_to_fit[~mask] - model_best_fit[~mask]) / error_to_fit[~mask] )**2 )
#      dof_mcmc = len(flux_to_fit[~mask]) - ndim - 1 #####
#
#      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8) ## plotting "masked" region

      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)

      # defining max of y axis 
      y_max = np.max( np.array( [np.max(flux_to_fit),np.max(model_best_fit)] ) )
      y_min = 0.0
      ax.set_ylim([y_min,y_max+0.5e-13])
      ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
      plt.ticklabel_format(useOffset=False)

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

