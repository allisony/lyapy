from pylab import *
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
import matplotlib.cm as cm

am_n_final_float_str = "{0:.2g}".format(am_n_final)
base, exponent = am_n_final_float_str.split("e")
am_n_exponent = float('1e'+exponent)

vs_n_dict = {
         'name': 'vs_n',
         'latex_name': 'v$_n$',
         'range': vs_n_range,
         'increments': MultipleLocator((vs_n_range[-1]-vs_n_range[0])/2.),
         'doplot': True,
         'index': 0,
         'error_minus': 0,
         'error_plus': 0
            }

am_n_dict = {
         'name': 'am_n',
         'latex_name': 'A$_n$',
         'range': am_n_range,
         'increments': MultipleLocator((am_n_range[-1]-am_n_range[0])/2.),
         'doplot': True,
         'index': 1,
         'error_minus': 0,
         'error_plus': 0
            }

fw_n_dict = {
         'name': 'fw_n',
         'latex_name': 'FW$_n$',
         'range': fw_n_range,
         'increments': MultipleLocator((fw_n_range[-1]-fw_n_range[0])/2.),
         'doplot': True,
         'index': 2,
         'error_minus': 0,
         'error_plus': 0
            }


h1_col_dict = {
         'name': 'h1_col',
         'latex_name': '$log$ N(HI)',
         'range': h1_col_range,
         'increments': MultipleLocator((h1_col_range[-1]-h1_col_range[0])/2.),
         'doplot': True,
         'index': 3,
         'error_minus': 0,
         'error_plus': 0
            }

h1_b_dict = {
         'name': 'h1_b',
         'latex_name': 'b',
         'range': h1_b_range,
         'increments': MultipleLocator((h1_b_range[-1]-h1_b_range[0])/2.),
         'doplot': True,
         'index': 4,
         'error_minus': 0,
         'error_plus': 0
            }

h1_vel_dict = {
         'name': 'h1_vel',
         'latex_name': 'v$_{HI}$',
         'range': h1_vel_range,
         'increments': MultipleLocator((h1_vel_range[-1]-h1_vel_range[0])/2.),
         'doplot': True,
         'index': 5,
         'error_minus': 0,
         'error_plus': 0
            }

parameter_dictionaries = [vs_n_dict,am_n_dict,fw_n_dict,
                          h1_col_dict,h1_b_dict,h1_vel_dict]


num_free_parameters = 0

for i in range(len(parameter_dictionaries)):
  num_free_parameters += 1
  if len(parameter_dictionaries[i]['range']) == 1:
    parameter_dictionaries[i]['doplot'] = False
    num_free_parameters -= 1



cmap='Greys_r'


figtext(0.5,0.93,spec_header['STAR'],fontdict={'fontsize':18})




f = figure()

## nested for loop using the dictionaries' index numbers

num_subplots = (num_free_parameters - 1.) ** 2 / 2. + (num_free_parameters - 1.) / 2.

for i in range(num_free_parameters-1):
  print "working on column #" + str(i+1)
  print "i = " + str(i)
  subplot_number = i*(num_free_parameters) + 1.

  for j in range(num_free_parameters - 1 - i):
    grid_indices = [best_fit_indices[0][0],best_fit_indices[1][0],
                    best_fit_indices[2][0],best_fit_indices[3][0],
                    best_fit_indices[4][0],best_fit_indices[5][0]]
                   
    print "subplot #" + str(subplot_number)
    ax = subplot(num_free_parameters-1,num_free_parameters-1,subplot_number)
    subplot_number += (num_free_parameters - 1)
    horiz_axis_parm_dict = parameter_dictionaries[i]
    vert_axis_parm_dict = parameter_dictionaries[i+j+1]
    grid_indices[horiz_axis_parm_dict['index']] = slice(None)
    grid_indices[vert_axis_parm_dict['index']] = slice(None)
    chisq_slice = reduced_chisq_grid[grid_indices]
    chisq_slice_to_plot = np.transpose(chisq_slice)
    extent=[np.min(horiz_axis_parm_dict['range']),np.max(horiz_axis_parm_dict['range']),
            np.min(vert_axis_parm_dict['range']),np.max(vert_axis_parm_dict['range'])]
    ax.imshow(chisq_slice_to_plot,cmap='gray',interpolation='nearest',
              extent=extent,origin='lower',aspect='auto',vmin=global_min,
              vmax=sig3_contour)
    cs = ax.contour(chisq_slice_to_plot,colors=contour_colors,
                    levels=contour_levels,extent=extent)
    ax.plot(horiz_axis_parm_dict['range'][best_fit_indices[i][0]],
            vert_axis_parm_dict['range'][best_fit_indices[j+i+1]],'wx')
    if j == np.max(range(num_free_parameters - 1 - i)):
      ax.set_xlabel(horiz_axis_parm_dict['latex_name'])
      ax.xaxis.set_major_locator( horiz_axis_parm_dict['increments'] )
#      ax.set_xlim(np.min(horiz_axis_parm_dict['range']),
#                  np.max(horiz_axis_parm_dict['range']))
    else:
      ax.xaxis.set_major_formatter( NullFormatter() )


    if i == 0:
      ax.set_ylabel(vert_axis_parm_dict['latex_name'])
      ax.yaxis.set_major_locator( vert_axis_parm_dict['increments'] )
#      ax.set_ylim(np.min(vert_axis_parm_dict['range']),
#                  np.max(vert_axis_parm_dict['range']))
    else:
      ax.yaxis.set_major_formatter( NullFormatter() )

    try:
     err = lyapy.extract_error_bars_from_contours(cs)
     horiz_axis_parm_dict['error_minus'] += err[0]
     horiz_axis_parm_dict['error_plus'] += err[1]
     vert_axis_parm_dict['error_minus'] += err[2]
     vert_axis_parm_dict['error_plus'] += err[3]
    except:
     print "sorry"



subplots_adjust(wspace=0,hspace=0)


for dic in parameter_dictionaries:
  dic['error_minus'] /= -(num_free_parameters - 1.)
  dic['error_minus'] += dic['range'][best_fit_indices[dic['index']][0]]  
  dic['error_plus'] /= (num_free_parameters - 1.)
  dic['error_plus'] -= dic['range'][best_fit_indices[dic['index']][0]]


do_plot=True

if do_plot:
      ### MAKING FINAL LYA FIT PLOT ############


      f = plt.figure()
      plt.rc('text', usetex=True)
      plt.rc('font', family='serif', size=14)
      ax = f.add_subplot(1,1,1)
      ax.step(wave_to_fit,flux_to_fit,'k')
      short_wave = np.linspace(wave_to_fit[0],wave_to_fit[-1],25)
      error_bars_short = np.interp(short_wave,wave_to_fit,error_to_fit)
      short_flux = np.interp(short_wave,wave_to_fit,flux_to_fit)
      ax.errorbar(short_wave,short_flux,yerr=error_bars_short,
                  fmt=None,ecolor='limegreen',elinewidth=3,capthick=3)
      ax.plot(wave_to_fit,model_best_fit,'deeppink',linewidth=1.5)
      ax.plot(wave_to_fit,lya_intrinsic_profile,'b--',linewidth=1.3)
      ax.step(wave_to_fit[mask],flux_to_fit[mask],'lightblue',linewidth=0.8)
      ax.set_ylabel(r'Flux ' r'(erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$)',fontsize=14)
      ax.set_xlabel(r'Wavelength (\AA)',fontsize=14)
      ax.set_title(spec_header['STAR'] + ' spectrum with best fit parameters',fontsize=16)

      # defining max of y axis 
      y_max = np.max( np.array( [np.max(flux_to_fit),np.max(model_best_fit)] ) )
      y_min = 0.0
      ax.set_ylim([y_min,y_max])
      ax.set_xlim( [np.min(wave_to_fit),np.max(wave_to_fit)] )
      plt.ticklabel_format(useOffset=False)
#"$v_{" + str(i) + "}$"

      # Inserting text
      ax.text(0.03,0.97,'V$_n$ = ' + str(round(vs_n_final,2)) + '$^{+' + str(round(vs_n_dict['error_plus'],1)) + '}_{-' + str(round(vs_n_dict['error_minus'],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,
        fontsize=12., color='black')
      ax.text(0.03,0.91,'A$_n$ = ('+ str(round(am_n_final/am_n_exponent,2)) + '$^{+' + str(round(am_n_dict['error_plus']/am_n_exponent,1)) + '}_{-' + str(round(am_n_dict['error_minus']/am_n_exponent,1)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')


      ax.text(0.03,0.85,'FW$_n$ = '+ str(round(fw_n_final,2)) + '$^{+' + str(round(fw_n_dict['error_plus'],0)) + '}_{-' + str(round(fw_n_dict['error_minus'],0)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.03,0.79,'log N(HI) = '+ str(round(h1_col_final,2)) + '$^{+' + str(round(h1_col_dict['error_plus'],2)) + '}_{-' + str(round(h1_col_dict['error_minus'],2)) + '}$',
        verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.03,0.73,'b = '+ str(round(h1_b_final,2)) + '$^{+' + str(round(h1_b_dict['error_plus'],1)) + '}_{-' + str(round(h1_b_dict['error_minus'],1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12., 
        color='black')
      ax.text(0.03,0.67,'V$_{HI}$ = '+ str(round(h1_vel_final,2)) + '$^{+' + str(round(h1_vel_dict['error_plus'],1)) + '}_{-' + str(round(h1_vel_dict['error_minus'],1)) + '}$',verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      #ax.text(0.03,0.61,'d2h = '+"{:.2E}".format(d2h_final)+' $\pm$  '+"{:.2E}".format(d2h_err),
  #      verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
  #      color='black')

      ax.text(0.97,0.97,r'Ly$\alpha$ flux = '+"{:.2E}".format(lya_intrinsic_flux),
        verticalalignment='top',horizontalalignment='right',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.97,0.93,'[erg/s/cm2]',verticalalignment='top',horizontalalignment='right',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.97,0.89,r'$\chi^{2}_{\nu}$ = '+str(round(global_min,2)),verticalalignment='top', 
        horizontalalignment='right',transform=ax.transAxes,fontsize=12., color='black')





