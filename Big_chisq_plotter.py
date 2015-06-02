from pylab import *
from matplotlib.ticker import NullFormatter,MultipleLocator, FormatStrFormatter
import matplotlib.cm as cm

cmap='Greys_r'

vs_n_increments = MultipleLocator(10.)
am_n_increments = MultipleLocator(1.5)
fw_n_increments = MultipleLocator(100.)
vs_b_increments = MultipleLocator(20.)
am_b_increments = MultipleLocator(0.25)
fw_b_increments = MultipleLocator(150.)
h1_col_increments = MultipleLocator(0.5)
h1_b_increments = MultipleLocator(5)
h1_vel_increments = MultipleLocator(5.)

f = figure()

figtext(0.5,0.93,spec_header['STAR'],fontdict={'fontsize':18})

vs_n_marg = np.zeros(len(vs_n_range))
am_n_marg = np.zeros(len(am_n_range))
fw_n_marg = np.zeros(len(fw_n_range))
vs_b_marg = np.zeros(len(vs_b_range))
am_b_marg = np.zeros(len(am_b_range))
fw_b_marg = np.zeros(len(fw_b_range))
h1_col_marg = np.zeros(len(h1_col_range))
h1_b_marg = np.zeros(len(h1_b_range))
h1_vel_marg = np.zeros(len(h1_vel_range))

## min on left, max on right
vs_n_errors = np.zeros((len(parameter_range)-1,2.))
am_n_errors = np.zeros((len(parameter_range)-1,2.))
fw_n_errors = np.zeros((len(parameter_range)-1,2.))
vs_b_errors = np.zeros((len(parameter_range)-1,2.))
am_b_errors = np.zeros((len(parameter_range)-1,2.))
fw_b_errors = np.zeros((len(parameter_range)-1,2.))
h1_col_errors = np.zeros((len(parameter_range)-1,2.))
h1_b_errors = np.zeros((len(parameter_range)-1,2.))
h1_vel_errors = np.zeros((len(parameter_range)-1,2.))

am_n_final_float_str = "{0:.2g}".format(am_n_final)
base, exponent = am_n_final_float_str.split("e")
am_n_exponent = float('1e'+exponent)

## 1
ax1 = subplot(8,8,1)  # 1st column (top)
vs_n_am_n1 = reduced_chisq_grid[:,:,best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_am_n = np.transpose(vs_n_am_n1)
extent1=[np.min(vs_n_range),np.max(vs_n_range),np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent)]
ax1.imshow(vs_n_am_n[0,:,:],cmap='gray',interpolation='nearest',extent=extent1,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs1 = ax1.contour(vs_n_am_n[0,:,:], colors=contour_colors, levels=contour_levels, extent=extent1)#,linewidths=2,extent=extent)
ax1.plot(vs_n_range[best_fit_indices[0]],am_n_range[best_fit_indices[1]]/am_n_exponent,'wx')#,mew=2,ms=5)
ax1.xaxis.set_major_formatter( NullFormatter() )
ax1.set_ylabel('A$_n$')
ax1.yaxis.set_major_locator( am_n_increments )
err1 = lyapy.extract_error_bars_from_contours(cs1)
vs_n_errors[0] = [err1[0],err1[1]]
am_n_errors[0] = [err1[2],err1[3]]


## 2
ax2 = subplot(8,8,9)
vs_n_fw_n1 = reduced_chisq_grid[:,best_fit_indices[1],:,best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_fw_n = np.transpose(vs_n_fw_n1)
extent2=[np.min(vs_n_range),np.max(vs_n_range),np.min(fw_n_range),np.max(fw_n_range)]
ax2.imshow(vs_n_fw_n[:,:,0],cmap='gray',interpolation='nearest',extent=extent2,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs2 = ax2.contour(vs_n_fw_n[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent2)
ax2.plot(vs_n_range[best_fit_indices[0]],fw_n_range[best_fit_indices[2]],'wx')
ax2.xaxis.set_major_formatter( NullFormatter() )
ax2.set_ylabel('FW$_n$')
ax2.yaxis.set_major_locator( fw_n_increments )
err2 = lyapy.extract_error_bars_from_contours(cs2)
vs_n_errors[1] = [err2[0],err2[1]]
fw_n_errors[0] = [err2[2],err2[3]]



## 3
ax3 = subplot(8,8,17)
vs_n_vs_b1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],:,
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_vs_b = np.transpose(vs_n_vs_b1) 
extent3=[np.min(vs_n_range),np.max(vs_n_range),np.min(vs_b_range),np.max(vs_b_range)]
ax3.imshow(vs_n_vs_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent3,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs3 = ax3.contour(vs_n_vs_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent3)
ax3.plot(vs_n_range[best_fit_indices[0]],vs_b_range[best_fit_indices[3]],'wx')
ax3.xaxis.set_major_formatter( NullFormatter() )
ax3.set_ylabel('v$_b$')
ax3.yaxis.set_major_locator( vs_b_increments )
err3 = lyapy.extract_error_bars_from_contours(cs3)
vs_n_errors[2] = [err3[0],err3[1]]
vs_b_errors[0] = [err3[2],err3[3]]


## 4
ax4 = subplot(8,8,25)
vs_n_am_b1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],
                               best_fit_indices[3],:,best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_am_b = np.transpose(vs_n_am_b1)
extent4=[np.min(vs_n_range),np.max(vs_n_range),np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent]
ax4.imshow(vs_n_am_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent4,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs4 = ax4.contour(vs_n_am_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent4)
ax4.plot(vs_n_range[best_fit_indices[0]],am_b_range[best_fit_indices[4]]/am_n_exponent,'wx')
ax4.xaxis.set_major_formatter( NullFormatter() )
ax4.set_ylabel('A$_b$')
ax4.yaxis.set_major_locator( am_b_increments )
err4 = lyapy.extract_error_bars_from_contours(cs4)
vs_n_errors[3] = [err4[0],err4[1]]
am_b_errors[0] = [err4[2],err4[3]]



## 5
ax5 = subplot(8,8,33)
vs_n_fw_b1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],:,
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_fw_b = np.transpose(vs_n_fw_b1)
extent5=[np.min(vs_n_range),np.max(vs_n_range),np.min(fw_b_range),np.max(fw_b_range)]
ax5.imshow(vs_n_fw_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent5,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs5 = ax5.contour(vs_n_fw_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent5)
ax5.plot(vs_n_range[best_fit_indices[0]],fw_b_range[best_fit_indices[5]],'wx')
ax5.xaxis.set_major_formatter( NullFormatter() )
ax5.set_ylabel('FW$_b$')
ax5.yaxis.set_major_locator( fw_b_increments )
err5 = lyapy.extract_error_bars_from_contours(cs5)
vs_n_errors[4] = [err5[0],err5[1]]
fw_b_errors[0] = [err5[2],err5[3]]


## 6
ax6 = subplot(8,8,41)
vs_n_h1_col1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],:,best_fit_indices[7],
                               best_fit_indices[8]]
vs_n_h1_col = np.transpose(vs_n_h1_col1)
extent6=[np.min(vs_n_range),np.max(vs_n_range),np.min(h1_col_range),np.max(h1_col_range)]
ax6.imshow(vs_n_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent6,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs6 = ax6.contour(vs_n_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent6)
ax6.plot(vs_n_range[best_fit_indices[0]],h1_col_range[best_fit_indices[6]],'wx')
ax6.xaxis.set_major_formatter( NullFormatter() )
ax6.set_ylabel('log N(HI)')
ax6.yaxis.set_major_locator( h1_col_increments )
err6 = lyapy.extract_error_bars_from_contours(cs6)
vs_n_errors[5] = [err6[0],err6[1]]
h1_col_errors[0] = [err6[2],err6[3]]


## 7
ax7 = subplot(8,8,49)
vs_n_h1_b1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],:,
                               best_fit_indices[8]]
vs_n_h1_b = np.transpose(vs_n_h1_b1)
extent7=[np.min(vs_n_range),np.max(vs_n_range),np.min(h1_b_range),np.max(h1_b_range)]
ax7.imshow(vs_n_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent7,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs7 = ax7.contour(vs_n_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent7)
ax7.plot(vs_n_range[best_fit_indices[0]],h1_b_range[best_fit_indices[7]],'wx')
ax7.xaxis.set_major_formatter( NullFormatter() )
ax7.set_ylabel('b')
ax7.yaxis.set_major_locator( h1_b_increments )
err7 = lyapy.extract_error_bars_from_contours(cs7)
vs_n_errors[6] = [err7[0],err7[1]]
h1_b_errors[0] = [err7[2],err7[3]]


## 8
ax8 = subplot(8,8,57) # 1st column (bottom)
vs_n_h1_vel1 = reduced_chisq_grid[:,best_fit_indices[1],best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],
                               best_fit_indices[7],:]
vs_n_h1_vel = np.transpose(vs_n_h1_vel1)
extent8=[np.min(vs_n_range),np.max(vs_n_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax8.imshow(vs_n_h1_vel[:,0,:],cmap='gray',interpolation='nearest',extent=extent8,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs8 = ax8.contour(vs_n_h1_vel[:,0,:], colors=contour_colors, levels=contour_levels, extent=extent8)
ax8.plot(vs_n_range[best_fit_indices[0]],h1_vel_range[best_fit_indices[8]],'wx')
ax8.set_xlabel('v$_n$')
ax8.set_ylabel('v$_{HI}$')
ax8.set_xlim(np.min(vs_n_range),np.max(vs_n_range))
ax8.xaxis.set_major_locator( vs_n_increments )
ax8.yaxis.set_major_locator( h1_vel_increments )
err8 = lyapy.extract_error_bars_from_contours(cs8)
vs_n_errors[7] = [err8[0],err8[1]]
h1_vel_errors[0] = [err8[2],err8[3]]




## 9
ax9 = subplot(8,8,10) # 2nd column (top)
am_n_fw_n1 = reduced_chisq_grid[best_fit_indices[0],:,:,best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
am_n_fw_n = np.transpose(am_n_fw_n1)
extent9=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(fw_n_range),np.max(fw_n_range)]
ax9.imshow(am_n_fw_n[:,:,0],cmap='gray',interpolation='nearest',extent=extent9,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs9 = ax9.contour(am_n_fw_n[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent9)
ax9.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,fw_n_range[best_fit_indices[2]],'wx')
ax9.xaxis.set_major_formatter( NullFormatter() )
ax9.yaxis.set_major_formatter( NullFormatter() )
#ax9.yaxis.set_major_locator( MultipleLocator(15) )
err9 = lyapy.extract_error_bars_from_contours(cs9)
am_n_errors[1] = [err9[0],err9[1]]
fw_n_errors[1] = [err9[2],err9[3]]



## 10
ax10 = subplot(8,8,18)
am_n_vs_b1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],:,
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
am_n_vs_b = np.transpose(am_n_vs_b1)
extent10=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(vs_b_range),np.max(vs_b_range)]
ax10.imshow(am_n_vs_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent10,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs10 = ax10.contour(am_n_vs_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent10)
ax10.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,vs_b_range[best_fit_indices[3]],'wx')
ax10.xaxis.set_major_formatter( NullFormatter() )
ax10.yaxis.set_major_formatter( NullFormatter() )
#ax10.yaxis.set_major_locator( MultipleLocator(20.) )
err10 = lyapy.extract_error_bars_from_contours(cs10)
am_n_errors[2] = [err10[0],err10[1]]
vs_b_errors[1] = [err10[2],err10[3]]



## 11
ax11 = subplot(8,8,26)
am_n_am_b1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],
                               best_fit_indices[3],:,best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
am_n_am_b = np.transpose(am_n_am_b1)
extent11=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent]
ax11.imshow(am_n_am_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent11,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs11 = ax11.contour(am_n_am_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent11)
ax11.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,am_b_range[best_fit_indices[4]]/am_n_exponent,'wx')
ax11.xaxis.set_major_formatter( NullFormatter() )
ax11.yaxis.set_major_formatter( NullFormatter() )
#ax11.yaxis.set_major_locator( MultipleLocator(3e-14) )
err11 = lyapy.extract_error_bars_from_contours(cs11)
am_n_errors[3] = [err11[0],err11[1]]
am_b_errors[1] = [err11[2],err11[3]]



## 12
ax12 = subplot(8,8,34)
am_n_fw_b1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],:,
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
am_n_fw_b = np.transpose(am_n_fw_b1)
extent12=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(fw_b_range),np.max(fw_b_range)]
ax12.imshow(am_n_fw_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent12,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs12 = ax12.contour(am_n_fw_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent12)
ax12.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,fw_b_range[best_fit_indices[5]],'wx')
ax12.xaxis.set_major_formatter( NullFormatter() )
ax12.yaxis.set_major_formatter( NullFormatter() )
#ax12.yaxis.set_major_locator( MultipleLocator(50.) )
err12 = lyapy.extract_error_bars_from_contours(cs12)
am_n_errors[4] = [err12[0],err12[1]]
fw_b_errors[1] = [err12[2],err12[3]]



## 13
ax13 = subplot(8,8,42)
am_n_h1_col1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],:,best_fit_indices[7],
                               best_fit_indices[8]]
am_n_h1_col = np.transpose(am_n_h1_col1)
extent13=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(h1_col_range),np.max(h1_col_range)]
ax13.imshow(am_n_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent13,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs13 = ax13.contour(am_n_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent13)
ax13.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,h1_col_range[best_fit_indices[6]],'wx')
ax13.xaxis.set_major_formatter( NullFormatter() )
ax13.yaxis.set_major_formatter( NullFormatter() )
#ax13.yaxis.set_major_locator( MultipleLocator(0.2) )
err13 = lyapy.extract_error_bars_from_contours(cs13)
am_n_errors[5] = [err13[0],err13[1]]
h1_col_errors[1] = [err13[2],err13[3]]


## 14
ax14 = subplot(8,8,50)
am_n_h1_b1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],:,
                               best_fit_indices[8]]
am_n_h1_b = np.transpose(am_n_h1_b1)
extent14=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(h1_b_range),np.max(h1_b_range)]
ax14.imshow(am_n_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent14,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs14 = ax14.contour(am_n_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent14)
ax14.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,h1_b_range[best_fit_indices[7]],'wx')
ax14.xaxis.set_major_formatter( NullFormatter() )
ax14.yaxis.set_major_formatter( NullFormatter() )
#ax14.yaxis.set_major_locator( MultipleLocator(5.) )
err14 = lyapy.extract_error_bars_from_contours(cs14)
am_n_errors[6] = [err14[0],err14[1]]
h1_b_errors[1] = [err14[2],err14[3]]



## 15
ax15 = subplot(8,8,58) # 2nd column (bottom)
am_n_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],:,best_fit_indices[2],
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],
                               best_fit_indices[7],:]
am_n_h1_vel = np.transpose(am_n_h1_vel1)
extent15=[np.min(am_n_range/am_n_exponent),np.max(am_n_range/am_n_exponent),np.min(h1_vel_range),np.max(h1_vel_range)]
ax15.imshow(am_n_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent15,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs15 = ax15.contour(am_n_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent15)
ax15.plot(am_n_range[best_fit_indices[1]]/am_n_exponent,h1_vel_range[best_fit_indices[8]],'wx')
ax15.yaxis.set_major_formatter( NullFormatter() )
ax15.set_xlabel('A$_n$')
ax15.xaxis.set_major_locator( am_n_increments )
#ax15.yaxis.set_major_locator( MultipleLocator(5.) )
err15 = lyapy.extract_error_bars_from_contours(cs15)
am_n_errors[7] = [err15[0],err15[1]]
h1_vel_errors[1] = [err15[2],err15[3]]




## 16
ax16 = subplot(8,8,19) # 3rd column (top)
fw_n_vs_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,:,
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
fw_n_vs_b = np.transpose(fw_n_vs_b1)
extent16=[np.min(fw_n_range),np.max(fw_n_range),np.min(vs_b_range),np.max(vs_b_range)]
ax16.imshow(fw_n_vs_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent16,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs16 = ax16.contour(fw_n_vs_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent16)
ax16.plot(fw_n_range[best_fit_indices[2]],vs_b_range[best_fit_indices[3]],'wx')
ax16.xaxis.set_major_formatter( NullFormatter() )
ax16.yaxis.set_major_formatter( NullFormatter() )
#ax16.yaxis.set_major_locator( MultipleLocator(20.) )
err16 = lyapy.extract_error_bars_from_contours(cs16)
fw_n_errors[2] = [err16[0],err16[1]]
vs_b_errors[2] = [err16[2],err16[3]]



## 17
ax17 = subplot(8,8,27)
fw_n_am_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,
                               best_fit_indices[3],:,best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
fw_n_am_b = np.transpose(fw_n_am_b1)
extent17=[np.min(fw_n_range),np.max(fw_n_range),np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent]
ax17.imshow(fw_n_am_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent17,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs17 = ax17.contour(fw_n_am_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent17)
ax17.plot(fw_n_range[best_fit_indices[2]],am_b_range[best_fit_indices[4]]/am_n_exponent,'wx')
ax17.xaxis.set_major_formatter( NullFormatter() )
ax17.yaxis.set_major_formatter( NullFormatter() )
#ax17.yaxis.set_major_locator( MultipleLocator(3e-14) )
err17 = lyapy.extract_error_bars_from_contours(cs17)
fw_n_errors[3] = [err17[0],err17[1]]
am_b_errors[2] = [err17[2],err17[3]]



## 18
ax18 = subplot(8,8,35)
fw_n_fw_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,
                               best_fit_indices[3],best_fit_indices[4],:,
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
fw_n_fw_b = np.transpose(fw_n_fw_b1)
extent18=[np.min(fw_n_range),np.max(fw_n_range),np.min(fw_b_range),np.max(fw_b_range)]
ax18.imshow(fw_n_fw_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent18,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs18 = ax18.contour(fw_n_fw_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent18)
ax18.plot(fw_n_range[best_fit_indices[2]],fw_b_range[best_fit_indices[5]],'wx')
ax18.xaxis.set_major_formatter( NullFormatter() )
ax18.yaxis.set_major_formatter( NullFormatter() )
#ax18.yaxis.set_major_locator( MultipleLocator(50.) )
err18 = lyapy.extract_error_bars_from_contours(cs18)
fw_n_errors[4] = [err18[0],err18[1]]
fw_b_errors[2] = [err18[2],err18[3]]



## 19
ax19 = subplot(8,8,43)
fw_n_h1_col1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],:,best_fit_indices[7],
                               best_fit_indices[8]]
fw_n_h1_col = np.transpose(fw_n_h1_col1)
extent19=[np.min(fw_n_range),np.max(fw_n_range),np.min(h1_col_range),np.max(h1_col_range)]
ax19.imshow(fw_n_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent19,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs19 = ax19.contour(fw_n_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent19)
ax19.plot(fw_n_range[best_fit_indices[2]],h1_col_range[best_fit_indices[6]],'wx')
ax19.xaxis.set_major_formatter( NullFormatter() )
ax19.yaxis.set_major_formatter( NullFormatter() )
#ax19.yaxis.set_major_locator( MultipleLocator(0.2) )
err19 = lyapy.extract_error_bars_from_contours(cs19)
fw_n_errors[5] = [err19[0],err19[1]]
h1_col_errors[2] = [err19[2],err19[3]]


## 20
ax20 = subplot(8,8,51)
fw_n_h1_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],:,
                               best_fit_indices[8]]
fw_n_h1_b = np.transpose(fw_n_h1_b1)
extent20=[np.min(fw_n_range),np.max(fw_n_range),np.min(h1_b_range),np.max(h1_b_range)]
ax20.imshow(fw_n_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent20,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs20 = ax20.contour(fw_n_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent20)
ax20.plot(fw_n_range[best_fit_indices[2]],h1_b_range[best_fit_indices[7]],'wx')
ax20.xaxis.set_major_formatter( NullFormatter() )
ax20.yaxis.set_major_formatter( NullFormatter() )
#ax20.yaxis.set_major_locator( MultipleLocator(5.) )
err20 = lyapy.extract_error_bars_from_contours(cs20)
fw_n_errors[6] = [err20[0],err20[1]]
h1_b_errors[2] = [err20[2],err20[3]]


## 21
ax21 = subplot(8,8,59) # 3rd column (bottom)
fw_n_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],:,
                               best_fit_indices[3],best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],
                               best_fit_indices[7],:]
fw_n_h1_vel = np.transpose(fw_n_h1_vel1)
extent21=[np.min(fw_n_range),np.max(fw_n_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax21.imshow(fw_n_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent21,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs21 = ax21.contour(fw_n_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent21)
ax21.plot(fw_n_range[best_fit_indices[2]],h1_vel_range[best_fit_indices[8]],'wx')
ax21.yaxis.set_major_formatter( NullFormatter() )
ax21.set_xlabel('FW$_n$')
ax21.xaxis.set_major_locator( fw_n_increments )
#ax21.yaxis.set_major_locator( MultipleLocator(5.) )
err21 = lyapy.extract_error_bars_from_contours(cs21)
fw_n_errors[7] = [err21[0],err21[1]]
h1_vel_errors[2] = [err21[2],err21[3]]




## 22
ax22 = subplot(8,8,28) # 4th column (top)
vs_b_am_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],:,:,best_fit_indices[5],
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_b_am_b = np.transpose(vs_b_am_b1)
extent22=[np.min(vs_b_range),np.max(vs_b_range),np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent]
ax22.imshow(vs_b_am_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent22,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs22 = ax22.contour(vs_b_am_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent22)
ax22.plot(vs_b_range[best_fit_indices[3]],am_b_range[best_fit_indices[4]]/am_n_exponent,'wx')
ax22.xaxis.set_major_formatter( NullFormatter() )
ax22.yaxis.set_major_formatter( NullFormatter() )
#ax22.xaxis.set_major_locator( MultipleLocator(20.) )
#ax22.yaxis.set_major_locator( MultipleLocator(3e-14) )
err22 = lyapy.extract_error_bars_from_contours(cs22)
vs_b_errors[3] = [err22[0],err22[1]]
am_b_errors[3] = [err22[2],err22[3]]


## 23
ax23 = subplot(8,8,36)
vs_b_fw_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],:,best_fit_indices[4],:,
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
vs_b_fw_b = np.transpose(vs_b_fw_b1)
extent23=[np.min(vs_b_range),np.max(vs_b_range),np.min(fw_b_range),np.max(fw_b_range)]
ax23.imshow(vs_b_fw_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent23,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs23 = ax23.contour(vs_b_fw_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent23)
ax23.plot(vs_b_range[best_fit_indices[3]],fw_b_range[best_fit_indices[5]],'wx')
ax23.xaxis.set_major_formatter( NullFormatter() )
ax23.yaxis.set_major_formatter( NullFormatter() )
#ax23.xaxis.set_major_locator( MultipleLocator(20.) )
#ax23.yaxis.set_major_locator( MultipleLocator(50.) )
err23 = lyapy.extract_error_bars_from_contours(cs23)
vs_b_errors[4] = [err23[0],err23[1]]
fw_b_errors[3] = [err23[2],err23[3]]



## 24
ax24 = subplot(8,8,44)
vs_b_h1_col1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],:,best_fit_indices[4],
                               best_fit_indices[5],:,best_fit_indices[7],
                               best_fit_indices[8]]
vs_b_h1_col = np.transpose(vs_b_h1_col1)
extent24=[np.min(vs_b_range),np.max(vs_b_range),np.min(h1_col_range),np.max(h1_col_range)]
ax24.imshow(vs_b_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent24,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs24 = ax24.contour(vs_b_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent24)
ax24.plot(vs_b_range[best_fit_indices[3]],h1_col_range[best_fit_indices[6]],'wx')
ax24.xaxis.set_major_formatter( NullFormatter() )
ax24.yaxis.set_major_formatter( NullFormatter() )
#ax24.xaxis.set_major_locator( MultipleLocator(20.) )
#ax24.yaxis.set_major_locator( MultipleLocator(0.2) )
err24 = lyapy.extract_error_bars_from_contours(cs24)
vs_b_errors[5] = [err24[0],err24[1]]
h1_col_errors[3] = [err24[2],err24[3]]


## 25
ax25 = subplot(8,8,52)
vs_b_h1_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],:,best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],:,
                               best_fit_indices[8]]
vs_b_h1_b = np.transpose(vs_b_h1_b1)
extent25=[np.min(vs_b_range),np.max(vs_b_range),np.min(h1_b_range),np.max(h1_b_range)]
ax25.imshow(vs_b_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent25,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs25 = ax25.contour(vs_b_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent25)
ax25.plot(vs_b_range[best_fit_indices[3]],h1_b_range[best_fit_indices[7]],'wx')
ax25.xaxis.set_major_formatter( NullFormatter() )
ax25.yaxis.set_major_formatter( NullFormatter() )
#ax25.xaxis.set_major_locator( MultipleLocator(20.) )
#ax25.yaxis.set_major_locator( MultipleLocator(5.) )
err25 = lyapy.extract_error_bars_from_contours(cs25)
vs_b_errors[6] = [err25[0],err25[1]]
h1_b_errors[3] = [err25[2],err25[3]]



## 26
ax26 = subplot(8,8,60) # 4th column (bottom)
vs_b_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],:,best_fit_indices[4],
                               best_fit_indices[5],best_fit_indices[6],
                               best_fit_indices[7],:]
vs_b_h1_vel = np.transpose(vs_b_h1_vel1)
extent26=[np.min(vs_b_range),np.max(vs_b_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax26.imshow(vs_b_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent26,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs26 = ax26.contour(vs_b_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent26)
ax26.plot(vs_b_range[best_fit_indices[3]],h1_vel_range[best_fit_indices[8]],'wx')
ax26.yaxis.set_major_formatter( NullFormatter() )
ax26.set_xlabel('v$_b$')
ax26.xaxis.set_major_locator( vs_b_increments )
#ax26.yaxis.set_major_locator( MultipleLocator(5.) )
err26 = lyapy.extract_error_bars_from_contours(cs26)
vs_b_errors[7] = [err26[0],err26[1]]
h1_vel_errors[3] = [err26[2],err26[3]]



## 27
ax27 = subplot(8,8,37) # 5th column (top)
am_b_fw_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],:,:,
                               best_fit_indices[6],best_fit_indices[7],
                               best_fit_indices[8]]
am_b_fw_b = np.transpose(am_b_fw_b1)
extent27=[np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent,np.min(fw_b_range),np.max(fw_b_range)]
ax27.imshow(am_b_fw_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent27,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs27 = ax27.contour(am_b_fw_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent27)
ax27.plot(am_b_range[best_fit_indices[4]]/am_n_exponent,fw_b_range[best_fit_indices[5]],'wx')
ax27.xaxis.set_major_formatter( NullFormatter() )
ax27.yaxis.set_major_formatter( NullFormatter() )
#ax27.xaxis.set_major_locator( MultipleLocator(3e14) )
#ax27.yaxis.set_major_locator( MultipleLocator(50.) )
err27 = lyapy.extract_error_bars_from_contours(cs27)
am_b_errors[4] = [err27[0],err27[1]]
fw_b_errors[4] = [err27[2],err27[3]]



## 28
ax28 = subplot(8,8,45)
am_b_h1_col1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],:,
                               best_fit_indices[5],:,best_fit_indices[7],
                               best_fit_indices[8]]
am_b_h1_col = np.transpose(am_b_h1_col1)
extent28=[np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent,np.min(h1_col_range),np.max(h1_col_range)]
ax28.imshow(am_b_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent28,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs28 = ax28.contour(am_b_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent28)
ax28.plot(am_b_range[best_fit_indices[4]]/am_n_exponent,h1_col_range[best_fit_indices[6]],'wx')
ax28.xaxis.set_major_formatter( NullFormatter() )
ax28.yaxis.set_major_formatter( NullFormatter() )
#ax28.xaxis.set_major_locator( MultipleLocator(3e14) )
#ax28.yaxis.set_major_locator( MultipleLocator(0.2) )
err28 = lyapy.extract_error_bars_from_contours(cs28)
am_b_errors[5] = [err28[0],err28[1]]
h1_col_errors[4] = [err28[2],err28[3]]




## 29
ax29 = subplot(8,8,53)
am_b_h1_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],:,
                               best_fit_indices[5],best_fit_indices[6],:,
                               best_fit_indices[8]]
am_b_h1_b = np.transpose(am_b_h1_b1)
extent29=[np.min(am_b_range)/am_n_exponent,np.max(am_b_range)/am_n_exponent,np.min(h1_b_range),np.max(h1_b_range)]
ax29.imshow(am_b_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent29,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs29 = ax29.contour(am_b_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent29)
ax29.plot(am_b_range[best_fit_indices[4]]/am_n_exponent,h1_b_range[best_fit_indices[7]],'wx')
ax29.xaxis.set_major_formatter( NullFormatter() )
ax29.yaxis.set_major_formatter( NullFormatter() )
#ax29.xaxis.set_major_locator( MultipleLocator(3e14) )
#ax29.yaxis.set_major_locator( MultipleLocator(5.) )
err29 = lyapy.extract_error_bars_from_contours(cs29)
am_b_errors[6] = [err29[0],err29[1]]
h1_b_errors[4] = [err29[2],err29[3]]



## 30
ax30 = subplot(8,8,61) # 5th column (bottom)
am_b_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],:,
                               best_fit_indices[5],best_fit_indices[6],
                               best_fit_indices[7],:]
am_b_h1_vel = np.transpose(am_b_h1_vel1)
extent30=[np.min(am_b_range/am_n_exponent),np.max(am_b_range/am_n_exponent),np.min(h1_vel_range),np.max(h1_vel_range)]
ax30.imshow(am_b_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent30,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs30 = ax30.contour(am_b_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent30)
ax30.plot(am_b_range[best_fit_indices[4]]/am_n_exponent,h1_vel_range[best_fit_indices[8]],'wx')
ax30.yaxis.set_major_formatter( NullFormatter() )
ax30.set_xlabel('A$_b$')
ax30.xaxis.set_major_locator( am_b_increments )
#ax30.yaxis.set_major_locator( MultipleLocator(5.) )
err30 = lyapy.extract_error_bars_from_contours(cs30)
am_b_errors[7] = [err30[0],err30[1]]
h1_vel_errors[4] = [err30[2],err30[3]]




## 31
ax31 = subplot(8,8,46) # 6th column (top)
fw_b_h1_col1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],:,:,best_fit_indices[7],
                               best_fit_indices[8]]
fw_b_h1_col = np.transpose(fw_b_h1_col1)
extent31=[np.min(fw_b_range),np.max(fw_b_range),np.min(h1_col_range),np.max(h1_col_range)]
ax31.imshow(fw_b_h1_col[:,:,0],cmap='gray',interpolation='nearest',extent=extent31,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs31 = ax31.contour(fw_b_h1_col[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent31)
ax31.plot(fw_b_range[best_fit_indices[5]],h1_col_range[best_fit_indices[6]],'wx')
ax31.xaxis.set_major_formatter( NullFormatter() )
ax31.yaxis.set_major_formatter( NullFormatter() )
#ax31.xaxis.set_major_locator( MultipleLocator(50) )
#ax31.yaxis.set_major_locator( MultipleLocator(0.2) )
err31 = lyapy.extract_error_bars_from_contours(cs31)
fw_b_errors[5] = [err31[0],err31[1]]
h1_col_errors[5] = [err31[2],err31[3]]





## 32
ax32 = subplot(8,8,54)
fw_b_h1_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],:,best_fit_indices[6],:,
                               best_fit_indices[8]]
fw_b_h1_b = np.transpose(fw_b_h1_b1)
extent32=[np.min(fw_b_range),np.max(fw_b_range),np.min(h1_b_range),np.max(h1_b_range)]
ax32.imshow(fw_b_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent32,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs32 = ax32.contour(fw_b_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent32)
ax32.plot(fw_b_range[best_fit_indices[5]],h1_b_range[best_fit_indices[7]],'wx')
ax32.xaxis.set_major_formatter( NullFormatter() )
ax32.yaxis.set_major_formatter( NullFormatter() )
#ax32.xaxis.set_major_locator( MultipleLocator(50) )
#ax32.yaxis.set_major_locator( MultipleLocator(5.) )
err32 = lyapy.extract_error_bars_from_contours(cs32)
fw_b_errors[6] = [err32[0],err32[1]]
h1_b_errors[5] = [err32[2],err32[3]]




## 33
ax33 = subplot(8,8,62) # 6th column (bottom)
fw_b_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],:,best_fit_indices[6],
                               best_fit_indices[7],:]
fw_b_h1_vel = np.transpose(fw_b_h1_vel1)
extent33=[np.min(fw_b_range),np.max(fw_b_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax33.imshow(fw_b_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent33,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs33 = ax33.contour(fw_b_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent33)
ax33.plot(fw_b_range[best_fit_indices[5]],h1_vel_range[best_fit_indices[8]],'wx')
ax33.yaxis.set_major_formatter( NullFormatter() )
ax33.set_xlabel('FW$_b$')
ax33.xaxis.set_major_locator( fw_b_increments )
#ax33.yaxis.set_major_locator( MultipleLocator(5.) )
err33 = lyapy.extract_error_bars_from_contours(cs33)
fw_b_errors[7] = [err33[0],err33[1]]
h1_vel_errors[5] = [err33[2],err33[3]]






## 34
ax34 = subplot(8,8,55) # 7th column (top)
h1_col_h1_b1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],:,:,
                               best_fit_indices[8]]
h1_col_h1_b = np.transpose(h1_col_h1_b1)
extent34=[np.min(h1_col_range),np.max(h1_col_range),np.min(h1_b_range),np.max(h1_b_range)]
ax34.imshow(h1_col_h1_b[:,:,0],cmap='gray',interpolation='nearest',extent=extent34,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs34 = ax34.contour(h1_col_h1_b[:,:,0], colors=contour_colors, levels=contour_levels, extent=extent34)
ax34.plot(h1_col_range[best_fit_indices[6]],h1_b_range[best_fit_indices[7]],'wx')
ax34.xaxis.set_major_formatter( NullFormatter() )
ax34.yaxis.set_major_formatter( NullFormatter() )
#ax34.xaxis.set_major_locator( MultipleLocator(0.2) )
#ax34.yaxis.set_major_locator( MultipleLocator(5.) )
err34 = lyapy.extract_error_bars_from_contours(cs34)
h1_col_errors[6] = [err34[0],err34[1]]
h1_b_errors[6] = [err34[2],err34[3]]




## 35
ax35 = subplot(8,8,63) # 7th column (bottom)
h1_col_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],:,
                               best_fit_indices[7],:]
h1_col_h1_vel = np.transpose(h1_col_h1_vel1)
extent35=[np.min(h1_col_range),np.max(h1_col_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax35.imshow(h1_col_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent35,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs35 = ax35.contour(h1_col_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels,extent=extent35)
ax35.plot(h1_col_range[best_fit_indices[6]],h1_vel_range[best_fit_indices[8]],'wx')
ax35.yaxis.set_major_formatter( NullFormatter() )
ax35.set_xlabel('$log$ N(HI)')
ax35.xaxis.set_major_locator( h1_col_increments )
#ax35.yaxis.set_major_locator( MultipleLocator(5.) )
err35 = lyapy.extract_error_bars_from_contours(cs35)
h1_col_errors[7] = [err35[0],err35[1]]
h1_vel_errors[6] = [err35[2],err35[3]]




## 36
ax36 = subplot(8,8,64) # 8th column
h1_b_h1_vel1 = reduced_chisq_grid[best_fit_indices[0],best_fit_indices[1],
                               best_fit_indices[2],best_fit_indices[3],
                               best_fit_indices[4],best_fit_indices[5],
                               best_fit_indices[6],:,:]
h1_b_h1_vel = np.transpose(h1_b_h1_vel1)
extent36=[np.min(h1_b_range),np.max(h1_b_range),np.min(h1_vel_range),np.max(h1_vel_range)]
ax36.imshow(h1_b_h1_vel[:,:,0],cmap='gray',interpolation='nearest',extent=extent36,origin='lower',aspect='auto',vmin=global_min,vmax=sig3_contour)
cs36 = ax36.contour(h1_b_h1_vel[:,:,0], colors=contour_colors, levels=contour_levels,extent=extent36)
ax36.plot(h1_b_range[best_fit_indices[7]],h1_vel_range[best_fit_indices[8]],'wx')
ax36.yaxis.set_major_formatter( NullFormatter() )
ax36.set_xlabel('b')
ax36.xaxis.set_major_locator( h1_b_increments )
#ax36.yaxis.set_major_locator( MultipleLocator(5.) )
err36 = lyapy.extract_error_bars_from_contours(cs36)
h1_b_errors[7] = [err36[0],err36[1]]
h1_vel_errors[7] = [err36[2],err36[3]]



subplots_adjust(wspace=0,hspace=0)


vs_n_minus = vs_n_final - vs_n_errors[:,0].mean()
vs_n_plus = vs_n_errors[:,1].mean() - vs_n_final

am_n_minus = am_n_final - (am_n_errors[:,0].mean())*am_n_exponent
am_n_plus = (am_n_errors[:,1].mean())*am_n_exponent - am_n_final

fw_n_minus = fw_n_final - fw_n_errors[:,0].mean()
fw_n_plus = fw_n_errors[:,1].mean() - fw_n_final

vs_b_minus = vs_b_final - vs_b_errors[:,0].mean()
vs_b_plus = vs_b_errors[:,1].mean() - vs_b_final

am_b_minus = am_b_final - (am_b_errors[:,0].mean())*am_n_exponent
am_b_plus = (am_b_errors[:,1].mean())*am_n_exponent - am_b_final

fw_b_minus = fw_b_final - fw_b_errors[:,0].mean()
fw_b_plus = fw_b_errors[:,1].mean() - fw_b_final

h1_col_minus = h1_col_final - h1_col_errors[:,0].mean()
h1_col_plus = h1_col_errors[:,1].mean() - h1_col_final

h1_b_minus = h1_b_final - h1_b_errors[:,0].mean()
h1_b_plus = h1_b_errors[:,1].mean() - h1_b_final

h1_vel_minus = h1_vel_final - h1_vel_errors[:,0].mean()
h1_vel_plus = h1_vel_errors[:,1].mean() - h1_vel_final



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
      ax.text(0.03,0.97,'V$_n$ = ' + str(round(vs_n_final,2)) + '$^{+' + str(round(vs_n_plus,1)) + '}_{-' + str(round(vs_n_minus,1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,
        fontsize=12., color='black')
      ax.text(0.03,0.91,'A$_n$ = ('+ str(round(am_n_final/am_n_exponent,2)) + '$^{+' + str(round(am_n_plus/am_n_exponent,1)) + '}_{-' + str(round(am_n_minus/am_n_exponent,1)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')


      ax.text(0.03,0.85,'FW$_n$ = '+ str(round(fw_n_final,2)) + '$^{+' + str(round(fw_n_plus,0)) + '}_{-' + str(round(fw_n_minus,0)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.03,0.79,'V$_b$ = '+ str(round(vs_b_final,2)) + '$^{+' + str(round(vs_b_plus,1)) + '}_{-' + str(round(vs_b_minus,1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')

      ax.text(0.03,0.73,'A$_b$ = ('+ str(round(am_b_final/am_n_exponent,2)) + '$^{+' + str(round(am_b_plus/am_n_exponent,3)) + '}_{-' + str(round(am_b_minus/am_n_exponent,3)) + '}$) ' + r'$\times$'+ ' 10$^{' + str(exponent) + '}$',
verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')

      ax.text(0.03,0.67,'FW$_b$ = '+ str(round(fw_b_final,2)) + '$^{+' + str(round(fw_b_plus,0)) + '}_{-' + str(round(fw_b_minus,0)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12.,
        color='black')
      ax.text(0.03,0.61,'log N(HI) = '+ str(round(h1_col_final,2)) + '$^{+' + str(round(h1_col_plus,2)) + '}_{-' + str(round(h1_col_minus,2)) + '}$',
        verticalalignment='top',horizontalalignment='left',
        transform=ax.transAxes,fontsize=12., color='black')
      ax.text(0.03,0.55,'b = '+ str(round(h1_b_final,2)) + '$^{+' + str(round(h1_b_plus,1)) + '}_{-' + str(round(h1_b_minus,1)) + '}$',
        verticalalignment='top',horizontalalignment='left',transform=ax.transAxes,fontsize=12., 
        color='black')
      ax.text(0.03,0.49,'V$_{HI}$ = '+ str(round(h1_vel_final,2)) + '$^{+' + str(round(h1_vel_plus,1)) + '}_{-' + str(round(h1_vel_minus,1)) + '}$',verticalalignment='top',horizontalalignment='left',
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




