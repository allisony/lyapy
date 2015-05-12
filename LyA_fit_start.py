import lyapy

## Read in fits file ##
input_filename = raw_input("Enter fits file name: ")


## Define Initial Guess Parameters ##
lya_rest = 1215.67
vs_n = 14.5256 # narrow LyA velocity shift     ; [km/s]  
am_n = 4.60542e-11       # amplitude of narrow LyA       ; [erg/cm2/s/A]
fw_n = 131.310         # FWHM of narrow LyA            ; [km/s]  

vs_b = 11.4870    # broad LyA velocity shift      ; [km/s]
am_b = 2.90763e-12           # amplitude of broad LyA        ; [erg/cm2/s/A]
fw_b = 390.009              # FWHM of broad LyA             ; [km/s]

h1_col = 17.88         # ISM N(HI)                     ; [cm-2]
h1_b   = 11.4         # ISM b(HI)
h1_vel = 17.0 # HI velocity  ; mpfit does NOT like letting the velocity run negative....need to fix. 
d2h = 1.5e-5                                     

initial_parameters = [vs_n,am_n,fw_n,vs_b,am_b,fw_b,h1_col,h1_b,h1_vel,d2h]

lyapy.LyA_fit(input_filename,initial_parameters,save_figure=True)



