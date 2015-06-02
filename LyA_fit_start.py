import lyapy

## Read in fits file ##
input_filename = raw_input("Enter fits file name: ")


## Define Initial Guess Parameters ##
lya_rest = 1215.67
vs_n = -12.93  # narrow LyA velocity shift     ; [km/s]  
am_n = 13.5e-12     # amplitude of narrow LyA       ; [erg/cm2/s/A]
fw_n = 131.17      # FWHM of narrow LyA            ; [km/s]  

vs_b = -93.04   # broad LyA velocity shift      ; [km/s]
am_b = 1.46e-14          # amplitude of broad LyA        ; [erg/cm2/s/A]
fw_b = 317.14            # FWHM of broad LyA             ; [km/s]

h1_col = 18.282         # ISM N(HI)                     ; [cm-2]
h1_b   = 12.83       # ISM b(HI)
h1_vel = -3.69  # HI velocity  ; mpfit does NOT like letting the velocity run negative....need to fix. 
d2h = 1.5e-5                                     

initial_parameters = [vs_n,am_n,fw_n,vs_b,am_b,fw_b,h1_col,h1_b,h1_vel,d2h]

lyapy.LyA_fit(input_filename,initial_parameters,save_figure=True)



