import meep as mp
import meep.materials as mt
import matplotlib.pyplot as plt
import numpy as np
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
# minimum and maximum wavelength
wl_min, wl_max = 1.4, 1.6

# %%____________________________________________________________________________________________________________________
# parameters to calculate for the cell
wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl
hght_wvgd = 0.5  # 500 nm
cntr_wvgd = mp.Vector3(0, 0, 0)  # waveguide center

sx = 5
sy = 5
sz = 0

dpml = 1  # PML thickness

# %%____________________________________________________________________________________________________________________
# use the above code block to create the MEEP geometries, simulation cell, and boundary layers
blk1 = mp.Block(
    size=mp.Vector3(wdth_wvgd, hght_wvgd, mp.inf),
    center=cntr_wvgd
)

hght_blk2 = (sy / 2) - (hght_wvgd / 2) + cntr_wvgd.y
offst_blk2 = (hght_blk2 / 2) + (hght_wvgd / 2) - cntr_wvgd.y
blk2 = mp.Block(
    size=mp.Vector3(mp.inf, hght_blk2, mp.inf),
    center=mp.Vector3(0, -offst_blk2)
)

cell = mp.Vector3(sx, sy, sz)

# Absorber boundary layers
ABSX = mp.Absorber(dpml, direction=mp.X)  # left and right
ABSZ = mp.Absorber(dpml, direction=mp.Z)  # front and back
ABSY = mp.Absorber(dpml, direction=mp.Y, side=mp.Low)  # bottom

# PML boundary layers
PMLY = mp.PML(dpml, direction=mp.Y, side=mp.High)  # top

# %%____________________________________________________________________________________________________________________
