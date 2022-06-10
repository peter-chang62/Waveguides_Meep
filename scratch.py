import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import utilities as util
import h5py

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
# minimum and maximum wavelength
wl_min, wl_max = 1 / np.array(mt.LiNbO3.valid_freq_range)[::-1]

# %%____________________________________________________________________________________________________________________
# parameters to calculate for the cell
wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl
hght_wvgd = 0.5  # 500 nm
cntr_wvgd = mp.Vector3(0, 0, 0)  # waveguide center

sy = 8
sz = 8

dpml = 1  # PML thickness

# %%____________________________________________________________________________________________________________________
# use the above code block to create the MEEP geometries, simulation cell, and boundary layers
blk1 = mp.Block(
    size=mp.Vector3(mp.inf, wdth_wvgd, hght_wvgd),
    center=cntr_wvgd
)

hght_blk2 = (sy / 2) - (hght_wvgd / 2) + cntr_wvgd.y
offst_blk2 = (hght_blk2 / 2) + (hght_wvgd / 2) - cntr_wvgd.y
blk2 = mp.Block(
    size=mp.Vector3(mp.inf, mp.inf, hght_blk2),
    center=mp.Vector3(0, 0, -offst_blk2)
)

lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

geometry = [blk1, blk2]

# %%____________________________________________________________________________________________________________________
# calculate epsilon
bw = np.array([1 / wl_max, 1 / wl_min])
f_src = float(np.diff(bw) / 2 + bw[0])

blk1.material = mt.LiNbO3
blk2.material = mt.SiO2

# %%____________________________________________________________________________________________________________________
# visualize comment out if not plotting (changes epsilon)!
# blk1.material = mp.Medium(epsilon=mt.LiNbO3.epsilon(f_src)[2, 2])
# blk2.material = mp.Medium(epsilon=mt.SiO2.epsilon(f_src)[2, 2])
# sim = mp.Simulation(cell_size=lattice.size,
#                     geometry=geometry,
#                     resolution=20)
# sim.plot2D()

# %%____________________________________________________________________________________________________________________
num_bands = 4
k_points = mp.interpolate(19, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])

# %%____________________________________________________________________________________________________________________
ms = mpb.ModeSolver(
    geometry_lattice=lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=30,
    num_bands=num_bands
)

# %%____________________________________________________________________________________________________________________
ms.run_te_yodd(mpb.display_group_velocities)

for i in k_points:
    pass