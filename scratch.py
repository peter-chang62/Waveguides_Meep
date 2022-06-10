"""
3D_LiNbO3_wvgd_harminv.py uses Harminv to retrieve the modes, here I want to use MPB which should be much faster
"""

import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import utilities as util
import h5py
import time

clipboard_and_style_sheet.style_sheet()

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
blk_wvgd = mp.Block(
    size=mp.Vector3(mp.inf, wdth_wvgd, hght_wvgd),
    center=cntr_wvgd)

hght_blk2 = (sy / 2) - (hght_wvgd / 2) + cntr_wvgd.y
offst_blk2 = (hght_blk2 / 2) + (hght_wvgd / 2) - cntr_wvgd.y
blk_sbstrt = mp.Block(
    size=mp.Vector3(mp.inf, mp.inf, hght_blk2),
    center=mp.Vector3(0, 0, -offst_blk2))

lattice = mp.Lattice(size=mp.Vector3(0, sy, sz))

geometry = [blk_wvgd, blk_sbstrt]

# %%____________________________________________________________________________________________________________________
resolution = 30
num_bands = 4

ms = mpb.ModeSolver(
    geometry_lattice=lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_bands)

# %%____________________________________________________________________________________________________________________
wvgd_mdm = mt.LiNbO3
sbstrt_mdm = mt.SiO2

# minimum and maximum wavelength
wl_min, wl_max = 0.4, 1.77
k_points = mp.interpolate(19, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
FREQ = np.zeros((len(k_points), 4))
EPS_WVGD = np.zeros(len(k_points))
EPS_SBSTRT = np.zeros(len(k_points))

start = time.time()
for n, k in enumerate(k_points):
    eps_wvgd = wvgd_mdm.epsilon(k.x)[2, 2]
    eps_sbstrt = sbstrt_mdm.epsilon(k.x)[2, 2]
    blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
    blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)

    ms.k_points = [k]
    ms.run()

    FREQ[n] = ms.all_freqs[0]
    EPS_WVGD[n] = eps_wvgd.real
    EPS_SBSTRT[n] = eps_sbstrt.real

    print(f'____________________________{len(k_points) - n}________________________________________')

stop = time.time()
print(f'finished after {(stop - start) / 60} minutes')

# %%____________________________________________________________________________________________________________________
# visualization check (make sure epsilons are not all 1, or else the plot is blank)
sim = mp.Simulation(cell_size=lattice.size,
                    geometry=geometry,
                    resolution=resolution)

# %%____________________________________________________________________________________________________________________
# plt.figure()
# kx = np.array([i.x for i in k_points])
# plt.plot(1 / kx, FREQ[:, 0], 'o-')
# plt.plot(1 / kx, FREQ[:, 1], 'o-')
# plt.plot(1 / kx, FREQ[:, 2], 'o-')
# plt.plot(1 / kx, FREQ[:, 3], 'o-')
# plt.plot(1 / kx, kx, 'k', label='light line')
# plt.legend(loc='best')

plt.figure()
kx = np.array([i.x for i in k_points])
plt.plot(kx, FREQ[:, 0], 'o-')
plt.plot(kx, FREQ[:, 1], 'o-')
plt.plot(kx, FREQ[:, 2], 'o-')
plt.plot(kx, FREQ[:, 3], 'o-')
plt.plot(kx, kx, 'k', label='light line')
plt.plot(kx, kx / EPS_SBSTRT, 'k--', label='light line substrate')
plt.legend(loc='best')
