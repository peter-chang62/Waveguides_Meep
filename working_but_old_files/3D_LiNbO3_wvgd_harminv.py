"""
Once dispersive, it's pretty cost intensive
"""

import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
import matplotlib.pyplot as plt
import h5py
from mayavi import mlab

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
"""Geometry """

# minimum and maximum wavelength
wl_min, wl_max = .4, 1.77
# wl_min, wl_max = 1 / np.array(mt.LiNbO3.valid_freq_range)[::-1]

# %%____________________________________________________________________________________________________________________
# parameters to calculate for the cell
wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl
hght_wvgd = 0.5  # 500 nm
cntr_wvgd = mp.Vector3(0, 0, 0)  # waveguide center

sy = 5
sz = 5

dpml = 1  # PML thickness

# %%____________________________________________________________________________________________________________________
# use the above code block to create the MEEP geometries, simulation cell, and boundary layers
blk1 = mp.Block(
    size=mp.Vector3(mp.inf, wdth_wvgd, hght_wvgd),
    center=cntr_wvgd)

hght_blk2 = (sy / 2) - (hght_wvgd / 2) + cntr_wvgd.y
offst_blk2 = (hght_blk2 / 2) + (hght_wvgd / 2) - cntr_wvgd.y
blk2 = mp.Block(
    size=mp.Vector3(mp.inf, mp.inf, hght_blk2),
    center=mp.Vector3(0, 0, -offst_blk2))

cell = mp.Vector3(0, sy, sz)

# Absorber boundary layers
ABSY = mp.Absorber(dpml, mp.Y)  # left and right
ABSZ = mp.Absorber(dpml, mp.Z, side=mp.Low)  # bottom
ABSList = [ABSY, ABSZ]

# PML boundary layers
PMLZ = mp.PML(dpml, direction=mp.Z, side=mp.High)  # top
PMLList = [PMLZ]

geometry = [blk1, blk2]
boundary_layers = [*ABSList, *PMLList]

# %%____________________________________________________________________________________________________________________
bw = np.array([1 / wl_max, 1 / wl_min])
f_src = float(np.diff(bw) / 2 + bw[0])
# blk1.material = mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(f_src).diagonal())
# blk2.material = mp.Medium(epsilon_diag=mt.SiO2.epsilon(f_src).diagonal())
blk1.material = mt.LiNbO3
blk2.material = mt.SiO2

# %%____________________________________________________________________________________________________________________
# I like to think of the simulation as tied to the simulation cell
# I can plot now and vet the geometries,
# and I'll continue to add sources below (then plot to vet sources again)
sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=40)

sim.use_output_directory('sim_output')

# %%____________________________________________________________________________________________________________________
"""Sources """

src = mp.GaussianSource(
    frequency=f_src,
    fwidth=float(np.diff(bw)))

pt_src_offst = mp.Vector3(0, 0.25 * wdth_wvgd, 0.25 * hght_wvgd)
pt_src = cntr_wvgd + pt_src_offst
source = mp.Source(
    src=src,
    component=mp.Ez,  # longitudinal is X, polarization is Z (that lies on ne for LiNbO3)
    center=pt_src)

sim.sources = [source]

# %%____________________________________________________________________________________________________________________
k_points = mp.interpolate(19, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
freq = sim.run_k_points(300, k_points)

# %%____________________________________________________________________________________________________________________
kx = np.array([i.x for i in k_points])
plt.figure()
plt.plot([kx.min(), kx.max()], [kx.min(), kx.max()], 'k', label='light line')
for n in range(len(freq)):
    if len(freq[n]) > 0:
        [plt.plot(kx[n], i.real, marker='.', color='C0') for i in freq[n]]
plt.legend(loc='best')
plt.xlabel("k ($\mathrm{\mu m}$)")
plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{\mu m}$)")
plt.ylim(.25, 2.5)
