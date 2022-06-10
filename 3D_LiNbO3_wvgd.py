"""
3D components:
    1. sx
    2. un-comment ABSX for boundary_layers
"""

import meep as mp
import meep.materials as mt
import numpy as np
import clipboard_and_style_sheet
import h5py
from mayavi import mlab

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
"""Geometry """

# minimum and maximum wavelength
wl_min, wl_max = 1.4, 1.6

# %%____________________________________________________________________________________________________________________
# parameters to calculate for the cell
wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl
hght_wvgd = 0.5  # 500 nm
cntr_wvgd = mp.Vector3(0, 0, 0)  # waveguide center

sx = 12
sy = 7
sz = 7

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

cell = mp.Vector3(sx, sy, sz)

# Absorber boundary layers
ABSX = mp.Absorber(dpml, mp.X)  # front and back
ABSY = mp.Absorber(dpml, mp.Y)  # left and right
ABSZ = mp.Absorber(dpml, mp.Z, side=mp.Low)  # bottom
ABSList = [
    ABSX,  # toggle 3D
    ABSY,
    ABSZ,
]

# PML boundary layers
PMLZ = mp.PML(dpml, direction=mp.Z, side=mp.High)  # top
PMLList = [PMLZ]

geometry = [
    blk1,
    blk2
]
boundary_layers = [
    *ABSList,
    *PMLList
]

# %%____________________________________________________________________________________________________________________
bw = np.array([1 / wl_max, 1 / wl_min])
f_src = float(np.diff(bw) / 2 + bw[0])
blk1.material = mp.Medium(epsilon=mt.LiNbO3.epsilon(f_src)[2, 2])
blk2.material = mp.Medium(epsilon=mt.SiO2.epsilon(f_src)[2, 2])

# %%____________________________________________________________________________________________________________________
# I like to think of the simulation as tied to the simulation cell
# I can plot now and vet the geometries,
# and I'll continue to add sources below
sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    boundary_layers=boundary_layers,
    resolution=20
)

sim.use_output_directory('sim_output')

# %%____________________________________________________________________________________________________________________
"""Sources """

src = mp.GaussianSource(
    frequency=f_src,
    fwidth=float(np.diff(bw))
)

pt_src_offst = mp.Vector3(0, 0.25 * wdth_wvgd, 0.25 * hght_wvgd)
pt_src = cntr_wvgd + pt_src_offst
source = mp.Source(
    src=src,
    component=mp.Ez,  # longitudinal is X, polarization is Z, that lies on ne for LiNbO3
    center=pt_src
)

sim.sources = [source]

# %%____________________________________________________________________________________________________________________
"""Run """
sim.run(
    mp.to_appended("ez", mp.at_every(.6, mp.output_efield_z)),
    until_after_sources=300
)

# %%____________________________________________________________________________________________________________________
# Done! Look at simulation results!
f = h5py.File('sim_output/3D_LiNbO3_wvgd-ez.h5', 'r')
data = f.get('ez')
mlab.figure()
Zero = np.zeros((data.shape[:-1]))
for n in range(data.shape[-1]):
    mlab.quiver3d(Zero, Zero, data[:, :, :, n])
    mlab.savefig(f'fig/{n}.png')
    mlab.clf()
    print(data.shape[-1] - n)
