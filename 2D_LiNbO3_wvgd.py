"""Getting started: 2D simulation in a straight LiNbO3 waveguide"""

import meep as mp
import meep.materials as mt
import matplotlib.pyplot as plt
import numpy as np
import utilities as util
import h5py
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
wl_min, wl_max = 1.4, 1.6

# %%____________________________________________________________________________________________________________________
# Set up the geometry of the problem. One waveguide laid out in the x direction
wl_wvgd = 3.5
n_center_wvl = mt.LiNbO3.epsilon(1 / wl_wvgd)[2, 2]  # z polarization
wdth_wvgd = 0.5 * wl_wvgd / n_center_wvl  # width of the waveguide is half a wavelength wide

dpml = 1  # PML thickness

cntr_wvgd = mp.Vector3(0, 0, 0)  # where the waveguide is centered
sx = 16  # size of the cell in the x direction
sy = 6  # size of the cell in y direction

# %%____________________________________________________________________________________________________________________
# create the geometric objects from the above
blk1 = mp.Block(size=mp.Vector3(mp.inf, wdth_wvgd, mp.inf), center=cntr_wvgd)

hght_blk2 = (sy / 2) - (wdth_wvgd / 2) + cntr_wvgd.y
offst_blk2 = (hght_blk2 / 2) + (wdth_wvgd / 2) - cntr_wvgd.y
blk2 = mp.Block(size=mp.Vector3(mp.inf, hght_blk2, mp.inf),
                center=mp.Vector3(0, -offst_blk2))

cell = mp.Vector3(sx, sy, 0)

PMLY = mp.PML(dpml, direction=mp.Y, side=mp.High)
PMLList = [PMLY]

ABSX = mp.Absorber(dpml, direction=mp.X)
ABSY = mp.Absorber(dpml, direction=mp.Y, side=mp.Low)
ABSList = [ABSX, ABSY]

# %%____________________________________________________________________________________________________________________
# set the appropriate media for the geometric objects
bw = np.array([1 / wl_max, 1 / wl_min])
f_src = float(np.diff(bw) / 2 + bw[0])

# blk1.material = mt.LiNbO3
# blk2.material = mt.SiO2
blk1.material = mp.Medium(epsilon=mt.LiNbO3.epsilon(f_src)[2, 2])
blk2.material = mp.Medium(epsilon=mt.SiO2.epsilon(f_src)[2, 2])

# %%____________________________________________________________________________________________________________________
# create the geometry and boundary layers list
geometry = [blk1, blk2]
boundary_layers = [*ABSList, *PMLList]

# %%____________________________________________________________________________________________________________________
# create a gaussian source instance and place it at the front of the waveguide
src = mp.GaussianSource(frequency=f_src, fwidth=float(np.diff(bw)))

pt = mp.Vector3() + cntr_wvgd
src_pt = pt + mp.Vector3(0, 0.25 * wdth_wvgd)

source = mp.Source(src=src,
                   component=mp.Ez,  # ne polarization
                   center=src_pt,
                   size=mp.Vector3())
Sources = [source]

# %%____________________________________________________________________________________________________________________
# Done with sources, initialize the simulation instance
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=Sources,
                    boundary_layers=boundary_layers,
                    resolution=30)
sim.use_output_directory('sim_output')

# %%____________________________________________________________________________________________________________________
# symmetries to exploit
sim.symmetries = [mp.Mirror(direction=mp.X)]

# %%____________________________________________________________________________________________________________________
sim.run(mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# %%____________________________________________________________________________________________________________________
# Done! Look at simulation results!
with h5py.File('sim_output/2D_LiNbO3_wvgd-ez.h5', 'r') as f:
    data = np.array(f[util.get_key(f)])

# %%____________________________________________________________________________________________________________________
# save = False
# fig, ax = plt.subplots(1, 1)
# for n in range(0, data.shape[2], 1):
#     ax.clear()
#     ax.imshow(data[:, ::-1, n].T, cmap='jet', vmax=np.max(data), vmin=data.min())
#     if save:
#         plt.savefig(f'../fig/{n}.png')
#     else:
#         plt.pause(.01)
