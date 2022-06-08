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
# Set up the geometry of the problem. One waveguide laid out in the x direction
wl_wvgd = 3.5
n_center_wvl = mt.LiNbO3.epsilon(1 / wl_wvgd)[2, 2]  # z polarization
w_wvgd = 0.5 * wl_wvgd / n_center_wvl  # width of the waveguide is half a wavelength wide

dpml = 1  # PML thickness

center_wvgd = mp.Vector3(0, 0, 0)  # where the waveguide is centered
sx = 16  # size of the cell in the x direction
sy = 6  # size of the cell in y direction

# %%____________________________________________________________________________________________________________________
# create the geometric objects from the above
blk = mp.Block(size=mp.Vector3(mp.inf, w_wvgd, mp.inf), center=center_wvgd)
cell = mp.Vector3(sx, sy, 0)
PML = mp.PML(dpml)

# %%____________________________________________________________________________________________________________________
# set the appropriate media for the geometric objects
blk.material = mt.LiNbO3

# %%____________________________________________________________________________________________________________________
# create the geometry and boundary layers list
geometry = [blk]
boundary_layers = [PML]

# %%____________________________________________________________________________________________________________________
# create a gaussian source instance and place it at the front of the waveguide
wl_src = 1.5
src = mp.GaussianSource(wavelength=wl_src, width=5)
pt = mp.Vector3() + center_wvgd
src_pt = pt + mp.Vector3(0, 0.25 * w_wvgd)

source = mp.Source(src=src,
                   component=mp.Ez,
                   center=src_pt,
                   size=mp.Vector3())
Sources = [source]

# %% Done with sources, initialize the simulation instance
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=Sources,
                    boundary_layers=boundary_layers,
                    resolution=30)
sim.use_output_directory('sim_output')

# %%____________________________________________________________________________________________________________________
# symmetries to exploit
Sym = [mp.Mirror(direction=mp.X)]
sim.symmetries = Sym

# %%____________________________________________________________________________________________________________________
sim.run(mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, pt, 1e-3))

# %%____________________________________________________________________________________________________________________
# Done! Look at simulation results!
with h5py.File('sim_output/1-scratch-ez.h5', 'r') as f:
    data = np.array(f[util.get_key(f)])

# %%____________________________________________________________________________________________________________________
save = False
fig, ax = plt.subplots(1, 1)
for n in range(0, data.shape[2], 1):
    ax.clear()
    ax.imshow(data[:, ::-1, n].T, cmap='jet', vmax=np.max(data), vmin=data.min())
    if save:
        plt.savefig(f'../fig/{n}.png')
    else:
        plt.pause(.01)
