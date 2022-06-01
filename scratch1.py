import matplotlib.pyplot as plt
import meep as mp
import clipboard_and_style_sheet
import numpy as np
import h5py
import utilities as ut

# %%
"""Dimensions are normalized to micron in meep 
2D cell 16 um x 16 um x 0 um 
items in geometry are overlaid on top of each other, with later items taking priority (see microring example in meep) 
"""

# simulation cell
cell = mp.Vector3(20, 20, 0)

# two blocks -> bent waveguide
geometry = [mp.Block(mp.Vector3(12, 1, mp.inf),
                     center=mp.Vector3(-2.5, -3.5),
                     material=mp.Medium(epsilon=12)),
            mp.Block(mp.Vector3(1, 12, mp.inf),
                     center=mp.Vector3(3.5, 2),
                     material=mp.Medium(epsilon=12))]

# pml layer 1 um thick, by default surrounding the entire cell
pml_layers = [mp.PML(thickness=4.0)]

# %%
"""Sources are defined as a list (of sources). A continuous source is CW. If you do not define a size parameter, 
it is a point source, otherwise you can give a size to make it a line source. Define either the vacuum wavelength (um) 
or the frequency (as inverse wavelength, so it's really a wavenumber, and no factor of 2 pi) 

Specify which component (Ex, Ey, Ez or Hx, Hy, Hyz), and the location (center). 
"""

component = mp.Ez
# together with size is centered in middle, and extends the width of the waveguide
center = mp.Vector3(-7, -3.5)
size = mp.Vector3(0, 1)
wavelength = 2 * (11 ** 0.5)

cont_src = mp.ContinuousSource(wavelength=wavelength, width=20)
gaussian_src = mp.GaussianSource(wavelength=wavelength, width=10, is_integrated=True, cutoff=10)

# CW line source
# sources = [mp.Source(cont_src,
#                      component=component,
#                      center=center,
#                      size=size)]

# GaussianBeamSource is a subclass of Source
sources = [mp.GaussianBeamSource(src=gaussian_src,
                                 center=center,
                                 size=size,
                                 beam_x0=mp.Vector3(0, 0),
                                 beam_kdir=mp.Vector3(1, 0),
                                 beam_w0=1,
                                 beam_E0=mp.Vector3(0, 0, 1))]

# %%
"""
Initialize the simulation: cell, boundary_layers (for the cell, the boudnary conditions inside the cell is set by
the geometry), the geometry (list), sources (list), and the resolution (int) 
"""

# spatial resolution (pixels / um), or (pixels / distance unit) that is used in the simulation, the courant parameter
# (S dx = c dt) is used to determine the time step, suggestion is at least 8 pixels / wavelength in the highest
# dielectric
resolution = 10

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# %%
"""
The way you retrieve simulation results is by passing arguments to the run command, there are quite a few options"""

sim.use_output_directory('sim_output/')
sim.run(  # mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(1, mp.output_efield_z)),
    until_after_sources=mp.stop_when_fields_decayed(dt=50, c=mp.Ez, pt=mp.Vector3(0), decay_by=1e-3))

# %%
"""Simulation is Done! Retrieve the h5 files. Meep outputs everything as h5 files because h5utils makes them easy to 
handle (in terminal). I'm having package issues with that, however, so I'm just using numpy and matplotlib to 
visualize """

with h5py.File('sim_output/scratch1-ez.h5', 'r') as f:
    data = np.array(f[ut.get_key(f)])

fig, ax = plt.subplots(1, 1)
for n in range(data.shape[-1]):
    ax.clear()
    ax.imshow(data[:, :, n].T, vmax=np.max(data), vmin=np.min(data), cmap='nipy_spectral')
    plt.pause(.001)
