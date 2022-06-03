import meep as mp
import meep.materials as mt
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import numpy as np

# %% Set up the geometry of the problem. One waveguide laid out in the x direction
w = 1  # width of the waveguide
dpml = 1  # PML thickness

sx = 16  # size of the cell in the x direction
sy = 8  # size of cell in y direction

# %% create the geometric objects from the cell above
blk = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf))
cell = mp.Vector3(sx, sy, 0)
PML = mp.PML(dpml)

# %% set the appropriate media for the geometric objects
blk.material = mt.LiNbO3

# %% create the geometry and boundary layers list
geometry = [blk]
boundary_layers = [PML]

# %% Done with geometry, moving on to sources, initialize an empty list for sources
Sources = []

# %% create a gaussian source instance and place it at the front of the waveguide
src = mp.GaussianSource(frequency=1.5, width=5)
source = mp.Source(src=src,
                   component=mp.Ez,
                   center=mp.Vector3(-0.5 * sx + dpml),
                   size=mp.Vector3(0, w))
Sources.append(source)

# %% Done with sources, initialize the simulation instance
sim = mp.Simulation(cell_size=cell,
                    geometry=geometry,
                    sources=Sources,
                    boundary_layers=boundary_layers,
                    resolution=20)
