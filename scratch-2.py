"""complex transmission of a waveguide taper, you can set things up following mode-decomposition.py from
meep/examples/ """

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

# %% ___________________________________________________________________________________________________________________
# dimensions
w1 = 1.0  # width of waveguide 1
w2 = 2.0  # width of waveguide 2
Lw = 10.0  # length of waveguides 1 and 2

Lt = 2  # length of taper

dair = 3.0  # length of air region
dpml_x = 6.0  # length of PML in x direction
dpml_y = 2.0  # length of PML in y direction

sy = dpml_y + dair + w2 + dair + dpml_y

sx = dpml_x + Lw + Lt + Lw + dpml_x

# %% ___________________________________________________________________________________________________________________
# cell, geometry and boundary layers
cell_size = mp.Vector3(sx, sy, 0)

boundary_layers = [mp.PML(dpml_x, direction=mp.X),
                   mp.PML(dpml_y, direction=mp.Y)]

# linear taper
vertices = [mp.Vector3(-0.5 * sx - 1, 0.5 * w1),
            mp.Vector3(-0.5 * Lt, 0.5 * w1),
            mp.Vector3(0.5 * Lt, 0.5 * w2),
            mp.Vector3(0.5 * sx + 1, 0.5 * w2),
            mp.Vector3(0.5 * sx + 1, -0.5 * w2),
            mp.Vector3(0.5 * Lt, -0.5 * w2),
            mp.Vector3(-0.5 * Lt, -0.5 * w1),
            mp.Vector3(-0.5 * sx - 1, -0.5 * w1)]

geometry = [mp.Prism(vertices, height=mp.inf, material=mp.Medium(epsilon=12.0))]

resolution = 25  # pixels/μm
sim = mp.Simulation(resolution=resolution,
                    cell_size=cell_size,
                    boundary_layers=boundary_layers,
                    geometry=geometry,
                    )

# %% ___________________________________________________________________________________________________________________
# sources
lcen = 6.67  # mode wavelength
fcen = 1 / lcen  # mode frequency
df = 0.2
src_pt = mp.Vector3(-0.5 * sx + dpml_x + 0.2 * Lw)
sources = [mp.EigenModeSource(src=mp.GaussianSource(fcen, fwidth=df * fcen),
                              center=src_pt,
                              size=mp.Vector3(y=sy - 2 * dpml_y),
                              eig_match_freq=True,
                              eig_parity=mp.ODD_Z + mp.EVEN_Y)]
sim.sources = sources

# %% ___________________________________________________________________________________________________________________
sim.symmetries = [mp.Mirror(mp.Y)]

# %% ___________________________________________________________________________________________________________________
