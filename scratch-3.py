"""A 3D taper can be a block with angled side walls? Yep """

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import clipboard_and_style_sheet
import h5py
from mayavi import mlab

vertices = [
    mp.Vector3(-1, 1),
    mp.Vector3(1, 1),
    mp.Vector3(1, -1),
    mp.Vector3(-1, -1)
]

prism = mp.Prism(vertices=vertices, height=.5, material=mp.Medium(index=3), sidewall_angle=20)

sim = mp.Simulation(
    cell_size=mp.Vector3(5, 5, 4),
    geometry=[prism],
    resolution=20
)
sim.init_sim()
eps = sim.get_epsilon()

# %%____________________________________________________________________________________________________________________
plt.figure()
plt.imshow(eps[:, :, 40])

plt.figure()
plt.imshow(eps[40, :, :])

plt.figure()
plt.imshow(eps[:, 40, :])
