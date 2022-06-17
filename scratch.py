import meep as mp
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import time

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def convert_block_to_trapezoid(blk, angle_deg):
    assert isinstance(blk, mp.Block), f"blk needs to be an mp.Block instance but got {type(blk)}"
    blk: mp.Block

    size = blk.size
    y_added = size.z / np.tan(deg_to_rad(angle_deg))
    pt1 = mp.Vector3(y=-size.y / 2, z=size.z / 2)
    pt2 = mp.Vector3(y=size.y / 2, z=size.z / 2)
    pt3 = mp.Vector3(y=size.y / 2 + y_added, z=-size.z / 2)
    pt4 = mp.Vector3(y=-size.y / 2 - y_added, z=-size.z / 2)
    vertices = [pt1, pt2, pt3, pt4]
    height = mp.inf
    prism = mp.Prism(vertices, height, axis=mp.Vector3(x=1))

    prism.material = blk.material
    return prism


# %%____________________________________________________________________________________________________________________
blk = mp.Block(mp.Vector3(mp.inf, 3, .7), material=mp.Medium(index=2))
trap = convert_block_to_trapezoid(blk, 80)
sim1 = mp.Simulation(cell_size=mp.Vector3(0, 5, 5), geometry=[blk], resolution=30)
sim2 = mp.Simulation(cell_size=mp.Vector3(0, 5, 5), geometry=[trap], resolution=30)

sim1.init_sim()
sim2.init_sim()
eps1 = sim1.get_epsilon()
eps2 = sim2.get_epsilon()

# checks out!
plt.imshow(eps1[::-1, ::-1].T, cmap='binary', alpha=0.5)
plt.imshow(eps2[::-1, ::-1].T, cmap='binary', alpha=0.5)
