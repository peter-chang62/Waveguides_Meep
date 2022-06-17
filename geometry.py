import meep as mp
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import time

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def convert_block_to_trapezoid(blk, angle_deg=80):
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
