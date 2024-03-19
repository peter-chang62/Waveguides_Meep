import meep as mp
import numpy as np

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def convert_block_to_trapezoid(blk, angle_deg=80):
    assert isinstance(
        blk, mp.Block
    ), f"blk needs to be an mp.Block instance but got {type(blk)}"
    blk: mp.Block

    # -------------------------------------------------------------------------
    # there are basically two things to get right about geometric objects:

    #   1. dimensions
    #   2. material

    # I noticed that setting prism.center = blk.center was a bad idea, so don't
    # go adding that later! The dimensions are correct the way it is done
    # here
    # -------------------------------------------------------------------------

    size = blk.size
    y_added = size.z / np.tan(deg_to_rad(angle_deg))
    pt1 = mp.Vector3(0, size.y / 2 + y_added, -size.z / 2)
    pt2 = mp.Vector3(0, -size.y / 2 - y_added, -size.z / 2)
    pt3 = mp.Vector3(0, -size.y / 2, size.z / 2)
    pt4 = mp.Vector3(0, size.y / 2, size.z / 2)
    vertices = [pt1, pt2, pt3, pt4]
    prism = mp.Prism(
        vertices=vertices,
        height=mp.inf,
        sidewall_angle=0,
        material=blk.material,
        axis=mp.Vector3(1, 0, 0),
    )

    # add the z_offset to center
    prism.center.z += blk.center.z
    return prism
