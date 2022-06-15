import meep as mp
import meep.materials as mt
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import utilities as util
import h5py
import time
import waveguide_dispersion as wg
import materials as mtp

ridge = wg.RidgeWaveguide(
    width=5,
    height=4,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    resolution=20,  # 20 -> 40 made neglibile difference!
    num_bands=8,
    cell_width=13,
    cell_height=12
)

# at 3 micron you need 5-6 bands (anyways, 4 was too small)
# keep in mind you really just need 10 pts to do a prety good spline
ridge.width = 5
ridge.height = 5
ridge.cell_width = 13
ridge.cell_height = 13
ridge.num_bands = 12

ridge.wvgd_mdm = mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(1 / 1.55).diagonal())
ridge.sbstrt_mdm = mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(1 / 1.55).diagonal())
res = ridge.calc_w_from_k(.4, 5, 15)
# res = ridge.calc_dispersion(.8, 5, 10)

# %%____________________________________________________________________________________________________________________
res.plot_dispersion()
