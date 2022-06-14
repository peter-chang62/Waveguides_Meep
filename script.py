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

wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl

ridge = wg.RidgeWaveguide(
    # width=wdth_wvgd,
    width=0.5,
    height=0.7,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    # substrate_medium=mp.Medium(index=1.45),  # non-dispersive
    # waveguide_medium=mp.Medium(index=3.45),  # non-dispersive
    resolution=20,  # 20 -> 40 made neglibile difference!
    num_bands=1,
    cell_width=7,
    cell_height=7
)

# at 3 micron you need 5-6 bands (anyways, 4 was too small)
# keep in mind you really just need 10 pts to do a prety good spline
ridge.width = 10
ridge.height = 2
ridge.cell_width = 20
ridge.cell_height = 10
ridge.num_bands = 7

ridge.wvgd_mdm = mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(1 / 1.55).diagonal())
ridge.sbstrt_mdm = mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(1 / 1.55).diagonal())
res = ridge.calc_w_from_k(.4, 5, 10)
# res = ridge.calc_dispersion(.8, 5, 10)

# %%____________________________________________________________________________________________________________________
res.plot_dispersion()
plt.plot(res.sm_dispersion[:, 0], res.sm_dispersion[:, 1], '.-', color='b', label='sm-dispersion')

plot = lambda n: ridge.plot_mode(res.sm_bands[n], n)

for n in range(len(res.kx)):
    plot(n)
