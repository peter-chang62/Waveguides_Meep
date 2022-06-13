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

wl_wvgd = 3.5  # um
n_cntr_wl = mt.LiNbO3.epsilon((1 / wl_wvgd))[2, 2]  # ne polarization
wdth_wvgd = 0.5 * wl_wvgd / n_cntr_wl

ridge = wg.RidgeWaveguide(
    width=wdth_wvgd,
    # width=1,
    height=.5,
    substrate_medium=mt.SiO2,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    # substrate_medium=mp.Medium(index=1.45),  # non-dispersive
    # waveguide_medium=mp.Medium(index=3.45),  # non-dispersive
    resolution=45,
    num_bands=1,
    cell_width=5,
    cell_height=5
)

res = ridge.calc_dispersion(.4, 1.77, 19)
res.plot_dispersion()

# plt.figure()
# [plt.plot(res.kx, res.freq[:, n], '.-') for n in range(res.freq.shape[1])]
# plt.xlabel("k ($\mathrm{\mu m}$)")
# plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{\mu m}$)")
# plt.ylim(.25, 2.5)

# %%____________________________________________________________________________________________________________________
# omega = 1 / 1
# n = ridge.wvgd_mdm.epsilon(1 / 1.55)[2, 2]
# kmag_guess = n * omega
#
# eps_wvgd = ridge.wvgd_mdm.epsilon(omega)
# eps_sbstrt = ridge.sbstrt_mdm.epsilon(omega)
# ridge.wvgd_mdm = mp.Medium(epsilon_diag=eps_wvgd.diagonal())
# ridge.sbstrt_mdm = mp.Medium(epsilon_diag=eps_sbstrt.diagonal())
#
# k = ridge.find_k(
#     p=mp.EVEN_Y,
#     omega=1,
#     band_min=1,
#     band_max=4,
#     korig_and_kdir=mp.Vector3(1),
#     tol=1e-6,
#     kmag_guess=kmag_guess,
#     kmag_min=kmag_guess * .1,
#     kmag_max=kmag_guess * 10
# )

# # As you can see, the light line for free space isn't the constraint,
# # it's the light line for the oxide!
# E = ridge.ms.get_efield(4, False)
# eps = ridge.ms.get_epsilon()
# for n, title in enumerate(['Ex', 'Ey', 'Ez']):
#     plt.figure()
#     x = E[:, :, 0, n].__abs__() ** 2
#     plt.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
#     plt.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
#     plt.axis(False)
#     plt.title(title)
