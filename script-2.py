import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import meep as mp
import meep.materials as mt
import materials as mtp
import waveguide_dispersion as wg
import os

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
sim = wg.ThinFilmWaveguide(3, .3, .7, mtp.Al2O3, mt.LiNbO3, 30, 1, 10, 4)

# %%____________________________________________________________________________________________________________________
# resolution requirement, negligibly small!
# not only that, but I think MPB does some sort of pixel smoothing,
# so I get 10 nm changes to be noticeable in plot_eps() even when resolution is 30 pixels/um
# res1 = sim.calc_dispersion(.8, 3.5, 15)
# sim.resolution = 30
# res2 = sim.calc_dispersion(.8, 3.5, 15)
# plt.plot(1 / res1.kx - 1 / res2.kx)

# %%____________________________________________________________________________________________________________________
etch_depth = wg.get_omega_axis(1 / .7, 1 / .3, 10)
etch_width = wg.get_omega_axis(1 / 5, 1 / 3, 20)
for w in etch_width:
    for d in etch_depth:
        sim.etch_width = w
        sim.etch_depth = d
        res = sim.calc_dispersion(.8, 5, 25)

        arr = np.c_[res.kx, res.freq]
        np.save(f'sim_output/06-16-2022/dispersion-curves/{w}_{d}.npy', arr)
        np.save(f'sim_output/06-16-2022/E-fields/{w}_{d}.npy', sim.E.__abs__() ** 2)
