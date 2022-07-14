import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import meep as mp
import meep.materials as mt
import materials as mtp
import waveguide_dispersion as wg
import os
import geometry
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
# Gayer paper Sellmeier equation for ne (taken from PyNLO
# 1 / omega is in um -> multiply by 1e3 to get to nm -> then square to go from ne to eps
eps_func_wvgd = lambda omega: Gayer5PctSellmeier(24.5).n((1 / omega) * 1e3) ** 2

# %%____________________________________________________________________________________________________________________
sim = wg.ThinFilmWaveguide(3, .3, .7, mtp.Al2O3, mt.LiNbO3, 30, 1, 10, 4)

# %%____________________________________________________________________________________________________________________
etch_width = wg.get_omega_axis(1 / 3, 1 / 0.3, 20)  # 300 nm to 3 um in 135 nm steps
# height = wg.get_omega_axis(1 / 1, 1 / 0.7, 10)  # 700 nm to 1 um in 30 nm steps
height = np.arange(1.05, 3.05, .05)  # continue parameter sweep: 1050 nm to 3000 nm in 50 nm steps
for w in etch_width:
    for h in height:
        sim.etch_width = w
        sim.height = h

        block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
        sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
        res = sim.calc_dispersion(.8, 5, 25, eps_func_wvgd=eps_func_wvgd)  # run simulation
        sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

        arr = np.c_[res.kx, res.freq, res.v_g[:, 0, 0]]  # 0, 0 -> first band, x-component (non-zero component)
        np.save(f'sim_output/06-16-2022/dispersion-curves/{w}_{h}.npy', arr)
        np.save(f'sim_output/06-16-2022/E-fields/{w}_{h}.npy', sim.E[:, :, :, :, 1].__abs__() ** 2)
        np.save(f'sim_output/06-16-2022/eps/{w}_{h}.npy', sim.ms.get_epsilon())
