import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
import meep as mp
import meep.materials as mt
import materials as mtp
import waveguide_dispersion as wg
import os
import geometry

clipboard_and_style_sheet.style_sheet()

# %%____________________________________________________________________________________________________________________
sim = wg.ThinFilmWaveguide(3, .3, .7, mtp.Al2O3, mt.LiNbO3, 30, 1, 10, 4)

# %%____________________________________________________________________________________________________________________
# etch_depth = wg.get_omega_axis(1 / .7, 1 / .3, 10)
etch_width = wg.get_omega_axis(1 / 5, 1 / 3, 20)
height = wg.get_omega_axis(1 / 1, 1 / 0.7, 10)
for w in etch_width:
    for h in height:
        sim.etch_width = w
        sim.height = h

        block_waveguide = sim.blk_wvgd  # save sim.blk_wvgd
        sim.blk_wvgd = geometry.convert_block_to_trapezoid(sim.blk_wvgd)  # set the blk_wvgd to a trapezoid
        res = sim.calc_dispersion(.8, 5, 25)  # run simulation
        sim.blk_wvgd = block_waveguide  # reset trapezoid back to blk_wvgd

        arr = np.c_[res.kx, res.freq, res.v_g[:, 0, 0]]  # 0, 0 -> first band, x-component (non-zero component)
        np.save(f'sim_output/06-16-2022/dispersion-curves/{w}_{h}.npy', arr)
        np.save(f'sim_output/06-16-2022/E-fields/{w}_{h}.npy', sim.E.__abs__() ** 2)
        np.save(f'sim_output/06-16-2022/eps/{w}_{h}.npy', sim.ms.get_epsilon())

# %%___________________________________________________Done ____________________________________________________________
# def width(s):
#     return float(s.split('_')[0])
#
#
# def depth(s):
#     return float(s.split('_')[1].split('.npy')[0])
#
#
# disp = [i.name for i in os.scandir('sim_output/06-16-2022/dispersion-curves')]
# disp = sorted(disp, key=width)
# disp = sorted(disp, key=depth)
#
#
# def plot(n, k_point=0, cmap='RdBu', alpha=.9):
#     s = disp[n]
#
#     E = np.load('sim_output/06-16-2022/E-fields/' + s)
#     band = 0
#     E = E[k_point, band, :, :, 1]
#
#     w = width(s)
#     d = depth(s)
#     sim.etch_width = w
#     sim.etch_depth = d
#     sim.sim.init_sim()
#     eps = sim.sim.get_epsilon()
#
#     plt.figure()
#     plt.imshow(eps[::-1, ::-1].T, cmap='binary')
#     plt.imshow(E[::-1, ::-1].T, cmap=cmap, alpha=alpha)
#
#
# step = 21
# for n in range(-step, 0, 1):
#     plot(n)
