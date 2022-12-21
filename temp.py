"""cleaning up sim_output directories"""
import os
import numpy as np
import waveguide_dispersion as wg
import itertools
import scipy.constants as sc

etch_width = wg.get_omega_axis(1 / 3, 1 / 0.3, 20)  # 300 nm to 3 um in 135 nm steps
etch_depth = np.arange(0.1, 1.05, .05)
etch_width = np.round(etch_width, 3)  # round the etch width
etch_depth = np.round(etch_depth, 3)  # round the etch depth
params = np.asarray(list(itertools.product(etch_width, etch_depth)))
params = [f'{i[0]}_{i[1]}.npy' for i in params]

get_disp = lambda n: np.load('sim_output/07-19-2022/dispersion-curves/' + params[n])

conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9  # already multiplied for some sim data (should be obvious)
for i in range(len(params)):
    x = get_disp(i)
    b2 = x[:, -1]
    if np.max(b2) > 100:
        print("uh oh")
        x[:, -1] /= conversion
        np.save(f'sim_output/07-19-2022/dispersion-curves/{params[i]}', x)
