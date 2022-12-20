"""cleaning up sim_output directories"""
import os
import numpy as np
import waveguide_dispersion as wg
import itertools

etch_width = wg.get_omega_axis(1 / 3, 1 / 0.3, 20)  # 300 nm to 3 um in 135 nm steps
etch_depth = np.arange(0.1, 1.05, .05)
etch_width = np.round(etch_width, 3)  # round the etch width
etch_depth = np.round(etch_depth, 3)  # round the etch depth
params = np.asarray(list(itertools.product(etch_width, etch_depth)))
params = [f'{i[0]}_{i[1]}.npy' for i in params]

src = "sim_output/07-19-2022/eps/"
names = [i.name for i in os.scandir(src)]
for i in names:
    if i in params:
        pass
    else:
        os.system(f'rm {src + i}')
        print(i)
