import numpy as np
import matplotlib.pyplot as plt
import clipboard
import os


def load(h):
    # path = "sim_output/02-23-2024/Al2O3/"
    path = "sim_output/02-23-2024/SiO2/"
    names = [i.name for i in os.scandir(path)]
    names = [i for i in names if str(float(h)) + "_t" in i]
    if len(names) == 0:
        print("no file found")
        return
    else:
        assert len(names) == 1
        (f,) = names
        return np.load(path + f)


ps = 1e-12
km = 1e3
width = np.arange(0.5, 2.1, 0.1)

x = load(0.4)
# etch depth, width: (wavelength, effective area, beta2)
