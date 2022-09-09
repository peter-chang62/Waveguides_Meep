"""sim data analysis the arrays were saved as np.c_[res.freq, beta, beta1, beta2] """

import numpy as np
import matplotlib.pyplot as plt
import os
from pynlo.media.crystals.XTAL_PPLN import Gayer5PctSellmeier
import clipboard_and_style_sheet
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy.constants as sc
import scipy.integrate as scint
from numpy import ma
import os

try:
    import materials as mtp

    on_linux = True
except:
    Al2O3 = np.load('convenience/freq_epsilon_data.npy')
    Al2O3 = InterpolatedUnivariateSpline(Al2O3[:, 0], Al2O3[:, 1])
    on_linux = False

clipboard_and_style_sheet.style_sheet()


def width(s):
    return float(s.split('_')[0])


def depth(s):
    return float(s.split('_')[1].split('.npy')[0])


def mode_area(I):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition that is used
    # reference: https://www.rp-photonics.com/effective_mode_area.html
    # this gives an overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


# %%____________________________________________________________________________________________________________________
resolution = 30
names = [i.name for i in os.scandir('sim_output/09-08-2022/dispersion-curves/')]
names = sorted(names, key=depth)
names = sorted(names, key=width)
get_disp = lambda n: np.load('sim_output/09-08-2022/dispersion-curves/' + names[n])
