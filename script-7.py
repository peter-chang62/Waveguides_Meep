"""
First attempt to run waveguide dispersion parameters through Connor's PyNLO class
"""

# Imports
import numpy as np
from scipy.constants import pi, c
from matplotlib import pyplot as plt

from PyNLO_Connor import pynlo
from PyNLO_Connor.pynlo import utility as utils

# %% Pulse Properties

# Pulse Properties _____________________________________________________________________________________________________
n_points = 2 ** 13
v_min = c / 3500e-9  # c / 3500 nm
v_max = c / 450e-9  # c / 900 nm
v0 = c / 1550e-9  # c / 1550 nm
e_p = 1e-9  # 1 nJ
t_fwhm = 50e-15  # 50 fs

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
pulse.rtf_grids(n_harmonic=2, update=True)  # anti-aliasing
