"""
creating a taper. I think I can make this similar to Pooja's Lumerical
simulations.
"""

# %% ----- package imports
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from meep import materials as mt
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
import scipy.constants as sc
from TFW_meep import geometry
from scipy.interpolate import UnivariateSpline
import copy
from scipy.constants import c
import clipboard
from tqdm import tqdm


# %% ----- Gayer paper Sellmeier equation for ne (taken from PyNLO 1 / omega is in um
# n = squrt[epsilon] so epsilon = n^2
def eps_func_wvgd(omega):
    # omega is in inverse micron
    um = 1e-6
    v = omega * sc.c / um
    return n_MgLN_G(v, T=24.5, axis="e") ** 2


def n_MgLN_G(v, T=24.5, axis="e"):
    """
    Range of Validity:
        - 500 nm to 4000 nm
        - 20 C to 200 C
        - 48.5 mol % Li
        - 5 mol % Mg

    Gayer, O., Sacks, Z., Galun, E. et al. Temperature and wavelength
    dependent refractive index equations for MgO-doped congruent and
    stoichiometric LiNbO3 . Appl. Phys. B 91, 343–348 (2008).

    https://doi.org/10.1007/s00340-008-2998-2

    """
    if axis == "e":
        a1 = 5.756  # plasmons in the far UV
        a2 = 0.0983  # weight of UV pole
        a3 = 0.2020  # pole in UV
        a4 = 189.32  # weight of IR pole
        a5 = 12.52  # pole in IR
        a6 = 1.32e-2  # phonon absorption in IR
        b1 = 2.860e-6
        b2 = 4.700e-8
        b3 = 6.113e-8
        b4 = 1.516e-4
    elif axis == "o":
        a1 = 5.653  # plasmons in the far UV
        a2 = 0.1185  # weight of UV pole
        a3 = 0.2091  # pole in UV
        a4 = 89.61  # weight of IR pole
        a5 = 10.85  # pole in IR
        a6 = 1.97e-2  # phonon absorption in IR
        b1 = 7.941e-7
        b2 = 3.134e-8
        b3 = -4.641e-9
        b4 = -2.188e-6

    else:
        raise ValueError("axis needs to be o or e")

    wvl = sc.c / v * 1e6  # um
    f = (T - 24.5) * (T + 570.82)
    n2 = (
        (a1 + b1 * f)
        + (a2 + b2 * f) / (wvl**2 - (a3 + b3 * f) ** 2)
        + (a4 + b4 * f) / (wvl**2 - a5**2)
        - a6 * wvl**2
    )
    return n2**0.5


# %% --------------------------------------------------------------------------
etch_width = 2.0
etch_depth = 0.350
LiN_thickness = 0.350
resolution = 20

sim1 = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=LiN_thickness,
    substrate_medium=mt.SiO2,
    # substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    z_offset_wvgd=1.5,
    num_bands=1,
)

etch_width = 0.2
sim2 = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=LiN_thickness,
    substrate_medium=mt.SiO2,
    # substrate_medium=mtp.Al2O3,
    waveguide_medium=mt.LiNbO3,
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    z_offset_wvgd=1.5,
    num_bands=1,
)

# %% --------------------------------------------------------------------------
# blk_wvgd.material = mp.Medium(epsilon=eps_func_wvgd(fcen))
# blk_film.material = mp.Medium(epsilon=eps_func_wvgd(fcen))
# blk_sbstrt.material = mp.Medium(epsilon=sim.sbstrt_mdm.epsilon(fcen).diagonal()[-1])

substrate = sim1.blk_sbstrt
film = sim1._blk_film
