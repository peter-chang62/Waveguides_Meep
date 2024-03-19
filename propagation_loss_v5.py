"""
Combine MEEP simulations and PyNLO into one script (will only run on a linux
computer). This is good for sort of point sampling of the big simulation
outputs. You can make small adjustment to parameters here
"""

# %% package imports
import numpy as np
import matplotlib.pyplot as plt
import clipboard
import meep.materials as mt
from TFW_meep import materials as mtp
from TFW_meep import waveguide_dispersion as wg
import scipy.constants as sc
import meep as mp
import copy

dwidth_extra = 8
dheight_extra_air = 3
dheight_extra_sbstrt = 8

dpml_x = 3
dpml_y = 3
dpml_z = 3

dx_source = 0.5
dyz_tr_buffer = 1.5
dx_tr_buffer = 3

# assert dx_tr_buffer > dx_source


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


# %% ----- creating 3D sim from 2D --------------------------------------------
rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def convert_2Dto23D(
    wvg,
    length,
    fcen,
    df,
    resolution,
    etch_angle=80,
):
    assert isinstance(
        wvg, wg.ThinFilmWaveguide
    ), f"wvg must be instance of ThinFilmWaveguide or it's child class but got {type(wvg)}"
    wvg: wg.ThinFilmWaveguide

    # --------------------
    # cell size
    original_cell_height = wvg.cell_height

    wvg.cell_height = (wvg.height / 2 + dheight_extra_sbstrt) * 2
    sz = wvg.height + dheight_extra_sbstrt + dheight_extra_air + dpml_z * 2
    sy = (wvg.width / 2 + dwidth_extra + dpml_y) * 2
    sx = dpml_x + length + dpml_x

    dz = (dheight_extra_sbstrt - dheight_extra_air) / 2
    wvg._blk_film.center.z += dz
    wvg.blk_wvgd.center.z += dz
    wvg.blk_sbstrt.center.z += dz

    cell_size = mp.Vector3(sx, sy, sz)

    # --------------------
    # geometry list

    # vertices of the waveguide
    y_added = wvg.height / np.tan(deg_to_rad(etch_angle))
    vertices = [
        mp.Vector3(-length / 2, wvg.width / 2 + y_added, -wvg.height / 2),
        mp.Vector3(-length / 2, -wvg.width / 2 - y_added, -wvg.height / 2),
        mp.Vector3(-length / 2, -wvg.width / 2, wvg.height / 2),
        mp.Vector3(-length / 2, wvg.width / 2, wvg.height / 2),
    ]

    # unstretched prism
    prism = mp.Prism(
        vertices=vertices,
        height=length,
        sidewall_angle=0,
        material=copy.deepcopy(wvg.wvgd_mdm),
        axis=mp.Vector3(1, 0, 0),
    )  # pass material its own copy of wvgd_mdm

    # stretched prism
    prism_stretch = mp.Prism(
        vertices=vertices,
        height=length - dx_tr_buffer * 2,
        center=mp.Vector3(0, 0, 0),
        sidewall_angle=0,
        material=copy.deepcopy(wvg.wvgd_mdm),
        axis=mp.Vector3(1, 0, 0),
    )  # pass material its own copy of wvgd_mdm

    prism.center.z += dz
    prism_stretch.center.z += dz

    # unstretched objects
    blk_film = copy.deepcopy(wvg._blk_film)  # taper owns its own copy
    blk_sbstrt = copy.deepcopy(wvg.blk_sbstrt)  # taper owns its own copy

    blk_film.size.x = length
    blk_sbstrt.size.x = length

    blk_film.size.y = sy - 2 * dpml_y
    blk_sbstrt.size.y = sy - 2 * dpml_y

    # stretched objects
    blk_film_stretch = copy.deepcopy(wvg._blk_film)  # taper owns its own copy
    blk_sbstrt_stretch = copy.deepcopy(wvg.blk_sbstrt)  # taper owns its own copy

    blk_film_stretch.size.x = length - dx_tr_buffer * 2
    blk_sbstrt_stretch.size.x = length - dx_tr_buffer * 2

    blk_film_stretch.size.y = sy - 2 * dpml_y - dyz_tr_buffer * 2
    blk_sbstrt_stretch.size.y = sy - 2 * dpml_y - dyz_tr_buffer * 2

    blk_sbstrt_stretch.size.z -= dyz_tr_buffer
    blk_sbstrt_stretch.center.z += dyz_tr_buffer / 2

    # air block
    air_block = mp.Block(
        size=mp.Vector3(
            length - dx_tr_buffer * 2,
            sy - dpml_y * 2 - dyz_tr_buffer * 2,
            sz / 2 - dpml_z - dyz_tr_buffer,
        ),
        center=mp.Vector3(0, 0, (sz / 2 - dpml_z - dyz_tr_buffer) / 2),
        material=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 1)),
    )

    geometry = [  # unstretched followed by stretched objects
        prism,
        blk_film,
        blk_sbstrt,
        air_block,
        prism_stretch,
        blk_film_stretch,
        blk_sbstrt_stretch,
    ]

    # --------------------
    pmlx = mp.PML(thickness=dpml_x, direction=mp.X)
    pmly = mp.PML(thickness=dpml_y, direction=mp.Y)
    pmlz = mp.PML(thickness=dpml_z, direction=mp.Z)

    boundary_layers = [
        # pmlx,
        pmly,
        pmlz,
    ]

    # --------------------
    # sources list
    # EigenModeSource (emits from a 2D plane)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            center=mp.Vector3(-length / 2 + dx_source, 0, 0),
            size=mp.Vector3(
                0,
                sy - dpml_y * 2,
                sz - dpml_z * 2,
            ),
            eig_match_freq=True,
            eig_parity=mp.ODD_Y,
        )
    ]

    # --------------------
    # create the simulation instance
    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=boundary_layers,
        sources=sources,
        resolution=resolution,
        symmetries=[mp.Mirror(mp.Y, phase=-1)],
    )

    # --------------------
    # reset the cell height after the centers! (reverse order of how they were
    # changed)
    wvg._blk_film.center.z -= dz
    wvg.blk_wvgd.center.z -= dz
    wvg.blk_sbstrt.center.z -= dz
    wvg.cell_height = original_cell_height
    return sim


def plot_slices(wvg):
    wvg: mp.Simulation
    x = wvg.cell_size.x
    y = wvg.cell_size.y
    z = wvg.cell_size.z

    wvg.cell_size.x = 0
    plt.figure()
    wvg.plot2D()
    wvg.cell_size.x = x

    wvg.cell_size.y = 0
    plt.figure()
    wvg.plot2D()
    wvg.cell_size.y = y

    wvg.cell_size.z = 0
    plt.figure()
    wvg.plot2D()
    wvg.cell_size.z = z


# %% ----- create sim instance ------------------------------------------------
resolution = 20
etch_width = 1.245
etch_depth = 0.7
film_thickness = 1

fcen = 1 / 4.0
df = 0.1

sim = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=film_thickness,
    substrate_medium=mp.Medium(epsilon=mtp.Al2O3.epsilon(fcen).diagonal()[-1]),
    waveguide_medium=mp.Medium(epsilon=mt.LiNbO3.epsilon(fcen).diagonal()[-1]),
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

resolution = (1 / fcen / mt.LiNbO3.epsilon(fcen).diagonal()[-1] / 10) ** -1
# length = 25 + 2 * dx_tr_buffer
length = 1

etch_angle = 80
wvg_3D = convert_2Dto23D(
    wvg=sim,
    length=length,
    fcen=fcen,
    df=df,
    resolution=resolution,
    etch_angle=etch_angle,
)

# %% ----- transform ----------------------------------------------------------
tr_x = 1 / 1
m_tr = mp.Matrix(diag=mp.Vector3(tr_x, 1, 1))
for i in wvg_3D.geometry[3:]:
    i.material.transform(m_tr)
courant = 0.5
wvg_3D.Courant = courant
print(f"stretching by {tr_x} and using courant factor of: {courant}")

# %% ----- monitor dft --------------------------------------------------------
nonpml_vol = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(
        wvg_3D.cell_size.x - dpml_x * 2,
        wvg_3D.cell_size.y - dpml_y * 2,
        wvg_3D.cell_size.z - dpml_z * 2,
    ),
)

# record dft fields
dft_obj = wvg_3D.add_dft_fields([mp.Ey], fcen, df, 1, where=nonpml_vol)

# %% ----- run ----------------------------------------------------------------
wvg_3D.run(
    # until_after_sources=mp.stop_when_energy_decayed(dt=50, decay_by=1e-3),
    # until_after_sources=mp.stop_when_dft_decayed(
    #     tol=1e-3, minimum_run_time=0, maximum_run_time=None
    # ),
    until_after_sources=1000,  # 1 us?
)

# %% ----- save simulation data -----------------------------------------------
eps_data = wvg_3D.get_array(vol=nonpml_vol, component=mp.Dielectric)
np.save(
    f"eps/eps_data_{np.round(1/fcen, 3)}_um_{np.round(tr_x, 3)}_x_stretch.npy", eps_data
)

fname = f"eps/E_data_{np.round(1/fcen, 3)}_um_{np.round(tr_x, 3)}_x_stretch"
wvg_3D.output_dft(dft_obj, fname)
