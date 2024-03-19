import copy
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import TFW_meep.materials as mtp
import TFW_meep.waveguide_dispersion as wg
from tqdm import tqdm
import tables as tb
import clipboard


rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


# The pml / absorbing layers in y and z only need to absorb what's left of
# evanescent waves, so shouldn't have to be as thick as the layers in x.
# Make sure they're not too close to the waveguide though, or else you'll
# suck power from the waveguide.
dwidth_extra = 8
dheight_extra = 8

dpml_x = 3
dpml_y = 3
dpml_z = 3
dx_source = 0


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
    # dimension parameters:
    original_cell_height = wvg.cell_height
    wvg.cell_height = (wvg.height / 2 + dheight_extra + dpml_z) * 2
    sz = wvg.cell_height
    sy = (wvg.width / 2 + dwidth_extra + dpml_y) * 2
    sx = dpml_x + dx_source + length + dx_source + dpml_x

    # --------------------
    # vertices of the waveguide
    y_added = wvg.height / np.tan(deg_to_rad(etch_angle))
    vertices = [
        mp.Vector3(-sx, wvg.width / 2 + y_added, -wvg.height / 2),
        mp.Vector3(-sx, -wvg.width / 2 - y_added, -wvg.height / 2),
        mp.Vector3(sx, -wvg.width / 2 - y_added, -wvg.height / 2),
        mp.Vector3(sx, wvg.width / 2 + y_added, -wvg.height / 2),
    ]

    # --------------------
    # cell + geometry list
    cell_size = mp.Vector3(sx, sy, sz)

    # prism that will be the waveguides + taper
    prism = mp.Prism(
        vertices=vertices,
        height=wvg.height,
        sidewall_angle=deg_to_rad(90 - etch_angle),  # takes angle in radians!
        material=copy.deepcopy(wvg.wvgd_mdm),
    )  # pass material its own copy of wvgd_mdm

    # blocks for the unetched waveguide + substrate
    blk_film = copy.deepcopy(wvg._blk_film)  # taper owns its own copy
    blk_sbstrt = copy.deepcopy(wvg.blk_sbstrt)  # taper owns its own copy
    geometry = [prism, blk_film, blk_sbstrt]

    # --------------------
    # boundary layers list
    absx = mp.Absorber(thickness=dpml_x, direction=mp.X)
    absy = mp.Absorber(thickness=dpml_y, direction=mp.Y)
    absz = mp.Absorber(thickness=dpml_z, direction=mp.Z, side=mp.Low)
    pmlz = mp.PML(thickness=dpml_z, direction=mp.Z, side=mp.High)
    boundary_layers = [
        absx,
        absy,
        absz,
        pmlz,
    ]

    # --------------------
    # sources list
    # EigenModeSource (emits from a 2D plane)
    sources = [
        mp.EigenModeSource(
            src=mp.GaussianSource(frequency=fcen, fwidth=df),
            center=mp.Vector3(-length / 2),
            size=mp.Vector3(
                0,
                sy - dpml_y * 2,
                sz - dpml_z * 2,
            ),
            eig_match_freq=True,
            eig_parity=mp.NO_PARITY,
            direction=mp.X,
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
    )

    # --------------------
    # reset
    wvg.cell_height = original_cell_height

    return sim


# %% --------------------------------------------------------------------------
# wl_ll, wl_ul = 2.9, 3.0
# nu_ll, nu_ul = 1 / wl_ul, 1 / wl_ll
# df = nu_ul - nu_ll
# fcen = (nu_ul - nu_ll) / 2 + nu_ll
# nfreq = 5

fcen = 1 / 1.55
df = 0.1
nfreq = 5

resolution = 10

etch_angle = 80
etch_width = 1.245
etch_depth = 0.7
wvg_2D = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=1,
    # substrate_medium=mtp.Al2O3,
    # waveguide_medium=mt.LiNbO3,
    substrate_medium=mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(fcen).diagonal()),
    waveguide_medium=mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(fcen).diagonal()),
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

length = 50
wvg_3D = convert_2Dto23D(
    wvg=wvg_2D,
    length=length,
    fcen=fcen,
    df=df,
    resolution=resolution,
    etch_angle=etch_angle,
)

nonpml_vol = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(
        wvg_3D.cell_size.x - dpml_x * 2,
        wvg_3D.cell_size.y - dpml_y * 2,
        wvg_3D.cell_size.z - dpml_z * 2,
    ),
)

# %% --------------------------------------------------------------------------
# record dft fields
dft_obj = wvg_3D.add_dft_fields([mp.Ey], fcen, df, nfreq, where=nonpml_vol)

# %% --------------------------------------------------------------------------
size = mp.Vector3(0, wvg_2D.width, wvg_2D.height)
pt_src_mon = -length / 2 + 1
flux_start = wvg_3D.add_mode_monitor(
    fcen,
    df,
    nfreq,
    mp.FluxRegion(center=mp.Vector3(pt_src_mon), size=size),
)
flux_end = [
    wvg_3D.add_mode_monitor(
        fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(i), size=size)
    )
    for i in np.arange(pt_src_mon + 1, length / 2, 1)
]

# %% --------------------------------------------------------------------------
wvg_3D.use_output_directory("eps/")
wvg_3D.run(
    until_after_sources=[  # until fields have decayed at both monitor points
        mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(length / 2), 1e-3),
        mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(-length / 2 + 1), 1e-3),
    ],
)
eps_data = wvg_3D.get_array(vol=nonpml_vol, component=mp.Dielectric)

# %% --------------------------------------------------------------------------
eig_coeff_start = wvg_3D.get_eigenmode_coefficients(
    flux_start,
    [1],
    eig_parity=mp.NO_PARITY,
    eig_tolerance=1e-16,
)
eig_coeff_end = [
    wvg_3D.get_eigenmode_coefficients(
        i, [1], eig_parity=mp.NO_PARITY, eig_tolerance=1e-16
    )
    for i in flux_end
]
alpha_end = [np.squeeze(i.alpha) for i in eig_coeff_end]
alpha_end = np.asarray(alpha_end)
alpha_start = np.squeeze(eig_coeff_start.alpha)

freq = np.asarray(flux_start.freq)

# %% --------------------------------------------------------------------------
# takes a lot of RAM, maybe split into a for loop and save to many files...
# or use pytables
file = tb.open_file(f"eps/Iy_data_{1/fcen}_um.h5", mode="w")
atom = tb.Float64Atom()
array = file.create_earray(
    file.root,
    "Iy_data",
    atom=atom,
    shape=(0, *eps_data.shape),
)

for n in tqdm(range(nfreq)):
    ey_data = wvg_3D.get_dft_array(dft_obj, mp.Ey, n)
    Iy_data = abs(ey_data) ** 2
    array.append(Iy_data[np.newaxis])

file.close()

# %% --------------------------------------------------------------------------
np.save(f"eps/freq_{1/fcen}_um.npy", freq)
np.save(f"eps/alpha_start_{1/fcen}_um.npy", alpha_start)
np.save(f"eps/alpha_end_{1/fcen}_um.npy", alpha_end)
np.save(f"eps/eps_data_{1/fcen}_um.npy", eps_data)
