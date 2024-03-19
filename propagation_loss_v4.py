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
dwidth_extra = 5
dheight_extra_air = 1.0
dheight_extra_sbstrt = 5

dpml_x = 3
dpml_y = 3
dpml_z = 3

# stretching
dstretch_buffer = 0.0
dx_source = 0.5
# assert dx_source < dstretch_buffer


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

    prism_before_stretch = mp.Prism(
        vertices=vertices,
        height=dstretch_buffer,
        sidewall_angle=0,
        material=copy.deepcopy(wvg.wvgd_mdm),
        axis=mp.Vector3(1, 0, 0),
    )  # pass material its own copy of wvgd_mdm

    prism_stretch = mp.Prism(
        vertices=vertices,
        height=length - dstretch_buffer,
        center=mp.Vector3(dstretch_buffer / 2, 0, 0),
        sidewall_angle=0,
        material=copy.deepcopy(wvg.wvgd_mdm),
        axis=mp.Vector3(1, 0, 0),
    )  # pass material its own copy of wvgd_mdm

    prism_before_stretch.center.z += dz
    prism_stretch.center.z += dz

    # blocks for the unetched waveguide + substrate
    blk_film_before_stretch = copy.deepcopy(wvg._blk_film)  # taper owns its own copy
    blk_film_before_stretch.size.x = dstretch_buffer
    blk_film_before_stretch.center.x = -length / 2 + dstretch_buffer / 2
    blk_sbstrt_before_stretch = copy.deepcopy(wvg.blk_sbstrt)  # taper owns its own copy
    blk_sbstrt_before_stretch.size.x = dstretch_buffer
    blk_sbstrt_before_stretch.center.x = -length / 2 + dstretch_buffer / 2

    blk_film_stretch = copy.deepcopy(wvg._blk_film)  # taper owns its own copy
    blk_film_stretch.size.x = length - dstretch_buffer
    blk_film_stretch.center.x = dstretch_buffer / 2
    blk_sbstrt_stretch = copy.deepcopy(wvg.blk_sbstrt)  # taper owns its own copy
    blk_sbstrt_stretch.size.x = length - dstretch_buffer
    blk_sbstrt_stretch.center.x = dstretch_buffer / 2

    blk_film_before_stretch.size.y = sy - 2 * dpml_y
    blk_film_stretch.size.y = sy - 2 * dpml_y
    blk_sbstrt_before_stretch.size.y = sy - 2 * dpml_y
    blk_sbstrt_stretch.size.y = sy - 2 * dpml_y

    air_block = mp.Block(
        size=mp.Vector3(
            length - dstretch_buffer,
            sy - dpml_y * 2,
            sz - dpml_z * 2,
        ),
        center=mp.Vector3(dstretch_buffer / 2),
        material=mp.Medium(epsilon_diag=mp.Vector3(1, 1, 1)),
    )

    geometry = [
        air_block,
        prism_before_stretch,
        blk_film_before_stretch,
        blk_sbstrt_before_stretch,
        prism_stretch,
        blk_film_stretch,
        blk_sbstrt_stretch,
    ]

    # --------------------
    # boundary layers list
    # absx = mp.Absorber(thickness=dpml_x, direction=mp.X)
    # absy = mp.Absorber(thickness=dpml_y, direction=mp.Y)
    # absz = mp.Absorber(thickness=dpml_z, direction=mp.Z, side=mp.Low)
    # pmlz = mp.PML(thickness=dpml_z, direction=mp.Z, side=mp.High)

    absx = mp.PML(thickness=dpml_x, direction=mp.X)
    absy = mp.PML(thickness=dpml_y, direction=mp.Y)
    absz = mp.PML(thickness=dpml_z, direction=mp.Z, side=mp.Low)
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


# %% --------------------------------------------------------------------------
fcen = 1 / 3.7
df = 0.1
nfreq = 1

resolution = (1 / fcen / mt.LiNbO3.epsilon(fcen).diagonal()[-1] / 8) ** -1

etch_angle = 80
etch_width = 1.245
etch_depth = 0.7
wvg_2D = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=1,
    substrate_medium=mp.Medium(epsilon=mtp.Al2O3.epsilon(fcen).diagonal()[-1]),
    waveguide_medium=mp.Medium(epsilon=mt.LiNbO3.epsilon(fcen).diagonal()[-1]),
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

length = 20
wvg_3D = convert_2Dto23D(
    wvg=wvg_2D,
    length=length,
    fcen=fcen,
    df=df,
    resolution=resolution,
    etch_angle=etch_angle,
)

# %% --------------------------------------------------------------------------
tr_x = 0.5
transform = mp.Matrix(diag=mp.Vector3(tr_x, 1, 1))
wvg_3D.geometry[0].material.transform(transform)  # air block
for i in wvg_3D.geometry[4:]:
    i.material.transform(transform)
courant = 0.5
wvg_3D.Courant = courant
print(f"stretching by {tr_x} and using courant factor of: {courant}")

# %% --------------------------------------------------------------------------
nonpml_vol = mp.Volume(
    center=mp.Vector3(),
    size=mp.Vector3(
        wvg_3D.cell_size.x - dpml_x * 2,
        wvg_3D.cell_size.y - dpml_y * 2,
        wvg_3D.cell_size.z - dpml_z * 2,
    ),
)

# record dft fields
dft_obj = wvg_3D.add_dft_fields([mp.Ey], fcen, df, nfreq, where=nonpml_vol)

# %% --------------------------------------------------------------------------
# size = mp.Vector3(0, wvg_2D.width, wvg_2D.height)
# pt_src_mon = -length / 2 + 1
# flux_start = wvg_3D.add_mode_monitor(
#     fcen,
#     df,
#     nfreq,
#     mp.FluxRegion(center=mp.Vector3(pt_src_mon), size=size),
# )
# flux_end = [
#     wvg_3D.add_mode_monitor(
#         fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(i), size=size)
#     )
#     for i in np.arange(pt_src_mon + 1, length / 2, 1)
# ]

# # %% --------------------------------------------------------------------------
# wvg_3D.use_output_directory("eps/")
# wvg_3D.run(
#     until_after_sources=[  # monitor starting from source every 1 um
#         mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(i, 0, 0), 1e-3)
#         for i in np.arange(-length / 2 + dx_source, length / 2, 1)
#     ],
#     # until_after_sources=500,
# )

# # %% --------------------------------------------------------------------------
# # save simulation data
# eps_data = wvg_3D.get_array(vol=nonpml_vol, component=mp.Dielectric)
# np.save(f"eps/eps_data_{1/fcen}_um_{tr_x}_x_stretch_{courant}_S.npy", eps_data)

# fname = f"eps/E_data_{1/fcen}_um_{tr_x}_x_stretch_{courant}_S"
# wvg_3D.output_dft(dft_obj, fname)

# # %% --------------------------------------------------------------------------
# # eig_coeff_start = wvg_3D.get_eigenmode_coefficients(
# #     flux_start,
# #     [1],
# #     eig_parity=mp.NO_PARITY,
# #     eig_tolerance=1e-16,
# # )
# # eig_coeff_end = [
# #     wvg_3D.get_eigenmode_coefficients(
# #         i, [1], eig_parity=mp.NO_PARITY, eig_tolerance=1e-16
# #     )
# #     for i in flux_end
# # ]
# # alpha_end = [np.squeeze(i.alpha) for i in eig_coeff_end]
# # alpha_end = np.asarray(alpha_end)
# # alpha_start = np.squeeze(eig_coeff_start.alpha)

# # freq = np.asarray(flux_start.freq)

# # np.save(f"eps/freq_{1/fcen}_um_{tr_x}_x_stretch.npy", freq)
# # np.save(f"eps/alpha_start_{1/fcen}_um_{tr_x}_x_stretch.npy", alpha_start)
# # np.save(f"eps/alpha_end_{1/fcen}_um_{tr_x}_x_stretch.npy", alpha_end)
