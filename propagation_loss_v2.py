import copy
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import TFW_meep.materials as mtp
import clipboard
import h5py
import TFW_meep.waveguide_dispersion as wg
from tqdm import tqdm

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


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
    nfreq,
    resolution,
    etch_angle=80,
):
    assert isinstance(
        wvg, wg.ThinFilmWaveguide
    ), f"wvg must be instance of ThinFilmWaveguide or it's child class but got {type(wvg)}"
    wvg: wg.ThinFilmWaveguide

    # --------------------
    # dimension parameters:
    # The pml / absorbing layers in y and z only need to absorb what's left of
    # evanescent waves, so shouldn't have to be as thick as the layers in x.
    # Make sure they're not too close to the waveguide though, or else you'll
    # suck power from the waveguide.
    dwidth_extra = 5
    dheight_extra = 5

    dpml_x = 3
    dpml_y = 3
    dpml_z = 3

    original_cell_height = wvg.cell_height
    wvg.cell_height = (wvg.height / 2 + dheight_extra + dpml_z) * 2
    sz = wvg.cell_height
    sy = (wvg.width / 2 + dwidth_extra + dpml_y) * 2
    sx = dpml_x + 2 + length + 2 + dpml_x

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
    # add the flux monitor
    # add_mode_monitor for no parity requirement, otherwise use add_flux
    # size = mp.Vector3(
    #     0,
    #     sy - 2 * dpml_y,
    #     sz - 2 * dpml_z,
    # )

    size = mp.Vector3(
        0,
        wvg.width,
        wvg.height,
    )

    flux_start = sim.add_mode_monitor(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(-length / 2 + 1), size=size),
    )

    flux_end = sim.add_mode_monitor(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(length / 2), size=size),
    )

    # --------------------
    # reset
    wvg.cell_height = original_cell_height

    return sim, flux_start, flux_end


# %% --------------------------------------------------------------------------
wl_ll, wl_ul = 2.95, 3.05
nu_ll, nu_ul = 1 / wl_ul, 1 / wl_ll
df = nu_ul - nu_ll
fcen = (nu_ul - nu_ll) / 2 + nu_ll

resolution = 10
etch_angle = 80

etch_width = 1.245
etch_depth = 0.7
wvg_2D = wg.ThinFilmWaveguide(
    etch_width=etch_width,
    etch_depth=etch_depth,
    film_thickness=1,  # I'll fix the height at 1 um now
    # substrate_medium=mtp.Al2O3,
    # waveguide_medium=mt.LiNbO3,
    substrate_medium=mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(fcen).diagonal()),
    waveguide_medium=mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(fcen).diagonal()),
    resolution=resolution,
    cell_width=10,
    cell_height=10,
    num_bands=1,
)

length = 10
wvg_3D, flux_start, flux_end = convert_2Dto23D(
    wvg=wvg_2D,
    length=length,
    fcen=fcen,
    df=df,
    nfreq=100,
    resolution=resolution,
    etch_angle=etch_angle,
)

# plot_slices causes some issues, so save one for visualization
wvg_3D_plot, _, _ = convert_2Dto23D(
    wvg=wvg_2D,
    length=length,
    fcen=fcen,
    df=df,
    nfreq=100,
    resolution=resolution,
    etch_angle=etch_angle,
)

# %% --------------------------------------------------------------------------
wvg_3D.use_output_directory("eps/")
wvg_3D.run(
    mp.to_appended("ey", mp.at_every(1, mp.output_efield_y)),
    mp.at_beginning(mp.output_epsilon(frequency=fcen)),
    until_after_sources=[  # until fields have decayed at both monitor points
        mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(length / 2), 1e-3),
        mp.stop_when_fields_decayed(50, mp.Ey, mp.Vector3(-length / 2 + 1), 1e-3),
    ],
)

coeff_start = wvg_3D.get_eigenmode_coefficients(
    flux_start,
    [1],
    eig_parity=mp.NO_PARITY,
    eig_tolerance=1e-16,
)
coeff_end = wvg_3D.get_eigenmode_coefficients(
    flux_end,
    [1],
    eig_parity=mp.NO_PARITY,
    eig_tolerance=1e-16,
)
start = np.c_[np.array(flux_start.freq), np.squeeze(coeff_start.alpha)]
end = np.c_[np.array(flux_end.freq), np.squeeze(coeff_end.alpha)]
freq = start[:, 0]
trans = abs(end[:, 1] / start[:, 1]) ** 2

# %% --------------------------------------------------------------------------
ey = h5py.File("eps/propagation_loss_v2-ey.h5", "r")
ey = ey["ey"]

Iy_tavg = 0
for n in tqdm(range(ey.shape[-1])):
    Iy = abs(ey[:, :, :, n]) ** 2
    Iy_tavg = (Iy_tavg * n + Iy) / (n + 1)
np.save("eps/Iy_tavg.npy", Iy_tavg)

# %% --------------------------------------------------------------------------
sim_2D = wvg_2D.calc_dispersion(freq_array=freq)  # simulate

# %% ----- plotting -----------------------------------------------------------
plt.figure()
plt.plot(1 / freq, abs(start[:, 1]) ** 2)
plt.plot(1 / freq, abs(end[:, 1]) ** 2)

plt.figure()
plt.plot(1 / freq, trans)

plot_slices(wvg_3D_plot)
