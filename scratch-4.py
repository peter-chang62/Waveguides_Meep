import copy
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import clipboard_and_style_sheet
import h5py
import waveguide_dispersion as wg

rad_to_deg = lambda rad: rad * 180 / np.pi
deg_to_rad = lambda deg: deg * np.pi / 180


def play_video(shape='linear', eps=None, save=False):
    assert any([shape == 'linear', shape == 'straight'])

    if shape == 'linear':
        with h5py.File('sim_output/06-19-2022/linear/scratch-4-ez.h5', 'r') as f:
            data = f.get('ez')
            fig, ax = plt.subplots(1, 1)
            for n in range(data.shape[2]):
                ax.clear()
                if eps is not None:
                    ax.imshow(eps[::-1, ::-1].T, cmap='binary')
                    alpha = 0.9
                else:
                    alpha = 1
                ax.imshow(data[:, :, n][::-1, ::-1].T, cmap='RdBu', alpha=alpha)
                plt.axis(False)
                if save:
                    plt.savefig(f'fig/{n}.png')
                else:
                    plt.pause(.01)

    if shape == 'straight':
        with h5py.File('sim_output/06-19-2022/straight/scratch-4-ez.h5', 'r') as f:
            data = f.get('ez')
            fig, ax = plt.subplots(1, 1)
            for n in range(data.shape[2]):
                ax.clear()
                if eps is not None:
                    ax.imshow(eps[::-1, ::-1].T, cmap='binary')
                    alpha = 0.9
                else:
                    alpha = 1
                ax.imshow(data[:, :, n][::-1, ::-1].T, cmap='RdBu', alpha=alpha)
                plt.axis(False)
                if save:
                    plt.savefig(f'fig/{n}.png')
                else:
                    plt.pause(.01)


def create_taper_sim(wvg1, wvg2, length_taper, fcen, df, nfreq, resolution, etch_angle=80):
    assert all([isinstance(wvg1, wg.ThinFilmWaveguide),
                isinstance(wvg2, wg.ThinFilmWaveguide)]), \
        f"wvg1 and wvg2 must be instances of ThinFilmWaveguide or " \
        f"it's child class but got {type(wvg1)} and {type(wvg2)}"
    wvg1: wg.ThinFilmWaveguide
    wvg2: wg.ThinFilmWaveguide

    assert wvg1.height == wvg2.height, \
        f"the heights of wvg1 and wvg2 must be the same but got {wvg1.height} and {wvg2.height}"

    assert wvg1.width <= wvg2.width, \
        f"the width of wvg1 must be less than or equal to that of wvg2, but got {wvg1.width} for wvg1 and " \
        f"{wvg2.width} for wvg2"

    # __________________________________________________________________________________________________________________
    # dimension parameters
    dair = 3.0
    dsubstrate = 3

    dpml_x = 3
    dpml_y = 3
    dpml_z = 2

    wvg1.cell_height += dpml_z * 2  # modify cell height
    sz = wvg1.cell_height
    sy = (wvg2.width / 2 + dsubstrate + dpml_y) * 2
    sx = dpml_x + 2 + length_taper + 2 + dpml_x

    # __________________________________________________________________________________________________________________
    # vertices of the waveguides + taper
    y_added = wvg1.height / np.tan(deg_to_rad(etch_angle))
    if wvg1.width < wvg2.width:
        vertices = [
            mp.Vector3(-sx, wvg1.width / 2 + y_added, -wvg1.height / 2),
            mp.Vector3(-sx, -wvg1.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(-length_taper / 2, -wvg1.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(length_taper / 2, -wvg2.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(sx, -wvg2.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(sx, wvg2.width / 2 + y_added, -wvg1.height / 2),
            mp.Vector3(length_taper / 2, wvg2.width / 2 + y_added, -wvg1.height / 2),
            mp.Vector3(-length_taper / 2, wvg1.width / 2 + y_added, -wvg1.height / 2),
        ]
    else:
        vertices = [
            mp.Vector3(-sx, wvg1.width / 2 + y_added, -wvg1.height / 2),
            mp.Vector3(-sx, -wvg1.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(sx, -wvg2.width / 2 - y_added, -wvg1.height / 2),
            mp.Vector3(sx, wvg2.width / 2 + y_added, -wvg1.height / 2),
        ]

    # __________________________________________________________________________________________________________________
    # cell + geometry list
    cell_size = mp.Vector3(sx, sy, sz)

    # prism that will be the waveguides + taper
    prism = mp.Prism(vertices=vertices,
                     height=wvg1.height,
                     sidewall_angle=deg_to_rad(90 - etch_angle),  # takes angle in radians!
                     material=copy.deepcopy(wvg1.wvgd_mdm))  # pass material its own copy of wvgd_mdm

    # blocks for the unetched waveguide + substrate
    blk_film = copy.deepcopy(wvg1._blk_film)  # taper owns its own copy
    blk_sbstrt = copy.deepcopy(wvg1.blk_sbstrt)  # taper owns its own copy
    geometry = [prism, blk_film, blk_sbstrt]

    # __________________________________________________________________________________________________________________
    # boundary layers list
    absx = mp.Absorber(thickness=dpml_x, direction=mp.X)
    absy = mp.Absorber(thickness=dpml_y, direction=mp.Y)
    absz = mp.Absorber(thickness=dpml_z, direction=mp.Z, side=mp.Low)
    pmlz = mp.PML(thickness=dpml_z, direction=mp.Z, side=mp.High)
    boundary_layers = [absx, absy, absz, pmlz]

    # __________________________________________________________________________________________________________________
    # sources list
    # EigenModeSource (emits from a 2D plane)
    sources = [mp.EigenModeSource(src=mp.GaussianSource(frequency=fcen, fwidth=df),
                                  center=mp.Vector3(-length_taper / 2 - 1),
                                  size=mp.Vector3(0, sy - dpml_y * 2, sz - dpml_z * 2),
                                  eig_match_freq=True,
                                  eig_parity=mp.NO_PARITY)]

    # __________________________________________________________________________________________________________________
    # create the simulation instance
    sim = mp.Simulation(cell_size=cell_size,
                        geometry=geometry,
                        boundary_layers=boundary_layers,
                        sources=sources,
                        resolution=resolution)

    # __________________________________________________________________________________________________________________
    # add the flux monitor
    mon_pt = mp.Vector3(length_taper / 2 + 1)
    # add_mode_monitor for no parity requirement, otherwise use add_flux
    flux = sim.add_mode_monitor(fcen,
                                df,
                                nfreq,
                                mp.FluxRegion(center=mon_pt,
                                              size=mp.Vector3(0, sy - 2 * dpml_y, sz - 2 * dpml_z)))

    # __________________________________________________________________________________________________________________
    # reset
    wvg1.cell_height -= dpml_z * 2

    return sim, flux, mon_pt


# %% ___________________________________________________________________________________________________________________
wvg1 = wg.ThinFilmWaveguide(etch_width=1,
                            etch_depth=.3,
                            film_thickness=.7,
                            substrate_medium=mtp.Al2O3,
                            waveguide_medium=mt.LiNbO3,
                            # substrate_medium=mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(1 / 1.55).diagonal()),
                            # waveguide_medium=mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(1 / 1.55).diagonal()),
                            resolution=30, num_bands=1,
                            cell_width=10,
                            cell_height=4)

wvg2 = wg.ThinFilmWaveguide(etch_width=3,
                            etch_depth=.3,
                            film_thickness=.7,
                            substrate_medium=mtp.Al2O3,
                            waveguide_medium=mt.LiNbO3,
                            # substrate_medium=mp.Medium(epsilon_diag=mtp.Al2O3.epsilon(1 / 1.55).diagonal()),
                            # waveguide_medium=mp.Medium(epsilon_diag=mt.LiNbO3.epsilon(1 / 1.55).diagonal()),
                            resolution=30, num_bands=1,
                            cell_width=10,
                            cell_height=4)
# %% ___________________________________________________________________________________________________________________
ll, ul = 0.8, 2
ll, ul = 1 / ul, 1 / ll
df = ul - ll
fcen = (ul - ll) / 2 + ll

lt = 1
nfreq = 100
resolution = 40
etch_angle = 80

taper_linear, flux_linear, mon_pt_linear = create_taper_sim(wvg1=wvg1,
                                                            wvg2=wvg2,
                                                            length_taper=lt,
                                                            fcen=fcen,
                                                            df=df,
                                                            nfreq=nfreq,
                                                            resolution=resolution,
                                                            etch_angle=etch_angle)

taper_straight, flux_straight, mon_pt_straight = create_taper_sim(wvg1=wvg1,
                                                                  wvg2=wvg1,
                                                                  length_taper=lt,
                                                                  fcen=fcen,
                                                                  df=df,
                                                                  nfreq=nfreq,
                                                                  resolution=resolution,
                                                                  etch_angle=etch_angle)

# %% ___________________________________________________________________________________________________________________
# 2D simulation
taper_linear.cell_size.z = 0
taper_linear.geometry = [taper_linear.geometry[0]]
taper_linear.boundary_layers = taper_linear.boundary_layers[:-2]
taper_linear.use_output_directory('sim_output/06-19-2022/linear')
taper_linear.run(
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_linear, 1e-3))
res_linear = taper_linear.get_eigenmode_coefficients(flux_linear, [1], eig_parity=mp.NO_PARITY)
arr_linear = np.c_[np.array(flux_linear.freq), np.squeeze(res_linear.alpha)]
np.save("alpha_linear.npy", arr_linear)

taper_straight.cell_size.z = 0
taper_straight.geometry = [taper_straight.geometry[0]]
taper_straight.boundary_layers = taper_straight.boundary_layers[:-2]
taper_straight.use_output_directory('sim_output/06-19-2022/straight')
taper_straight.run(
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_straight, 1e-3))
res_straight = taper_straight.get_eigenmode_coefficients(flux_straight, [1], eig_parity=mp.NO_PARITY)
arr_straight = np.c_[np.array(flux_straight.freq), np.squeeze(res_straight.alpha)]
np.save("alpha_straight.npy", arr_straight)

# plotting
arr_linear = np.load('gif/06-19-2022/alpha_linear_1um_taper.npy')
arr_straight = np.load('gif/06-19-2022/alpha_straight_1um_taper.npy')
fig, ax = plt.subplots(1, 1)
ax.plot(1 / arr_linear[:, 0], arr_linear[:, 1].__abs__() ** 2 / arr_straight[:, 1].__abs__() ** 2,
        label='transmission')
ax2 = ax.twinx()
phase = np.unwrap(np.arctan2(arr_linear[:, 1].imag, arr_linear[:, 1].real))
p = np.polyfit(arr_linear[:, 0], phase, deg=1)
z = np.poly1d(p)
phase -= z(arr_linear[:, 0]).real
ax2.plot(1 / arr_linear[:, 0], rad_to_deg(phase), 'C1',
         label='phase')
plt.xlabel("wavelength ($\mathrm{\\mu m}$)")
ax.set_ylabel("transmission")
ax2.set_ylabel("phase (deg)")
ax.legend(loc='best')
ax2.legend(loc='best')

# %% ___________________________________________________________________________________________________________________
# 3D simulation
# memory requirement is already too much!
# taper_linear.use_output_directory('sim_output/06-19-2022')
# taper_linear.run(
#     # mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
#     until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_linear, 1e-3))
# res_linear = taper_linear.get_eigenmode_coefficients(flux_linear, [1], eig_parity=mp.NO_PARITY)
# arr_linear = np.c_[np.array(flux_linear.freq), np.squeeze(res_linear.alpha)]
#
# taper_straight.use_output_directory('sim_output/06-19-2022')
# taper_straight.run(
#     # mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
#     until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mon_pt_straight, 1e-3))
# res_straight = taper_straight.get_eigenmode_coefficients(flux_straight, [1], eig_parity=mp.NO_PARITY)
# arr_straight = np.c_[np.array(flux_straight.freq), np.squeeze(res_straight.alpha)]
