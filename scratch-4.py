import copy
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import meep.materials as mt
import materials as mtp
import clipboard_and_style_sheet
import h5py
import waveguide_dispersion as wg


def create_taper_sim(wvg1, wvg2, length_taper, fcen, df, nfreq, resolution, etch_angle=80):
    assert all([isinstance(wvg1, wg.ThinFilmWaveguide),
                isinstance(wvg2, wg.ThinFilmWaveguide)]), \
        f"wvg1 and wvg2 must be instances of ThinFilmWaveguide or " \
        f"it's child class but got {type(wvg1)} and {type(wvg2)}"
    wvg1: wg.ThinFilmWaveguide
    wvg2: wg.ThinFilmWaveguide

    assert wvg1.height == wvg2.height, \
        f"the heights of wvg1 and wvg2 must be the same but got {wvg1.height} and {wvg2.height}"

    assert wvg1.width < wvg2.width, \
        f"the width of wvg1 must be less than that of wvg2, but got {wvg1.width} for wvg1 and " \
        f"{wvg2.width} for wvg2"

    # __________________________________________________________________________________________________________________
    # dimension parameters
    dair = 3.0
    dsubstrate = 3

    dpml_x = 3
    dpml_y = 3
    dpml_z = 2

    wvg1.cell_height += dpml_z * 2
    wvg2.cell_height += dpml_z * 2
    sz = wvg1.cell_height
    sy = (wvg2.width / 2 + dsubstrate + dpml_y) * 2
    sx = dpml_x + 2 + length_taper + 2 + dpml_x

    # __________________________________________________________________________________________________________________
    # vertices of the waveguides + taper
    vertices = [
        mp.Vector3(-sx, wvg1.width / 2, -wvg1.height / 2),
        mp.Vector3(-sx, -wvg1.width / 2, -wvg1.height / 2),
        mp.Vector3(-length_taper / 2, -wvg1.width / 2, -wvg1.height / 2),
        mp.Vector3(length_taper / 2, -wvg2.width / 2, -wvg1.height / 2),
        mp.Vector3(sx, -wvg2.width / 2, -wvg1.height / 2),
        mp.Vector3(sx, wvg2.width / 2, -wvg1.height / 2),
        mp.Vector3(length_taper / 2, wvg2.width / 2, -wvg1.height / 2),
        mp.Vector3(-length_taper / 2, wvg1.width / 2, -wvg1.height / 2),
    ]

    # __________________________________________________________________________________________________________________
    # cell + geometry list
    cell_size = mp.Vector3(sx, sy, sz)

    # prism that will be the waveguides + taper
    prism = mp.Prism(vertices=vertices,
                     height=wvg1.height,
                     sidewall_angle=90 - etch_angle,
                     material=wvg1.wvgd_mdm)

    # blocks for the unetched waveguide + substrate
    blk_film = wvg1._blk_film
    blk_sbstrt = wvg1.blk_sbstrt
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
                                  eig_parity=mp.EVEN_Z + mp.ODD_Y,
                                  direction=mp.X)]

    # point source
    # sources = [mp.Source(src=mp.GaussianSource(frequency=fcen, fwidth=df),
    #                      component=mp.Ez,
    #                      center=mp.Vector3(-length_taper / 2 - 1))]

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
    flux = sim.add_flux(fcen,
                        df,
                        nfreq,
                        mp.FluxRegion(center=mon_pt,
                                      size=mp.Vector3(0, sy - 2 * dpml_y, sz - 2 * dpml_z), direction=mp.X))

    return sim, flux, mon_pt


# %% ___________________________________________________________________________________________________________________
wvg1 = wg.ThinFilmWaveguide(1, .3, .7, mp.Medium(index=1.45), mp.Medium(index=3.45), 30, 1, 10, 4)
wvg2 = wg.ThinFilmWaveguide(3, .3, .7, mp.Medium(index=1.45), mp.Medium(index=3.45), 30, 1, 10, 4)
taper, flux, mon_pt = create_taper_sim(wvg1=wvg1,
                                       wvg2=wvg2,
                                       length_taper=5,
                                       fcen=1 / 1.55,
                                       df=0.2,
                                       nfreq=10,
                                       resolution=20,
                                       etch_angle=90)

# %% ___________________________________________________________________________________________________________________
# taper.init_sim()
# eps = taper.get_epsilon()
# np.save('eps.npy', eps)
