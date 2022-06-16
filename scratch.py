import meep.materials as mt
import waveguide_dispersion as wg
import materials as mtp
import meep as mp
import matplotlib.pyplot as plt

film = wg.ThinFilmWaveguide(
    etch_width=3,
    etch_depth=.3,
    film_thickness=0.630,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    resolution=40,  # 20 -> 40 made neglibile difference!
    num_bands=4,
    cell_width=8,
    cell_height=4
)

film.num_bands = 1
res = film.calc_dispersion(.8, 3.5, 30)

res.plot_dispersion()
# [film.plot_mode(0, n) for n in range(len(res.kx))]
