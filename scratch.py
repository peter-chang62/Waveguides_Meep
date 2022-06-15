import meep.materials as mt
import waveguide_dispersion as wg
import materials as mtp
import meep as mp

film = wg.ThinFilmWaveguide(
    etch_width=3,
    etch_depth=.35,
    film_thickness=0.35,
    substrate_medium=mtp.Al2O3,  # dispersive
    waveguide_medium=mt.LiNbO3,  # dispersive
    resolution=20,  # 20 -> 40 made neglibile difference!
    num_bands=8,
    cell_width=5,
    cell_height=5
)

film.plot2D()
