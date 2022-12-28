"""MPB can solve for omega's given k, or solve for k's given omegas. The
former is called with ms.run() (and its variations), and the latter is called
using ms.find_k(). Just like MEEP's Simulation run() function, you can pass
arguments to MPB's ModeSolver's find_k() and run(). For example, you can tell
it to output z and y parities, and any function you define takes arguments:
func(ms_instance, band_index) that will be called at each (k, band) or (
omega, band) point. In my case, I pass a function that retrieves and stores
the E and H fields that I intend to use to calculate mode-area. I realize now
that I don't need to store the H-fields, but honestly the memory requirement
here is small so I don't really care. """

import meep as mp
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import time
import scipy.integrate as scint
import scipy.constants as sc

clipboard_and_style_sheet.style_sheet()

# conversion of beta2 calculated here to beta2 in ps^2/km
conversion = sc.c ** -2 * 1e12 ** 2 * 1e3 ** 2 * 1e-9


def store_fields(ms, which_band, cls):
    assert isinstance(cls, RidgeWaveguide), \
        f"cls must be an instance of RidgeWaveguide but got {type(cls)}"
    cls: RidgeWaveguide
    ms: mpb.ModeSolver

    cls.E.append(ms.get_efield(which_band=which_band, bloch_phase=False))
    cls.H.append(ms.get_hfield(which_band=which_band, bloch_phase=False))


def store_group_velocity(ms, which_band, cls):
    assert isinstance(cls, RidgeWaveguide), \
        f"cls must be an instance of RidgeWaveguide but got {type(cls)}"
    cls: RidgeWaveguide
    ms: mpb.ModeSolver

    v_g = np.array(ms.compute_one_group_velocity(which_band))
    cls.v_g.append(v_g)


def get_omega_axis(wl_min, wl_max, NPTS):
    k_min, k_max = 1 / wl_max, 1 / wl_min
    step = (k_max - k_min) / NPTS
    OMEGA = np.arange(k_min, k_max + step, step)
    return OMEGA[:NPTS + 1]


def normalize(vec):
    return vec / np.max(abs(vec))


def mode_area(I, resolution):
    # integral(I * dA) ** 2  / integral(I ** 2 * dA) is the common definition
    # that is used reference:
    # https://www.rp-photonics.com/effective_mode_area.html this gives an
    # overall dimension of dA in the numerator
    area = scint.simpson(scint.simpson(I)) ** 2 / \
        scint.simpson(scint.simpson(I ** 2))
    area /= resolution ** 2
    return area


class RidgeWaveguide:
    """
    The RidgeWaveguide class is for calculating the waveguide dispersion of a
    rectangular waveguide sitting on top of an infinitely large substrate.

    Relevant properties (dimensions, and material dispersion) can be easily
    modified after class instantiation. This way, one can easily sweep
    waveguide parameters (such as waveguide width / height).
    """

    def __init__(self, width, height, substrate_medium, waveguide_medium,
                 resolution=64, num_bands=4, cell_width=2, cell_height=2):

        self.init_finished = False

        # create the lattice (cell)
        self.lattice = mp.Lattice(size=mp.Vector3(0, cell_width, cell_height))

        # create the waveguide and the substrate
        # the substrate dimensions are calculated from the waveguide and
        # lattice dimensions (so it has to be created after the first two)
        self.blk_wvgd = mp.Block(size=mp.Vector3(mp.inf, width, height))
        self.blk_sbstrt = mp.Block(size=mp.Vector3(mp.inf, mp.inf,
                                                   self._hght_sbsrt),
                                   center=mp.Vector3(0, 0,
                                                     -self._z_offst_sbstrt))

        # set the substrate and waveguide medium
        self.sbstrt_mdm = substrate_medium
        self.wvgd_mdm = waveguide_medium

        # list of MEEP objects that will be passed to the mode solver
        self.geometry = [self.blk_wvgd, self.blk_sbstrt]

        # create the mode solver instance
        # self.geometry and self.lattice are passed by pointer, whereas
        # self.num_bands and self.resolution are passed by copy
        self.ms = mpb.ModeSolver(geometry_lattice=self.lattice,
                                 geometry=self.geometry,
                                 resolution=resolution,
                                 num_bands=num_bands)

        # create an mp.Simulation instance
        # this is only used for visualization (sim.plot2D())
        # plotting is a useful tool to see if the dimensions are set to what
        # you had intended to
        self.redef_sim()

        self.band_funcs = []
        self.run = self.ms.run
        self.init_finished = True

    def redef_sim(self):
        """
        re-initialize the mp.Simulation instance used for visualization
        """

        geometry_sim = copy.deepcopy(self.geometry)
        for n in range(len(geometry_sim)):
            geometry_sim[n].material = mp.Medium(
                epsilon=self.geometry[n].material.epsilon(1 / 1.55)[2, 2]
            )

        self.sim = mp.Simulation(cell_size=self.lattice.size,
                                 geometry=geometry_sim,
                                 resolution=self.resolution[0])

    def redef_ms(self):
        """
        when I look at relevant parameters in self.ms (
        self.ms.geometry_lattice for example) the dimensions are up to date,
        but the epsilon grid is only partially updated when I run the
        simulation. I can, however, re-instantiate self.ms without issue
        since all the parameters in the currently existing self.ms are
        correct. So, that is what I do here. I've verified that this works
        correctly

        This is basically copied over from __init__
        """

        # create the mode solver instance
        # self.geometry and self.lattice are passed by pointer, whereas
        # self.num_bands and self.resolution are passed by copy
        self.ms = mpb.ModeSolver(geometry_lattice=self.lattice,
                                 geometry=self.geometry,
                                 resolution=self.resolution[0],
                                 num_bands=self.num_bands)
        self.run = self.ms.run

    def redef_sbstrt_dim(self):
        """
        if waveguide dimensions that affect the substrate dimensions are
        changed, the substrate dimensions need to be appropriately modified
        """

        self.blk_sbstrt.size.z = self._hght_sbsrt
        self.blk_sbstrt.center.z = -self._z_offst_sbstrt

    @property
    def _hght_sbsrt(self):
        # calculate the appropriate height of the substrate
        return (self.cell_height / 2) - (self.height / 2)

    @property
    def _z_offst_sbstrt(self):
        # calculate the appropriate vertical offset of the substrate
        # to place it directly beneath the waveguide
        return (self._hght_sbsrt / 2) + (self.height / 2)

    @property
    def width(self):
        # return the width of the waveguide
        return self.blk_wvgd.size.y

    @width.setter
    def width(self, width):
        # set the width of the waveguide
        self.blk_wvgd.size.y = width
        self.redef_sim()

    @property
    def height(self):
        # return the height of the waveguide
        return self.blk_wvgd.size.z

    @height.setter
    def height(self, height):
        # set the height of the waveguide
        self.blk_wvgd.size.z = height
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def cell_width(self):
        # return the width of the supercell
        return self.lattice.size.y

    @cell_width.setter
    def cell_width(self, sy):
        # set the width of the supercell
        self.lattice.size.y = sy
        self.redef_sim()

    @property
    def cell_height(self):
        # return the height of the supercell
        return self.lattice.size.z

    @cell_height.setter
    def cell_height(self, sz):
        # set the height of the supercell
        self.lattice.size.z = sz
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def resolution(self):
        # return the resolution of the simulation grid
        return self.ms.resolution

    @resolution.setter
    def resolution(self, resolution):
        # set the resolution of the simulation grid
        self.ms.resolution = resolution
        self.redef_sim()

    @property
    def num_bands(self):
        # return the number of bands to be calculated
        return self.ms.num_bands

    @num_bands.setter
    def num_bands(self, num):
        # set the number of bands to be calculated
        self.ms.num_bands = num

    # __________________________________________________________________________
    # MPB does not support dispersive materials. To get around this, we pass
    # the omegas at which to calculate k, one at a time, each time modifying
    # epsilon.

    # When modifying epsilon, however, we set the medium of the waveguide and
    # substrate to a material with fixed epsilon. So, it's important to keep
    # a copy of the original dispersive mp.Medium() instance, for reference
    # in the dispersive calculations. This is stored in self._wvgd_mdm and
    # self._sbstrt_mdm.
    # __________________________________________________________________________

    @property
    def wvgd_mdm(self):
        # return the waveguide medium
        return self._wvgd_mdm

    @wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        # set the waveguide medium
        assert isinstance(medium, mp.Medium), \
            f"waveguide medium must be a mp.Medium instance but got type " \
            f"{type(medium)} "
        medium: mp.Medium

        self.blk_wvgd.material = medium
        self._wvgd_mdm = medium

        if self.init_finished:
            self.redef_sim()

    @property
    def sbstrt_mdm(self):
        # return the substrate medium
        return self._sbstrt_mdm

    @sbstrt_mdm.setter
    def sbstrt_mdm(self, medium):
        # set the substrate medium
        assert isinstance(medium, mp.Medium), \
            f"substrate medium must be a mp.Medium instance but got type " \
            f"{type(medium)} "
        medium: mp.Medium

        self.blk_sbstrt.material = medium
        self._sbstrt_mdm = medium

        if self.init_finished:
            self.redef_sim()

    # __________________________________________________________________________
    # I only do this for the waveguide GeometricObject: You can substitute
    # the waveguide geometric object via self.blk_wvgd = mp.GeometricObject(
    # ), and it will update the geometry list, which will incorporate the
    # changes into the simulation. I don't see the need to do this for any
    # other geometric object in the simulation cell.

    # The main use case for this, (although you can always create your own
    # waveguide geometry), is to substitute:

    # self.blk_wvgd = geometry.convert_block_to_trapezoid(blk=self.blk_wvgd,
    #                                                     angle_deg=80)

    # *realize this will mess up many functionalities (like self.width =
    # new_width)!!!* So, the typical use case is to:
    #   1. save self.blk_wvgd to another variable
    #   2. set self.blk_wvgd to a new mp.GeometricObject()
    #   3. run the simulation
    #   4. reset self.blk_wvgd to the original mp.GeometricObject() that you
    # had saved in step 1.
    # __________________________________________________________________________

    @property
    def blk_wvgd(self):
        if self.init_finished:
            return self.geometry[0]
        else:
            return self._blk_wvgd

    @blk_wvgd.setter
    def blk_wvgd(self, GeometricObject):
        self._blk_wvgd = GeometricObject

        if self.init_finished:
            self.geometry[0] = GeometricObject
            self.redef_sim()

    def plot2D(self):
        """
        call sim.plot2D() to visualize the supercell. I realized that plot2D(
        ) might not capture small changes in waveguide dimensions (less than
        20 nm for example), but the changes are actually there in the epsilon
        grid ( returned with plot_eps). The sub-pixel smoothing feature is
        also clear there, since I can see noticeable changes with plot_eps()
        even if I change the dimensions by something less than 1 / resolution
        """

        self.sim.plot2D()

    def plot_eps(self, cmap='Greys'):
        """
        Realize that if your materials are dispersive then epsilon is all 1,
        I do take care of this in calc_dispersion and find_k, but if you haven't
        run those yet then you'll see a big array of ones.
        There's a weird issue where
            1. eps array updates if I change the waveguide width
            2. eps array does not update if I change the cell_width
                a. however, if I call self.sim.plot2D() and then
                self.plot_eps() then it has not changed the shape of eps,
                but has changed the width of the waveguide to sort of "scale"
                b. If I run a simulation, however, then call plot_eps,
                then the eps array *is correct, showing that it was updated
                for the simulation*
        All in all, it appears that things ship shape up for the simulation,
        so actual calculations should be all good, but visualization can be a
        little frustrating. Just run between plot2D() and plot_eps() as you
        see fit.

        :param cmap: matplotlib color map
        :return: epsilon array
        """

        self.ms.init_params(mp.NO_PARITY, False)
        eps = self.ms.get_epsilon()

        plt.imshow(eps[::-1, ::-1].T, cmap=cmap)
        return eps

    def _initialize_E_and_H_lists(self):
        self.E = []
        self.H = []
        self.v_g = []

    def plot_mode(self, which_band, which_index_k, component=mp.Ey):
        assert which_band < self.num_bands, \
            f"which_band must be <= {self.num_bands - 1} but got {which_band}"

        eps = self.ms.get_epsilon()

        fig, ax = plt.subplots(1, 1)
        x = self.E[which_index_k][which_band][:, :, component].__abs__() ** 2
        area = mode_area(x, self.resolution[0])

        ax.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
        ax.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
        ax.axis(False)
        ax.set_title("Ez" + '\n' +
                     'width=' + '%.2f' % self.width +
                     ' $\\mathrm{\\mu m}$' + ', ' +
                     'height=' + '%.2f' % self.height +
                     ' $\\mathrm{\\mu m}$' + '\n' +
                     '$\\mathrm{A_{eff}}$ = %.3f' % area +
                     ' $\\mathrm{\\mu m^2}$')

        return fig, ax  # in case you want to add additional things

    def calc_dispersion(self, wl_min, wl_max, NPTS,
                        eps_func_wvgd=None, eps_func_sbstrt=None):
        """
        :param wl_min: shortest wavelength
        :param wl_max: longest wavelength
        :param NPTS: number of k_points to interpolate from shortest ->
        longest wavelength
        :return: result instance with attributes kx (shape: kx, num_bands),
        freq (shape: kx), *notice the difference in array shapes from
        calc_w_from_k
        """

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        # ______________________________________________________________________
        # MPB's find_k functions uses Newton's method which needs bounds and
        # an initial guess that is somewhat close to the real answer (order
        # magnitude). We expect materials to have weak dispersion, and so I
        # set epsilon to epsilon( f_center), and solve for waveguide
        # dispersion omega(k). From there, I interpolate to get a k(omega)
        # that can be used to extrapolate out and provide good gueses for
        # kmag_guess
        # ______________________________________________________________________

        k_min, k_max = 1 / wl_max, 1 / wl_min
        f_center = (k_max - k_min) / 2 + k_min

        # ______________________________________________________________________

        # if eps_func_wvgd is not provided, then set the material epsilon via
        # calling the usual mp.Medium().epsilon otherwise, you can set
        # epsilon to something you provide. The use case for this is that
        # MEEP materials can only use the simple Sellmeier equation,
        # but sometimes you would like to use the extended formulas that are
        # more accurate over broad bandwidths or at your particular
        # experiment temperature. originally this was:
        #
        #   self.blk_wvgd.material = mp.Medium(
        #       epsilon=self.wvgd_mdm.epsilon(f_center)[2, 2])
        #   self.blk_sbstrt.material = mp.Medium(
        #       epsilon=self.sbstrt_mdm.epsilon(f_center)[2, 2])

        if eps_func_wvgd is None:
            self.blk_wvgd.material = mp.Medium(
                epsilon=self.wvgd_mdm.epsilon(f_center)[2, 2])
        else:
            self.blk_wvgd.material = mp.Medium(
                epsilon=eps_func_wvgd(f_center))

        if eps_func_sbstrt is None:
            self.blk_sbstrt.material = mp.Medium(
                epsilon=self.sbstrt_mdm.epsilon(f_center)[2, 2])
        else:
            self.blk_sbstrt.material = mp.Medium(
                epsilon=eps_func_sbstrt(f_center))
        # ______________________________________________________________________

        start = time.time()

        # I just use the fundamental band for the interpolation (
        # which_band=1), I just interpolate over 10 pts, if using the
        # user-provided NPTS, we might get an overkill of data points,
        # or not enough
        num_bands = self.num_bands  # store self.num_bands
        self.num_bands = 1  # set the mode-solver to only calculate one band
        # wl_min * 0.5: interpolation will cover close to wl_min
        res = self.calc_w_from_k(wl_min * 0.5, wl_max, 10)
        self.num_bands = num_bands  # set num_bands back to what it was before

        z = np.polyfit(res.freq[:, 0], res.kx, deg=1)
        spl = np.poly1d(z)  # straight up linear fit, great idea!
        # spl = UnivariateSpline(res.freq[:, 0], res.kx, s=0) # HORRIBLE IDEA!

        # fields were stored by calc_w_from_k
        self._initialize_E_and_H_lists()

        print(f"_____________start iteration over Omega's ____________________")

        OMEGA = get_omega_axis(wl_min, wl_max, NPTS)
        KX = []
        for n, omega in enumerate(OMEGA):
            # use the interpolated spline to provide a guess for kmag_guess,
            # and pass that to run find_k the material epsilon is already
            # changed for each omega inside find_k
            kmag_guess = float(spl(omega))
            kx = self.find_k(mp.NO_PARITY, omega, 1,
                             self.num_bands, mp.Vector3(1), 1e-6,
                             kmag_guess, kmag_guess * 0.1,
                             kmag_guess * 10, *self.band_funcs,
                             eps_func_wvgd=eps_func_wvgd,
                             eps_func_sbstrt=eps_func_sbstrt)
            KX.append(kx)

            # delete ________________________ this is a curiosity! _____________
            # to fix this, I now set epsilon to epsilon[2, 2] (ne component)
            # eps = self.ms.get_epsilon()
            # eps_z = self.wvgd_mdm.epsilon(omega)[2, 2]
            # if abs(eps - eps_z).min() < 1e-8:
            #     print(True)
            # delete ___________________________________________________________

            print(
                f'____________________{len(OMEGA) - n}______________________')
        stop = time.time()
        print(f"finished in {(stop - start) / 60} minutes")

        # ____________________________ Done ____________________________________

        self.E = np.squeeze(np.array(self.E))
        self.E = self.E.reshape((len(KX), self.num_bands, *self.E.shape[1:]))
        self.H = np.squeeze(np.array(self.H))
        self.H = self.H.reshape((len(KX), self.num_bands, *self.H.shape[1:]))
        self.v_g = np.squeeze(np.array(self.v_g))
        self.v_g = self.v_g.reshape((len(KX), self.num_bands,
                                     *self.v_g.shape[1:]))

        class results:
            def __init__(self, parent):
                parent: RidgeWaveguide
                self.kx = np.array(KX)  # owns its own array after return call
                self.freq = OMEGA  # owns its own array after return call
                self.v_g = np.copy(parent.v_g)  # passed by pointer from parent
                # float
                self.index_sbstrt = \
                    parent.sbstrt_mdm.epsilon(f_center)[2, 2].real ** 0.5

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx[:, n], self.freq, '.-')
                 for n in range(self.kx.shape[1])]
                plt.plot(self.kx[:, 0], self.kx[:, 0] / self.index_sbstrt,
                         'k', label='light-line substrate')
                plt.xlabel("k ($\\mathrm{1/ \\mu m}$)")
                plt.ylabel("$\\mathrm{\\nu}$ ($\\mathrm{1/ \\mu m}$)")
                plt.legend(loc='best')

        return results(self)

    def calc_w_from_k(self, wl_min, wl_max, NPTS):
        """
        :param wl_min: shortest wavelength
        :param wl_max: longest wavelength
        :param NPTS: number of k_points to interpolate from shortest ->
        longest wavelength
        :return: result instance with attributes kx (shape: kx), freq (shape:
        kx, num_bands)
        """

        # calc_dispersion calls calc_w_from_k before going on to calculate
        # anything else, so moving this here should be fine

        # re-initialize E and H to empty lists and
        # create the list to pass to *band_funcs
        self._initialize_E_and_H_lists()

        def band_func1(ms, which_band): return store_fields(
            ms, which_band, self)

        def band_func2(ms, which_band): return store_group_velocity(ms,
                                                                    which_band,
                                                                    self)
        self.band_funcs = [band_func1, band_func2, mpb.display_yparities,
                           mpb.display_zparities]

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max),
                                         mp.Vector3(1 / wl_min)])
        self.ms.k_points = k_points
        self.run(*self.band_funcs)

        # ______________________ Done __________________________________________

        self.E = np.squeeze(np.array(self.E))
        self.E = self.E.reshape((len(k_points),
                                 self.num_bands, *self.E.shape[1:]))
        self.H = np.squeeze(np.array(self.H))
        self.H = self.H.reshape((len(k_points), self.num_bands,
                                 *self.H.shape[1:]))
        self.v_g = np.squeeze(np.array(self.v_g))
        self.v_g = self.v_g.reshape((len(k_points), self.num_bands,
                                     *self.v_g.shape[1:]))

        class results:
            def __init__(self, parent):
                parent: RidgeWaveguide
                # owns it's own array
                self.kx = np.array([i.x for i in k_points])
                # passed by pointer from parent
                self.freq = np.copy(parent.ms.all_freqs)
                self.v_g = np.copy(parent.v_g)  # passed by pointer from parent
                # float
                self.index_sbstrt = \
                    parent.sbstrt_mdm.epsilon(1 / 1.55)[2, 2].real ** 0.5

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx, self.freq[:, n], '.-')
                 for n in range(self.freq.shape[1])]
                plt.plot(self.kx, self.kx / self.index_sbstrt, 'k',
                         label='light-line substrate')
                plt.xlabel("k ($\\mathrm{\\mu m}$)")
                plt.ylabel("$\\mathrm{\\nu}$ ($\\mathrm{\\mu m}$)")
                plt.legend(loc='best')

        return results(self)

    # __________________________________________________________________________
    # this was originally meant to take the same arguments as find_k() from
    # mpb.ModeSolver(). However, I've now added eps_func_wvgd and
    # eps_func_sbstrt *make sure that if you decide to use these, that you
    # pass them in as kwargs!!* Otherwise they'll be interpreted as part of
    # band_funcs (not to worry too much, you'll just get an error and then
    # it'll be obvious what you did wrong)
    # __________________________________________________________________________
    def find_k(self, p, omega, band_min, band_max, korig_and_kdir, tol,
               kmag_guess, kmag_min, kmag_max, *band_funcs, eps_func_wvgd=None,
               eps_func_sbstrt=None):
        """
        :param p: parity
        :param omega: frequency
        :param band_min: minimum band index
        :param band_max: maximum band index
        :param korig_and_kdir: k direction (unit vector)
        :param tol: tolerance
        :param kmag_guess: guess for the wave-vector magnitude (n/lambda)
        :param kmag_min: minimum wave-vector magnitude
        :param kmag_max: maximum wave-vector magnitude
        :param band_funcs: additional arguments to pass to ms.find_k()
        :param eps_func_wvgd: default None, function that takes omega and
        returns eps (float) for the waveguide
        :param eps_func_sbstrt: default None, function that takes omega and
        returns eps (float) for the substrate

        :return: k (list of float(s))
        """

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        args = [p, omega, band_min, band_max, korig_and_kdir, tol,
                kmag_guess, kmag_min, kmag_max, *band_funcs]

        # if epsilon functions are provided, then use those
        # otherwise obtain epsilon from mp.Medium().epsilon()
        if eps_func_wvgd is not None:
            eps_wvgd = eps_func_wvgd(omega)
            self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
        else:
            eps_wvgd = self.wvgd_mdm.epsilon(omega)
            self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd[2, 2])

        if eps_func_sbstrt is not None:
            eps_sbstrt = eps_func_sbstrt(omega)
            self.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)
        else:
            eps_sbstrt = self.sbstrt_mdm.epsilon(omega)
            self.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt[2, 2])

        return self.ms.find_k(*args)


class ThinFilmWaveguide(RidgeWaveguide):
    """
    height will be redundant for film thickness and will likely be fixed (you
    buy a wafer with a fixed film thickness). The value that will likely be
    varied is the etch depth (although you can vary the film thickness if
    that's what you're up to)
    """

    def __init__(self, etch_width, etch_depth, film_thickness,
                 substrate_medium, waveguide_medium,
                 resolution=64, num_bands=4, cell_width=2, cell_height=2):
        assert etch_depth <= film_thickness, \
            "the etch depth cannot exceed the film thickness!"

        # create a ridgewaveguide with width etch_width, and height
        # film_thickness we will add a second waveguide block for the
        # remainder of the film below
        super().__init__(etch_width, film_thickness, substrate_medium,
                         waveguide_medium, resolution, num_bands, cell_width,
                         cell_height)

        # calculate the remaining film thickness after the etch depth and add
        # a block with the appropriate z offset to place it below the etched
        # out waveguide
        blk_film_thickness = film_thickness - etch_depth
        self._blk_film = mp.Block(size=mp.Vector3(mp.inf,
                                                  mp.inf, blk_film_thickness),
                                  center=mp.Vector3(0, 0,
                                                    - (film_thickness -
                                                       blk_film_thickness) / 2))
        # same medium as the waveguide
        self._blk_film.material = self.blk_wvgd.material
        self.geometry += [self._blk_film]  # add it to the geometry

        # these re-initialize the mp.Simulation and mpb.ModeSolver instances
        # based off current attributes
        self.redef_sim()
        self.redef_ms()

        # need to initialize self._etch_depth
        # will also call self.redef_sbstrt_dim() and self.redef_sim() which
        # is redundant here, but whatever
        self.etch_depth = etch_depth

    def redef_sbstrt_dim(self):
        # in addition to redef_sbstrt_dim() from RidgeWaveguide,
        # also re-initialize the remaining film block, this is basically a
        # copy of what was done in __init__

        super().redef_sbstrt_dim()
        blk_film_thickness = self.film_thickness - self.etch_depth
        self._blk_film.size.z = blk_film_thickness
        self._blk_film.center.z = \
            - (self.film_thickness - blk_film_thickness) / 2

    @property
    def film_thickness(self):
        # return the film thickness (redundant for height) of the waveguide
        return self.height

    @film_thickness.setter
    # set the film thickness (redundant for height of the waveguide)
    def film_thickness(self, film_thickness):
        self.height = film_thickness

    @property
    def etch_width(self):
        # return the width (synonymous for etch_width) of the waveguide
        return self.width

    @etch_width.setter
    def etch_width(self, etch_width):
        # set the width (synonymous for etch_width) of the waveguide
        self.width = etch_width

    @RidgeWaveguide.height.setter
    def height(self, height):
        # for sanity checks: don't let the film thickness be less than the
        # etch depth, otherwise proceed with the same height setter function
        # as RidgeWaveguide
        assert height >= self.etch_depth, \
            f"the film thickness (height) must be greater or equal to the " \
            f"etch depth, but etch_depth = {self.etch_depth} and " \
            f"film_thickness = {height} "

        # ________________ copied over from RidgeWaveguide _____________________
        # set the height of the waveguide
        self.blk_wvgd.size.z = height
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def etch_depth(self):
        # return the etch depth
        return self._etch_depth

    @etch_depth.setter
    def etch_depth(self, etch_depth):
        # sanity check: don't let the etch depth exceed the film thickness of
        # the waveguide otherwise set the etch depth
        assert etch_depth <= self.film_thickness, \
            f"the etch depth must be less than or equal to the film " \
            f"thickness, but etch_depth = {etch_depth} and film_thickness = " \
            f"{self.height} "

        self._etch_depth = etch_depth
        self.redef_sbstrt_dim()
        self.redef_sim()

    @RidgeWaveguide.wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        # ______________________________________________________________________
        # this is the same as sbstrt_mdm in RidgeWaveguide but with the added
        # line:
        #         self._blk_film.material = medium
        # inside the self.init_finished if statement
        # ______________________________________________________________________

        # set the waveguide medium
        assert isinstance(medium, mp.Medium), \
            f"waveguide medium must be a mp.Medium instance but got type " \
            f"{type(medium)} "
        medium: mp.Medium

        self.blk_wvgd.material = medium
        self._wvgd_mdm = medium

        if self.init_finished:
            self.redef_sim()
            self._blk_film.material = medium

    def plot_mode(self, which_band, which_index_k, component=mp.Ey):
        # ______________________________________________________________________
        # this is the same as plot_mode from RidgeWaveguide, but it adds the
        # etch_depth to the title
        # ______________________________________________________________________

        x = self.E[which_index_k][which_band][:, :, component].__abs__() ** 2
        area = mode_area(x, self.resolution[0])
        fig, ax = super().plot_mode(which_band, which_index_k, component)
        ax.set_title("Ez" + '\n' +
                     'width=' + '%.2f' % self.width +
                     ' $\\mathrm{\\mu m}$' + ', ' +
                     'height=' + '%.2f' % self.height +
                     ' $\\mathrm{\\mu m}$' + ', ' +
                     'depth=' + '%.2f' % self.etch_depth +
                     ' $\\mathrm{\\mu m}$' + '\n' +
                     '$\\mathrm{A_{eff}}$ = %.3f' % area +
                     ' $\\mathrm{\\mu m^2}$')
        return fig, ax  # in case you want to add additional things

    # __________________________________________________________________________
    # this was originally meant to take the same arguments as find_k() from
    # mpb.ModeSolver(). However, I've now added eps_func_wvgd and
    # eps_func_sbstrt *make sure that if you decide to use these, that you
    # pass them in as kwargs!!* Otherwise they'll be interpreted as part of
    # band_funcs (not to worry too much, you'll just get an error and then
    # it'll be obvious what you did wrong)
    # __________________________________________________________________________
    def find_k(self, p, omega, band_min, band_max, korig_and_kdir, tol,
               kmag_guess, kmag_min, kmag_max, *band_funcs, eps_func_wvgd=None,
               eps_func_sbstrt=None):

        # ______________________________________________________________________

        # this is the same as find_k from RidgeWaveguide but with the added
        # line:
        #
        #   self._blk_film.material = mp.Medium(epsilon=eps_wvgd[2, 2])
        #
        # that is included whenever self.blk_wvgd.material is altered,
        # namely wherever there is the line:
        #
        #   self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
        # ______________________________________________________________________
        """
        :param p: parity
        :param omega: frequency
        :param band_min: minimum band index
        :param band_max: maximum band index
        :param korig_and_kdir: k direction (unit vector)
        :param tol: tolerance
        :param kmag_guess: guess for the wave-vector magnitude (n/lambda)
        :param kmag_min: minimum wave-vector magnitude
        :param kmag_max: maximum wave-vector magnitude
        :param band_funcs: additional arguments to pass to ms.find_k()
        :param eps_func_wvgd: default None, function that takes omega and
        returns eps (float) for the waveguide
        :param eps_func_sbstrt: default None, function that takes omega and
        returns eps (float) for the substrate
        :return: k (list of float(s))
        """

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        args = [p, omega, band_min, band_max, korig_and_kdir, tol,
                kmag_guess, kmag_min, kmag_max, *band_funcs]

        # if epsilon functions are provided, then use those
        # otherwise obtain epsilon from mp.Medium().epsilon()
        if eps_func_wvgd is not None:
            eps_wvgd = eps_func_wvgd(omega)
            self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd)
            self._blk_film.material = mp.Medium(epsilon=eps_wvgd)
        else:
            eps_wvgd = self.wvgd_mdm.epsilon(omega)
            self.blk_wvgd.material = mp.Medium(epsilon=eps_wvgd[2, 2])
            self._blk_film.material = mp.Medium(epsilon=eps_wvgd[2, 2])

        if eps_func_sbstrt is not None:
            eps_sbstrt = eps_func_sbstrt(omega)
            self.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt)
        else:
            eps_sbstrt = self.sbstrt_mdm.epsilon(omega)
            self.blk_sbstrt.material = mp.Medium(epsilon=eps_sbstrt[2, 2])

        return self.ms.find_k(*args)
