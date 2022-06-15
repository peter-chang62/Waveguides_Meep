"""MPB can solve for omega's given k, or solve for k's given omegas. The former is called with ms.run() (and its
variations), and the latter is called using ms.find_k(). In the RidgeWaveguide class, I use the former to calculate
the dispersion relation. You can use the latter if you are interested to know what modes can propagate at a given
frequency. For example, you can imagine that for a frequency well above the cutoff, there are multiple eigenmodes
that can propagate which results in different k's. MPB solves for the modes starting with highest order and going
down to the fundamental mode. This is because information retained from the higher order modes can be used to
accelerate the calculations for the lower ones. You can pass arguments to ms.find_k(), for example, if you wanted it
to retain the E and H fields that are calculated. This would allow you to plot the field cross-sections inside the
waveguide for the different modes. """

import meep as mp
import numpy as np
import copy
import clipboard_and_style_sheet
from meep import mpb
import matplotlib.pyplot as plt
import time
import scipy.integrate as scint

clipboard_and_style_sheet.style_sheet()


def fft(x):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(x)))


def store_fields(ms, which_band, cls):
    assert isinstance(cls, RidgeWaveguide), \
        f"cls must be an instance of RidgeWaveguide but got {type(cls)}"
    cls: RidgeWaveguide
    ms: mpb.ModeSolver
    cls.E.append(ms.get_efield(which_band=which_band, bloch_phase=False))
    cls.H.append(ms.get_hfield(which_band=which_band, bloch_phase=False))


def get_omega_axis(wl_min, wl_max, NPTS):
    k_min, k_max = 1 / wl_max, 1 / wl_min
    step = (k_max - k_min) / NPTS
    OMEGA = np.arange(k_min, k_max + step, step)
    return OMEGA[:NPTS + 1]


def normalize(vec):
    return vec / np.max(abs(vec))


class RidgeWaveguide:
    """
    The RidgeWaveguide class is for calculating the waveguide dispersion of a rectangular waveguide sitting on top of
    an infinitely large substrate.

    Relevant properties (dimensions, and material dispersion) can be easily modified after class instantiation. This
    way, one can easily sweep waveguide parameters (such as waveguide width / height).
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
        self.blk_sbstrt = mp.Block(size=mp.Vector3(mp.inf, mp.inf, self._hght_sbsrt),
                                   center=mp.Vector3(0, 0, -self._z_offst_sbstrt))

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
        self.run = self.ms.run_yeven
        self.store_fields = True
        self.init_finished = True

    def redef_sim(self):
        """
        re-initialize the mp.Simulation instance used for visualization
        """

        geometry_sim = copy.deepcopy(self.geometry)
        for n in range(len(geometry_sim)):
            geometry_sim[n].material = mp.Medium(epsilon_diag=self.geometry[n].material.epsilon(1 / 1.55).diagonal())

        self.sim = mp.Simulation(cell_size=self.lattice.size,
                                 geometry=geometry_sim,
                                 resolution=self.resolution[0])

    def redef_ms(self):
        """
        when I look at relevant parameters in self.ms (self.ms.geometry_lattice for example) the dimensions are up to
        date, but the epsilon grid is only partially updated when I run the simulation. I can, however,
        re-instantiate self.ms without issue since all the parameters in the currently existing self.ms are correct.
        So, that is what I do here. I've verified that this works correctly
        """

        # create the mode solver instance
        # self.geometry and self.lattice are passed by pointer, whereas
        # self.num_bands and self.resolution are passed by copy
        self.ms = mpb.ModeSolver(geometry_lattice=self.lattice,
                                 geometry=self.geometry,
                                 resolution=self.resolution[0],
                                 num_bands=self.num_bands)
        self.run = self.ms.run_yeven

    def redef_sbstrt_dim(self):
        """
        if waveguide dimensions that affect the substrate dimensions are changed, the substrate dimensions need to be
        appropriately modified
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

    """MPB does not support dispersive materials. To get around this, we pass the k_points at which to calculate 
    omega one at a time, each time modifying epsilon. This is really the same run time because if you pass a list of 
    k_points, it simply calculates the omegas serially for each k_point in the list. 
    
    When modifying epsilon, however, we set the medium of the waveguide and substrate to a material with fixed 
    epsilon. So, it's important to keep a copy of the original dispersive mp.Medium() instance, for reference in the 
    dispersive calculations. This is stored in self._wvgd_mdm and self._sbstrt_mdm. """

    @property
    def wvgd_mdm(self):
        # return the waveguide medium
        return self._wvgd_mdm

    @wvgd_mdm.setter
    def wvgd_mdm(self, medium):
        # set the waveguide medium
        assert isinstance(medium, mp.Medium), \
            f"waveguide medium must be a mp.Medium instance but got type {type(medium)}"
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
            f"substrate medium must be a mp.Medium instance but got type {type(medium)}"
        medium: mp.Medium

        self.blk_sbstrt.material = medium
        self._sbstrt_mdm = medium

        if self.init_finished:
            self.redef_sim()

    def plot2D(self):
        """
        call sim.plot2D() to visualize the supercell
        """

        self.sim.plot2D()

    def _initialize_E_and_H_lists(self):
        self.E = []
        self.H = []

    def plot_mode(self, which_band, which_index_k, Ez_only=True):
        assert which_band < self.num_bands, \
            f"which_band must be <= {self.num_bands - 1} but got {which_band}"

        eps = self.ms.get_epsilon()

        if Ez_only:
            fig, (ax1, ax2) = plt.subplots(1, 2)
            x = self.E[which_index_k][which_band][:, :, 2].__abs__() ** 2
            ax1.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
            ax1.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
            ax1.axis(False)
            fig.suptitle("Ez")

            ax2.plot(np.fft.fftshift(x, 0)[0], label='vertical cut')
            ax2.plot(np.fft.fftshift(x.T, 0)[0], label='horizontal cut')
            ax2.legend(loc='best')
            ax2.axis(False)

        else:
            for n, title in enumerate(['Ex', 'Ey', 'Ez']):
                fig, (ax1, ax2) = plt.subplots(1, 2)
                x = self.E[which_index_k][which_band][:, :, n].__abs__() ** 2
                ax1.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
                ax1.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
                ax1.axis(False)
                fig.suptitle(title)

                ax2.plot(np.fft.fftshift(x, 0)[0], label='vertical cut')
                ax2.plot(np.fft.fftshift(x.T, 0)[0], label='horizontal cut')
                ax2.legend(loc='best')
                ax2.axis(False)

        # don't see the need to plot the H-fields too...
        # for n, title in enumerate(['Hx', 'Hy', 'Hz']):
        #     plt.figure()
        #     x = H[which_index_k][:, :, n].__abs__() ** 2
        #     plt.imshow(eps[::-1, ::-1].T, interpolation='spline36', cmap='binary')
        #     plt.imshow(x[::-1, ::-1].T, cmap='RdBu', alpha=0.9)
        #     plt.axis(False)
        #     plt.title(title)

    def calc_dispersion(self, wl_min, wl_max, NPTS):
        """
        :param wl_min: shortest wavelength
        :param wl_max: longest wavelength
        :param NPTS: number of k_points to interpolate from shortest -> longest wavelength
        :return: result instance with attributes kx (shape: kx, num_bands), freq (shape: kx), *notice the difference
        in array shapes from calc_w_from_k
        """

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        """MPB's find_k functions uses Newton's method which needs bounds and an initial guess that is somewhat close 
        to the real answer (order magnitude). I've run into issues where it couldn't converge, however, so I instead 
        make sure to pass a good guess. We expect materials to have weak dispersion, and so I set epsilon to epsilon(
        f_center), and solve for waveguide dispersion omega(k). From there, I interpolate to get a k(omega) that can 
        be use to extrapolate out and provide good gueses for kmag_guess """

        k_min, k_max = 1 / wl_max, 1 / wl_min
        f_center = (k_max - k_min) / 2 + k_min
        self.blk_wvgd.material = mp.Medium(epsilon_diag=self.wvgd_mdm.epsilon(f_center).diagonal())
        self.blk_sbstrt.material = mp.Medium(epsilon_diag=self.sbstrt_mdm.epsilon(f_center).diagonal())

        start = time.time()

        # I just use the fundamental band for the interpolation (which_band=1),
        # I just interpolate over 10 pts, if using the user-provided NPTS, we might get an overkill
        # of data points, or not enough
        num_bands = self.num_bands  # store self.num_bands
        self.num_bands = 1  # set the mode-solver to only calculate one band
        res = self.calc_w_from_k(wl_min * 0.5, wl_max, 10)  # wl_min * 0.5: interpolation will cover close to wl_min
        self.num_bands = num_bands  # set num_bands back to what it was before

        z = np.polyfit(res.freq[:, 0], res.kx, deg=1)
        spl = np.poly1d(z)  # straight up linear fit, great idea!
        # spl = UnivariateSpline(res.freq[:, 0], res.kx, s=0) # HORRIBLE IDEA!

        if self.store_fields:
            # fields were stored by _calc_non_dispersive
            self._initialize_E_and_H_lists()

        print(f"_______________________________start iteration over Omega's _____________________________________")

        OMEGA = get_omega_axis(wl_min, wl_max, NPTS)
        KX = []
        for n, omega in enumerate(OMEGA):
            # set waveguide epsilon to epsilon(omega)
            self.blk_wvgd.material = mp.Medium(epsilon_diag=self.wvgd_mdm.epsilon(omega).diagonal())
            self.blk_sbstrt.material = mp.Medium(epsilon_diag=self.sbstrt_mdm.epsilon(omega).diagonal())

            # use the interpolated spline to provide a guess for kmag_guess, and pass that to run find_k
            kmag_guess = float(spl(omega))
            kx = self.find_k(mp.EVEN_Y, omega, 1, self.num_bands, mp.Vector3(1), 1e-4,
                             kmag_guess, kmag_guess * 0.1, kmag_guess * 10, *self.band_funcs)
            KX.append(kx)

            print(f'___________________________________{len(OMEGA) - n}__________________________________________')
        stop = time.time()
        print(f"finished in {(stop - start) / 60} minutes")

        if self.store_fields:
            self.E = np.squeeze(np.array(self.E))
            self.E = self.E.reshape((len(KX), self.num_bands, *self.E.shape[1:]))
            self.H = np.squeeze(np.array(self.H))
            self.H = self.H.reshape((len(KX), self.num_bands, *self.H.shape[1:]))

        # ____________________________________ Done ___________________________________________

        class results:
            def __init__(self, parent):
                parent: RidgeWaveguide
                self.kx = np.array(KX)
                self.freq = OMEGA
                self.index_sbstrt = parent.sbstrt_mdm.epsilon(f_center)[2, 2].real ** 0.5
                self.sm_bands = parent.get_sm_band_for_k_axis(self.kx)
                self.sm_dispersion = np.c_[[self.kx[n, i] for n, i in enumerate(self.sm_bands)], self.freq]

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx[:, n], self.freq, '.-') for n in range(self.kx.shape[1])]
                plt.plot(self.kx[:, 0], self.kx[:, 0] / self.index_sbstrt, 'k', label='light-line substrate')
                plt.plot(self.sm_dispersion[:, 0], self.sm_dispersion[:, 1], '.-', color='b', label='sm-dispersion')
                plt.xlabel("k ($\mathrm{1/ \mu m}$)")
                plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{1/ \mu m}$)")
                plt.legend(loc='best')

        return results(self)

    def calc_w_from_k(self, wl_min, wl_max, NPTS):
        """
        :param wl_min: shortest wavelength
        :param wl_max: longest wavelength
        :param NPTS: number of k_points to interpolate from shortest -> longest wavelength
        :return: result instance with attributes kx (shape: kx), freq (shape: kx, num_bands)
        """

        # calc_dispersion calls calc_w_from_k before going on to calculate anything else, so moving this here should
        # be fine

        # if store_fields is true, then re-initialize E and H to empty lists and
        # create the list to pass to *band_funcs
        if self.store_fields:
            self._initialize_E_and_H_lists()
            band_func = lambda ms, which_band: store_fields(ms, which_band, self)
            self.band_funcs = [band_func]

        else:  # otherwise no band_funcs
            self.band_funcs = []

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        k_points = mp.interpolate(NPTS, [mp.Vector3(1 / wl_max), mp.Vector3(1 / wl_min)])
        self.ms.k_points = k_points
        self.run(*self.band_funcs)

        # ____________________________________ Done ___________________________________________

        class results:
            def __init__(self, parent):
                parent: RidgeWaveguide
                self.kx = np.array([i.x for i in k_points])
                self.freq = parent.ms.all_freqs
                self.index_sbstrt = parent.sbstrt_mdm.epsilon(1 / 1.55)[2, 2].real ** 0.5
                self.sm_bands = parent.get_sm_band_for_k_axis(self.kx)
                self.sm_dispersion = np.c_[self.kx, [self.freq[n, i] for n, i in enumerate(self.sm_bands)]]

            def plot_dispersion(self):
                plt.figure()
                [plt.plot(self.kx, self.freq[:, n], '.-') for n in range(self.freq.shape[1])]
                plt.plot(self.kx, self.kx / self.index_sbstrt, 'k', label='light-line substrate')
                plt.plot(self.sm_dispersion[:, 0], self.sm_dispersion[:, 1], '.-', color='b', label='sm-dispersion')
                plt.xlabel("k ($\mathrm{\mu m}$)")
                plt.ylabel("$\mathrm{\\nu}$ ($\mathrm{\mu m}$)")
                plt.legend(loc='best')

        if self.store_fields:
            self.E = np.squeeze(np.array(self.E))
            self.E = self.E.reshape((len(k_points), self.num_bands, *self.E.shape[1:]))
            self.H = np.squeeze(np.array(self.H))
            self.H = self.H.reshape((len(k_points), self.num_bands, *self.H.shape[1:]))

        return results(self)

    def find_k(self, p, omega, band_min, band_max, korig_and_kdir, tol,
               kmag_guess, kmag_min, kmag_max, *band_funcs):
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
        :return: k (list of float(s))
        """

        # make sure all geometric and material parameters are up to date
        self.redef_ms()

        args = [p, omega, band_min, band_max, korig_and_kdir, tol,
                kmag_guess, kmag_min, kmag_max, *band_funcs]

        if (self.wvgd_mdm.valid_freq_range[-1] == 1e20) and (self.sbstrt_mdm.valid_freq_range[-1] == 1e20):
            # materials are NON dispersive
            return self.ms.find_k(*args)

        else:
            # materials are dispersive
            # set the substrate and waveguide epsilon for the input wavelength
            # then run the simulation
            eps_wvgd = self.wvgd_mdm.epsilon(omega)
            eps_sbstrt = self.sbstrt_mdm.epsilon(omega)
            self.blk_wvgd.material = mp.Medium(epsilon_diag=eps_wvgd.diagonal())
            self.blk_sbstrt.material = mp.Medium(epsilon_diag=eps_sbstrt.diagonal())
            return self.ms.find_k(*args)

    def _get_cut_for_sm(self, which_band, k_index):
        mode = self.E[k_index][which_band][:, :, 2].__abs__() ** 2
        resolution = self.resolution[0]

        wvgd_width = self.width * resolution
        wvgd_height = self.height * resolution
        cell_width = self.cell_width * resolution
        cell_height = self.cell_height * resolution
        edge_side_h = (cell_width - wvgd_width) / 2
        edge_side_v = (cell_height - wvgd_height) / 2

        # for some reason, the epsilon grid can be slightly different from
        # what I expect, so scale accordingly to get the right waveguide indices
        # on the simulation grid
        eps = self.ms.get_epsilon()
        h_factor = eps.shape[0] / cell_width
        v_factor = eps.shape[1] / cell_height

        edge_side_h = int(np.round(edge_side_h * h_factor))
        edge_side_v = int(np.round(edge_side_v * v_factor))
        cell_width = int(np.round(cell_width * h_factor))
        cell_height = int(np.round(cell_height * v_factor))

        h_center = np.fft.fftshift(mode.T, 0)[0][edge_side_h:cell_width - edge_side_h]
        v_center = np.fft.fftshift(mode, 0)[0][edge_side_v:cell_height - edge_side_v]

        return h_center, v_center

    def _index_rank_sm(self, k_index):
        IND = []
        for band in range(self.num_bands):
            cnt_h, cnt_v = self._get_cut_for_sm(band, k_index)

            # ___________________________________________________________________________
            # fft, and look at first component after DC
            # (or else DC component will always beat out everything else)
            ind_cnt_h = len(cnt_h) // 2
            ind_cnt_v = len(cnt_v) // 2
            ft_h = fft(cnt_h)[ind_cnt_h:][1:].__abs__()
            ft_v = fft(cnt_v)[ind_cnt_v:][1:].__abs__()
            if any([np.argmax(ft_h) != 0, np.argmax(ft_v) != 0]):
                ind = 0
            else:
                ind = scint.simps(cnt_h) + scint.simps(cnt_v)
            # ___________________________________________________________________________

            IND.append(ind)

        return np.array(IND)

    def get_sm_band_at_k_index(self, k_index):
        return np.argmax(self._index_rank_sm(k_index))

    def get_sm_band_for_k_axis(self, kx):
        band = np.array([self.get_sm_band_at_k_index(i) for i in range(len(kx))])
        return band


class ThinFilmWaveguide(RidgeWaveguide):
    """
    height will be redundant for film thickness and will likely be fixed (you buy a wafer with a fixed film
    thickness). The value that will likely be varied is the etch depth (although you can vary the film thickness if
    that's what you're up to)
    """

    def __init__(self, etch_width, etch_depth, film_thickness, substrate_medium, waveguide_medium,
                 resolution=64, num_bands=4, cell_width=2, cell_height=2):
        assert etch_depth <= film_thickness, "the etch depth cannot exceed the film thickness!"

        super().__init__(etch_width, etch_depth, substrate_medium, waveguide_medium,
                         resolution, num_bands, cell_width, cell_height)

        _blk_film_thickness = film_thickness - etch_depth
        self._blk_film = mp.Block(size=mp.Vector3(mp.inf, mp.inf, _blk_film_thickness),
                                  center=mp.Vector3(0, 0, - (film_thickness - _blk_film_thickness) / 2))
        self._blk_film.material = self.blk_wvgd.material
        self.geometry += [self._blk_film]
        self.redef_sim()
        self.redef_ms()

        self.etch_depth = etch_depth

    def redef_sbstrt_dim(self):
        super().redef_sbstrt_dim()
        _blk_film_thickness = self.film_thickness - self.etch_depth
        self._blk_film.size.z = _blk_film_thickness
        self._blk_film.center.z = - (self.film_thickness - _blk_film_thickness) / 2

    @property
    def film_thickness(self):
        return self.height

    @film_thickness.setter
    def film_thickness(self, film_thickness):
        self.height = film_thickness

    @property
    def etch_width(self):
        return self.width

    @etch_width.setter
    def etch_width(self, etch_width):
        self.width = etch_width

    @RidgeWaveguide.height.setter
    def height(self, height):
        assert height >= self.etch_depth, f"the film thickness (height) must be greater or equal to the etch depth, " \
                                          f"but etch_depth = {self.etch_depth} and film_thickness = {height} "

        # set the height of the waveguide
        self.blk_wvgd.size.z = height
        self.redef_sbstrt_dim()
        self.redef_sim()

    @property
    def etch_depth(self):
        return self._etch_depth

    @etch_depth.setter
    def etch_depth(self, etch_depth):
        assert etch_depth <= self.film_thickness, f"the etch depth must be less than or equal to the " \
                                                  f"film thickness, but etch_depth = {etch_depth} and " \
                                                  f"film_thickness = {self.height} "

        self._etch_depth = etch_depth
        self.redef_sbstrt_dim()
        self.redef_sim()
