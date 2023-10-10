from pystrict import strict

from PySDM.dynamics.collisions.breakup_efficiencies import ConstEb
from PySDM.dynamics.collisions.breakup_fragmentations import Feingold1988Frag
from PySDM.dynamics.collisions.coalescence_efficiencies import ConstEc
from PySDM.dynamics.collisions.collision_kernels import Golovin
from PySDM.formulae import Formulae
from PySDM.initialisation.spectra import Gamma, Lognormal, Sum
from PySDM.physics.constants import si
from PySDM.physics.constants_defaults import rho_w

TRIVIA = Formulae().trivia


@strict
class Settings:
    def __init__(self, formulae: Formulae = None):
        self.n_sd = 1000
        self.n_part = 1e4 / si.cm**3
        self.theta = 0.33e-9 * si.g / rho_w
        self.k = 1
        self.dv = 0.1 * si.m**3
        self.norm_factor = self.n_part * self.dv
        self.dt = 1 * si.seconds
        self.adaptive = False
        self.seed = 44
        self._steps = list(range(1000))
        self.kernel = Golovin(b=2000 * si.cm**3 / si.g / si.s * rho_w)
        self.coal_effs = [ConstEc(Ec=0.8), ConstEc(Ec=0.9), ConstEc(Ec=1.0)]
        self.vmin = 1.0 * si.um**3
        self.nfmax = 10
        self.fragtol = 1e-3
        # self.kappa = 1.28,
        # self.RH = 0.8,
        # self.T = 283.,
        self.fragmentation = Feingold1988Frag(
            scale=self.k * self.theta,
            fragtol=self.fragtol,
            vmin=self.vmin,
            nfmax=self.nfmax,
        )
        self.break_eff = ConstEb(1.0)
        # self.spectrum = Gamma(norm_factor=self.norm_factor, k=self.k, theta=self.theta)

        self.spectrum = Lognormal(
            norm_factor=125 / si.cm**3, m_mode=TRIVIA.volume(5 * si.um), s_geom=1.2
        )

        self.rho = rho_w
        self.formulae = formulae or Formulae(
            fragmentation_function=self.fragmentation.__class__.__name__
        )

    @property
    def output_steps(self):
        return [int(step / self.dt) for step in self._steps]

    # def equilibrate_wet_radii(
    #         self,
    #         r_dry: np.array,
    #         #kappa: float,
    #         #T: float,
    #         #RH: float,
    # ):
    #     def A(T):
    #         R_v = 461.5
    #         # sigma = 0.07564
    #         sigma = 0.07275 * (1 - 0.002 * (T - 291.))
    #         rho_l = 997
    #         return 2 * sigma / (rho_l * R_v * T)
    #
    #     def Kohler(r, kappa, r_d, T):
    #         return 1 + A(T) / r - kappa * (r_d ** 3) / (r ** 3)
    #
    #     def Kohler_derivative(r, kappa, r_d, T):
    #         return -A(T) / (r ** 2) + 3 * kappa * r_d ** 3 / (r ** 4)
    #
    #     def rw_eq(rw0, RH, kappa, rd, T):
    #         rw = [rw0]
    #         for n in range(10 ** 6):
    #             h = -(Kohler(rw[n], kappa, rd, T) - RH) / Kohler_derivative(rw[n], kappa, rd, T)
    #             rw.append(rw[n] + h)
    #             if np.abs(rw[-1] - rw[-2]) < 0.001 * 10 ** (-6):
    #                 break
    #         return rw[-1]
    #
    #     r_wet = []
    #     for i in range(len(r_dry)):
    #         r_wet.append(rw_eq(r_dry[i], self.RH[0], self.kappa[0], r_dry[i], self.T[0]))
    #     return r_wet
