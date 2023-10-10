import numpy as np
from PySDM_examples.utils import BasicSimulation

from PySDM import Builder
from PySDM.backends import CPU
from PySDM.backends.impl_numba.test_helpers import scipy_ode_condensation_solver
from PySDM.dynamics import AmbientThermodynamics, Coalescence, Collision, Condensation
from PySDM.dynamics.collisions.breakup_efficiencies import constEb
from PySDM.dynamics.collisions.breakup_fragmentations import always_n, constant_size
from PySDM.dynamics.collisions.coalescence_efficiencies import _gravitational, berry1967
from PySDM.dynamics.collisions.collision_kernels import Geometric, Golovin, Hydrodynamic
from PySDM.environments import Box, Parcel
from PySDM.initialisation import equilibrate_wet_radii
from PySDM.initialisation.sampling.spectral_sampling import (
    ConstantMultiplicity,
    Linear,
    Logarithmic,
    UniformRandom,
)
from PySDM.physics import si


class Simulation(BasicSimulation):
    def __init__(
        self,
        settings,
        sampling_class=ConstantMultiplicity,
        products=None,
        scipy_solver=False,
    ):
        # env = Box(
        #     dt=settings.timestep,
        #     dv= settings.dv
        # )
        env = Parcel(
            dt=settings.timestep,
            p0=settings.initial_pressure,
            q0=settings.initial_vapour_mixing_ratio,
            T0=settings.initial_temperature,
            w=settings.vertical_velocity,
            mass_of_dry_air=44 * si.kg,
        )
        builder = Builder(n_sd=settings.n_sd, backend=CPU(formulae=settings.formulae))
        builder.set_environment(env)
        builder.add_dynamic(AmbientThermodynamics())
        builder.add_dynamic(
            Condensation(rtol_thd=settings.rtol_thd, rtol_x=settings.rtol_x)
        )
        # builder.add_dynamic(Collision(collision_kernel=Golovin(b=1.5*si.m**3 /(si.kg*si.s)),
        #             coalescence_efficiency=_gravitational,
        #             breakup_efficiency=constEb,
        #             fragmentation_function=constant_size,
        #             enable_breakup=True))
        builder.add_dynamic(
            Coalescence(
                collision_kernel=Golovin(b=0.5 * si.m**3 / (si.kg * si.s))
                # collision_kernel=Hydrodynamic()
            )
        )

        for attribute in (
            "critical supersaturation",
            "equilibrium supersaturation",
            "critical volume",
        ):
            builder.request_attribute(attribute)

        volume = env.mass_of_dry_air / settings.initial_air_density
        attributes = {
            k: np.empty(0)
            for k in ("dry volume", "kappa times dry volume", "multiplicity", "kappa")
        }

        assert len(settings.aerosol_modes_by_kappa.keys()) == 1
        kappa = tuple(settings.aerosol_modes_by_kappa.keys())[0]
        spectrum = settings.aerosol_modes_by_kappa[kappa]

        # r_dry, n_per_volume = sampling_class(spectrum, size_range=(1e-9,0.5e-6)).sample(settings.n_sd,backend=CPU(formulae=settings.formulae))
        r_dry, n_per_volume = sampling_class(spectrum).sample(
            settings.n_sd, backend=CPU(formulae=settings.formulae)
        )
        v_dry = settings.formulae.trivia.volume(radius=r_dry)
        attributes["multiplicity"] = np.append(
            attributes["multiplicity"], n_per_volume * volume
        )
        attributes["dry volume"] = np.append(attributes["dry volume"], v_dry)
        attributes["kappa times dry volume"] = np.append(
            attributes["kappa times dry volume"], v_dry * kappa
        )
        attributes["kappa"] = np.append(
            attributes["kappa"], np.full(settings.n_sd, kappa)
        )
        r_wet = equilibrate_wet_radii(
            r_dry=settings.formulae.trivia.radius(volume=attributes["dry volume"]),
            environment=env,
            kappa_times_dry_volume=attributes["kappa times dry volume"],
        )
        attributes["volume"] = settings.formulae.trivia.volume(radius=r_wet)

        super().__init__(
            particulator=builder.build(attributes=attributes, products=products)
        )
        if scipy_solver:
            scipy_ode_condensation_solver.patch_particulator(self.particulator)

        self.output_attributes = {
            "volume": tuple([] for _ in range(self.particulator.n_sd)),
            "dry volume": tuple([] for _ in range(self.particulator.n_sd)),
            "critical supersaturation": tuple(
                [] for _ in range(self.particulator.n_sd)
            ),
            "equilibrium supersaturation": tuple(
                [] for _ in range(self.particulator.n_sd)
            ),
            "critical volume": tuple([] for _ in range(self.particulator.n_sd)),
            "multiplicity": tuple([] for _ in range(self.particulator.n_sd)),
        }
        self.settings = settings

        # self.__sanity_checks(attributes, volume)

    def __sanity_checks(self, attributes, volume):
        for attribute in attributes.values():
            assert attribute.shape[0] == self.particulator.n_sd
        np.testing.assert_approx_equal(
            sum(attributes["multiplicity"]) / volume,
            sum(
                mode.norm_factor
                for mode in self.settings.aerosol_modes_by_kappa.values()
            ),
            significant=4,
        )

    def _save(self, output):
        for key, attr in self.output_attributes.items():
            attr_data = self.particulator.attributes[key].to_ndarray()
            for drop_id in range(self.particulator.n_sd):
                attr[drop_id].append(attr_data[drop_id])
        super()._save(output)

    def run(self):
        output_products = super()._run(
            self.settings.nt, self.settings.steps_per_output_interval
        )
        return {"products": output_products, "attributes": self.output_attributes}
