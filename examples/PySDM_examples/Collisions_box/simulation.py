import PySDM.products.size_spectral.arbitrary_moment as am
from PySDM import Formulae
from PySDM.backends import CPU
from PySDM.builder import Builder
from PySDM.dynamics import Collision
from PySDM.environments import Box
from PySDM.initialisation.sampling.spectral_sampling import ConstantMultiplicity
from PySDM.products import CollisionRatePerGridbox, MeanRadius, ParticleConcentration

TRIVIA = Formulae().trivia


def make_core(settings, coal_eff):
    backend = CPU

    builder = Builder(n_sd=settings.n_sd, backend=backend(settings.formulae))
    env = Box(dv=settings.dv, dt=settings.dt)
    builder.set_environment(env)
    env["rhod"] = 1.0
    attributes = {}
    attributes["volume"], attributes["multiplicity"] = ConstantMultiplicity(
        settings.spectrum
    ).sample(settings.n_sd)

    # attributes["volume"] = TRIVIA.volume(np.asarray(settings.equilibrate_wet_radii(TRIVIA.radius(np.asarray(attributes["dry volume"])))))
    # attributes["kappa times dry volume"] = settings.kappa[0]*np.asarray(attributes["dry volume"])
    # attributes["dry volume organic"] = np.zeros(attributes["dry volume"].shape)

    collision = Collision(
        collision_kernel=settings.kernel,
        coalescence_efficiency=coal_eff,
        breakup_efficiency=settings.break_eff,
        fragmentation_function=settings.fragmentation,
        adaptive=settings.adaptive,
    )
    builder.add_dynamic(collision)
    M0 = am.make_arbitrary_moment_product(rank=0, attr="volume", attr_unit="m^3")
    M1 = am.make_arbitrary_moment_product(rank=1, attr="volume", attr_unit="m^3")
    M2 = am.make_arbitrary_moment_product(rank=2, attr="volume", attr_unit="m^3")
    products = (
        M0(name="M0"),
        M1(name="M1"),
        M2(name="M2"),
        ParticleConcentration(name="n_part", stp=False),
        MeanRadius(name="r_mean"),
        CollisionRatePerGridbox(name="collision_rate"),
    )

    return builder.build(attributes, products)
