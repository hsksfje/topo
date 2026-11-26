import jax.numpy as jnp
import pytest
from jax import config, vmap
from jax.test_util import check_grads
from metatopia.dataset.to.validation_problems import mbb_beam
from metatopia.solver import setup_solver_cvxopt
from omegaconf import DictConfig


config.update("jax_enable_x64", True)


@pytest.fixture
def mbb_config_and_problem():
    """Basic configuration for testing"""
    width, height = 128, 64

    config = DictConfig(
        {
            "young": 1.0,
            "young_min": 1e-9,
            "poisson": 0.3,
            "Lx": 2.0,
            "Ly": 1.0,
            "Nx": width,
            "Ny": height,
            "penalty": 3.0,
        }
    )

    problem = mbb_beam(width, height, 0.5)
    return problem, config


def get_test_rhos(config, batch_size=10):
    """Helper function to generate test density arrays"""
    domain_shape = (config.Ny, config.Nx)
    rhos = jnp.ones((batch_size, *domain_shape))
    vfs = jnp.linspace(0.1, 1.0, batch_size)
    return rhos * vfs[:, None, None]


def setup_test_solver(config, problem):
    """Helper function to set up solver with common configuration"""
    solver = setup_solver_cvxopt(**config)
    vmapped_solver = vmap(solver, in_axes=(0, None))
    return solver, vmapped_solver


def test_solver_forward(mbb_config_and_problem):
    """Test forward pass of the solver"""
    problem, config = mbb_config_and_problem
    _, vmapped_solver = setup_test_solver(config, problem)
    rhos = get_test_rhos(config)

    compliances = vmapped_solver(rhos, problem)

    assert compliances.shape == (rhos.shape[0],), "Output shape is incorrect"
    assert jnp.all(compliances > 0.0), "Compliance should be positive"
    assert jnp.all(
        jnp.diff(compliances) < 0.0
    ), "Compliance should decrease with volume fraction"


def test_solver_gradients(mbb_config_and_problem):
    """Test gradients of the solver"""
    problem, config = mbb_config_and_problem
    solver, vmapped_solver = setup_test_solver(config, problem)
    rhos = get_test_rhos(config)

    # Define compliance functions
    compliance_fn = lambda rho: solver(rho, problem)
    mean_compliance_fn = lambda rhos: jnp.mean(vmapped_solver(rhos, problem))

    check_grads(
        compliance_fn,
        (rhos[-1],),
        modes="rev",
        order=1,
        atol=1e-3,
        rtol=1e-3,
    )

    check_grads(
        mean_compliance_fn,
        (rhos,),
        modes="rev",
        order=1,
        atol=1e-3,
        rtol=1e-3,
    )
