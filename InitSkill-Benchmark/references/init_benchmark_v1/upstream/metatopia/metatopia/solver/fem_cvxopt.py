from typing import Callable

import cvxopt
import cvxopt.cholmod
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, PyTree


def get_stiffness_matrix(young, poisson, Lx, Ly, Nx, Ny):
    H = Ly / Ny
    W = Lx / Nx

    # Element stiffness matrix (scaled)
    E, nu = young, poisson

    k = np.array(
        [
            12 * H**2 - 6 * W**2 * nu + 6 * W**2,
            4.5 * H * W * (1 + nu),
            -12 * H**2 - 3 * W**2 * nu + 3 * W**2,
            36 * H * W * (-0.125 + 0.375 * nu),
            -6 * H**2 + 3 * W**2 * nu - 3 * W**2,
            -4.5 * H * W * (1 + nu),
            6 * H**2 + 6 * W**2 * nu - 6 * W**2,
            36 * H * W * (0.125 - 0.375 * nu),
        ]
    )

    return (
        E
        / (36 * W * H * (1 - nu**2))
        * np.array(
            [
                [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
            ]
        )
    )


def young_modulus(x, E_0, E_min, p=3):
    return E_min + x**p * (E_0 - E_min)


def get_B_matrix(Lx, Ly, Nx, Ny):
    H = Ly / Ny
    W = Lx / Nx

    # Strain-displacement matrix (B) - this is the proper scaling!
    B = (
        1
        / (H * W)
        * np.array(
            [
                [-H, 0, H, 0, H, 0, -H, 0],
                [0, -W, 0, -W, 0, W, 0, W],
                [-W, -H, -W, H, W, H, W, -H],
            ]
        )
    )

    return B


def get_k(stiffness, ke):
    # Constructs a sparse stiffness matrix, k, for use in the displace function.
    Ny, Nx = stiffness.shape

    # get position of the nodes of each element in the stiffness matrix
    ely, elx = np.meshgrid(np.arange(Ny), np.arange(Nx))  # x, y coords
    ely, elx = ely.reshape(-1, 1), elx.reshape(-1, 1)

    n1 = (Ny + 1) * (elx + 0) + (ely + 0)
    n2 = (Ny + 1) * (elx + 1) + (ely + 0)
    n3 = (Ny + 1) * (elx + 1) + (ely + 1)
    n4 = (Ny + 1) * (elx + 0) + (ely + 1)
    edof = np.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]
    )
    edof = edof.T[0]

    x_list = np.repeat(edof, 8)  # flat list pointer of each node in an element
    # flat list pointer of each node in elem
    y_list = np.tile(edof, 8).flatten()

    # make the stiffness matrix
    kd = stiffness.T.reshape(Nx * Ny, 1, 1)
    value_list = (kd * np.tile(ke, kd.shape)).flatten()
    return value_list, y_list, x_list


def inverse_permutation(indices):
    inverse_perm = np.zeros(len(indices), dtype=np.int64)
    inverse_perm[indices] = np.arange(len(indices), dtype=np.int64)
    return inverse_perm


def _get_dof_indices(freedofs, fixdofs, k_xlist, k_ylist):
    index_map = inverse_permutation(np.concatenate([freedofs, fixdofs]))
    keep = np.isin(k_xlist, freedofs) & np.isin(k_ylist, freedofs)
    i = index_map[k_ylist][keep]
    j = index_map[k_xlist][keep]
    return index_map, keep, np.stack([i, j])


def displace_cvxopt(
    x_phys, ke, forces, freedofs, fixdofs, *, penalty=3, E_min=1e-9, E_0=1.0
):
    # Displaces the load x using finite element techniques. The spsolve here
    # occupies the majority of this entire simulation's runtime.
    stiffness = young_modulus(x_phys, E_0, E_min, p=penalty)
    k_entries, k_ylist, k_xlist = get_k(stiffness, ke)

    index_map, keep, indices = _get_dof_indices(freedofs, fixdofs, k_ylist, k_xlist)

    K = cvxopt.spmatrix(k_entries[keep], indices[0, :], indices[1, :])
    B = cvxopt.matrix(forces[freedofs])

    cvxopt.cholmod.linsolve(K, B)
    u_nonzero = np.array(B)
    u_values = np.concatenate([u_nonzero[:, 0], np.zeros(len(fixdofs))])

    return u_values[index_map]


def get_stress(x_phys, u, B, young, young_min, poisson, penalty):
    # Index map
    Ny, Nx = x_phys.shape
    ely, elx = np.meshgrid(np.arange(Ny), np.arange(Nx))

    # Quad nodes
    n1 = (Ny + 1) * (elx + 0) + (ely + 0)
    n2 = (Ny + 1) * (elx + 1) + (ely + 0)
    n3 = (Ny + 1) * (elx + 1) + (ely + 1)
    n4 = (Ny + 1) * (elx + 0) + (ely + 1)

    all_ixs = np.array(
        [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1, 2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]
    )

    # Get element displacements
    u_selected = u[all_ixs]
    u_selected = u_selected.reshape(8, Nx, Ny)

    # SIMP interpolation for Young's modulus (elementwise moduli)
    E = young_min + x_phys**penalty * (young - young_min)

    # Constitutive matrix (D)
    d = (
        1.0
        / (1.0 - poisson**2)
        * np.array(
            [
                [1.0, poisson, 0.0],
                [poisson, 1.0, 0.0],
                [0.0, 0.0, (1.0 - poisson) / 2.0],
            ]
        )
    )

    # Calculate stresses using einsum operations
    D = np.einsum("ij,kl->ijkl", E, d)  # Material matrix for each element
    T = np.einsum("ijkl,lm->ijkm", D, B)  # Combine with B matrix
    stress = np.einsum("ijkl, lji->kij", T, u_selected)  # Calculate element stresses

    return stress

def compliance(x_phys, u, ke):
    # index map
    Ny, Nx = x_phys.shape
    ely, elx = np.meshgrid(np.arange(Ny), np.arange(Nx))  # x, y coords

    # nodes
    n1 = (Ny + 1) * (elx + 0) + (ely + 0)
    n2 = (Ny + 1) * (elx + 1) + (ely + 0)
    n3 = (Ny + 1) * (elx + 1) + (ely + 1)
    n4 = (Ny + 1) * (elx + 0) + (ely + 1)

    all_ixs = np.array(
        [
            2 * n1,
            2 * n1 + 1,
            2 * n2,
            2 * n2 + 1,
            2 * n3,
            2 * n3 + 1,
            2 * n4,
            2 * n4 + 1,
        ]
    )

    # select from u matrix
    u_selected = u[all_ixs]

    # compute x^penalty * U.T @ ke @ U in a vectorized way
    ke_u = np.einsum("ij,jkl->ikl", ke, u_selected)
    ce = np.einsum("ijk,ijk->jk", u_selected, ke_u)
    return ce.T


def setup_solver_cvxopt(
    young, young_min, poisson, Lx, Ly, Nx, Ny, penalty, **kwargs
) -> Callable[[Float[Array, "Ny Nx"], PyTree], Float[Array, ""]]:
    """
    Sets up a function which calculates the compliance for input design
    variables, and problem setup using the SIMP (Solid Isotropic
    Material with Penalization) method and JAX's custom gradient
    capabilities.

    Parameters
    ----------
    young : float
        Young's modulus for solid elements.
    young_min : float
        Minimum Young's modulus for void elements.
    poisson : float
        Poisson's ratio of the material.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.
    Nx : int
        Number of elements in the x-direction.
    Ny : int
        Number of elements in the y-direction.
    penalty : float
        Penalization factor for the SIMP method (typically between 3-5).
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    compliance_loss_jax : callable
        A JAX-compatible function that computes the compliance loss for
        a given design variable configuration and problem setup.

    """
    ke = get_stiffness_matrix(young, poisson, Lx, Ly, Nx, Ny)

    def compliance_fn_cvxopt(x, normals, forces):
        x = np.asarray(x)

        normals = np.ravel(normals)
        forces = np.ravel(forces)
        fixdofs = np.flatnonzero(normals.ravel())
        alldofs = np.arange(2 * (Nx + 1) * (Ny + 1))
        freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

        u = displace_cvxopt(
            x,
            ke,
            forces,
            freedofs,
            fixdofs,
            penalty=penalty,
            E_min=young_min,
            E_0=young,
        )

        ce = compliance(x, u, ke)

        return ce

    @jax.custom_jvp
    def compliance_fn(x, problem):
        def _compliance_fn_host(x, normals, forces, id):
            try:
                result = compliance_fn_cvxopt(x, normals, forces)
            except ArithmeticError:
                # Skip the result if there is an error
                result = np.zeros_like(x)
                print(
                    f"Skipping problem {id} due to arithmetic error",
                )

            return result.astype(x.dtype)

        result_shape_dtypes = jax.ShapeDtypeStruct(
            jnp.broadcast_shapes(x.shape), x.dtype
        )

        return jax.pure_callback(
            _compliance_fn_host,
            result_shape_dtypes,
            x,
            problem["normals"],
            problem["forces"],
            problem["id"],
            vmap_method="sequential",
        )

    @compliance_fn.defjvp
    def f_jvp(primals, tangents):
        (x, problem) = primals
        (x_dot, p_dot) = tangents
        ce = compliance_fn(x, problem)  # Primal out
        return ce, ce

    # Variables aliases
    E_min = young_min
    E_0 = young
    p = penalty

    def simp(x):
        return E_min + x ** (p) * (E_0 - E_min)

    @jax.custom_vjp
    def compliance_loss_jax(x, problem):
        ce = compliance_fn(x, problem)
        c = ce * simp(x)  # SIMP law
        return jnp.sum(c)

    def compliance_fwd(x, problem):
        ce = compliance_fn(x, problem)
        c = ce * simp(x)
        return jnp.sum(c), (x, ce)

    def compliance_bwd(res, g):
        x, ce = res
        c = -ce * p * (x ** (p - 1) * (E_0 - E_min))
        return (g * c, None)

    compliance_loss_jax.defvjp(compliance_fwd, compliance_bwd)
    return compliance_loss_jax


def setup_solver_cvxopt_elementwise(
    young, young_min, poisson, Lx, Ly, Nx, Ny, penalty, **kwargs
):
    """
    Analogous to setup_solver_cvxopt, but returns elementwise compliances.

    Parameters
    ----------
    young : float
        Young's modulus for solid elements.
    young_min : float
        Minimum Young's modulus for void elements.
    poisson : float
        Poisson's ratio of the material.
    Lx : float
        Length of the domain in the x-direction.
    Ly : float
        Length of the domain in the y-direction.
    Nx : int
        Number of elements in the x-direction.
    Ny : int
        Number of elements in the y-direction.
    penalty : float
        Penalization factor for the SIMP method (typically between 3-5).
    **kwargs : dict
        Additional keyword arguments.



    """
    # Create the element stiffness matrix
    ke = get_stiffness_matrix(young, poisson, Lx, Ly, Nx, Ny)

    def compliance_fn_elementwise(x, normals, forces):
        x = np.asarray(x)
        normals = np.ravel(normals)
        forces = np.ravel(forces)
        fixdofs = np.flatnonzero(normals.ravel())
        alldofs = np.arange(2 * (Nx + 1) * (Ny + 1))
        freedofs = np.sort(list(set(alldofs) - set(fixdofs)))

        u = displace_cvxopt(
            x,
            ke,
            forces,
            freedofs,
            fixdofs,
            penalty=penalty,
            E_min=young_min,
            E_0=young,
        )
        ce = compliance(x, u, ke)
        return ce

    def compliance_fn(x, problem):
        def _compliance_fn_host(x, normals, forces, id):
            try:
                result = compliance_fn_elementwise(x, normals, forces)
            except ArithmeticError:
                result = np.ones_like(x)
                print(
                    f"Skipping problem {id} due to arithmetic error",
                )

            return result

        result_shape_dtypes = jax.ShapeDtypeStruct(
            jnp.broadcast_shapes(x.shape), x.dtype
        )

        return jax.pure_callback(
            _compliance_fn_host,
            result_shape_dtypes,
            x,
            problem["normals"],
            problem["forces"],
            problem["id"],
            vmap_method="sequential",
        )

        # Variables aliases

    E_min = young_min
    E_0 = young
    p = penalty

    simp = lambda x: E_min + x ** (p) * (E_0 - E_min)

    def compliance_loss_elementwise(x, problem):
        ce = compliance_fn(x, problem)
        c = ce * simp(x)  # SIMP law
        return jnp.sum(c), ce

    return compliance_loss_elementwise
