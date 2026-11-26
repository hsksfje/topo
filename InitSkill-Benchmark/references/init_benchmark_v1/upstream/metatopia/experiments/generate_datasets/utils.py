from functools import partial

import numpy as np
import numpy.random as npr


def random_nodes(n, Ny, Nx, p_edge=10.0, p_matrix=None, r_scale=0.5):
    shape = (Nx + 1, Ny + 1)
    n_max = shape[0] * shape[1]
    node_args = np.arange(n_max)

    if p_matrix is None:
        p_matrix = np.full(shape=shape, fill_value=p_edge)
        p_matrix[1:-1, 1:-1] = 1.0

    coords = np.indices(np.array(shape))  # .T
    coords = coords.reshape(2, -1) / np.array([Nx, Ny]).reshape(2, 1) - 0.5

    coords = coords.T
    nodes = []

    for _ in range(n):
        p_matrix /= p_matrix.sum()
        probs = p_matrix.ravel()
        node_i = npr.choice(node_args, size=1, replace=False, p=probs)
        nodes.append(node_i)
        R = 0.5 / np.sqrt(n)
        rbf = np.exp(
            -np.sum((coords - coords[node_i]) ** 2, axis=1) / (r_scale * R) ** 2
        )
        rbf = rbf.reshape(shape)
        p_matrix = p_matrix * (1 - rbf)

    nodes = np.array(nodes)

    return nodes, p_matrix


def find_closest_grid_coord(point, grid_coords):
    distances = np.linalg.norm(grid_coords - point, axis=1)
    closest_index = np.argmin(distances)
    return grid_coords[closest_index]


def rbff(coords, point, r_scale):
    R = 0.5 / np.sqrt(2)
    rbf = np.exp(
        -np.sum((coords[None, :, :] - point[:, None, :]) ** 2, axis=-1)
        / (r_scale * R) ** 2
    )
    return rbf


def random_line(
    n,
    Ny,
    Nx,
    p_edge=10.0,
    p_matrix=None,
    r_scale=0.5,
    distributed_bcs_angles=[0, np.pi / 2, np.pi],
):
    shape = (Nx + 1, Ny + 1)
    n_max = shape[0] * shape[1]
    node_args = np.arange(n_max)

    Nn = np.max(shape)

    if p_matrix is None:
        p_matrix = np.full(shape=shape, fill_value=p_edge)
        p_matrix[1:-1, 1:-1] = 1.0

    coords = np.indices(np.array(shape))  # .T
    coords = coords.reshape(2, -1) / np.array([Nx, Ny]).reshape(2, 1) - 0.5
    coords = coords.T
    nodes = []

    def angle_filter(node_coords):
        pmat = np.zeros_like(p_matrix)
        if distributed_bcs_angles is not None:
            for angle in distributed_bcs_angles:
                node_idx = (node_coords[0] + 0.5) * np.array([Nx, Ny])
                if angle < np.pi / 4:
                    dx = 1
                    dy = np.tan(angle)

                else:
                    dx = 1 / np.tan(angle)
                    dy = 1
                for j in np.arange(-Nn, Nn):
                    carry_prob = 1.0
                    xi = np.round(j * dx) + node_idx[0]
                    yi = np.round(j * dy) + node_idx[1]
                    if xi >= 0 and xi <= Nx and yi >= 0 and yi <= Ny:
                        # added to avoid intersections
                        carry_prob = np.minimum(carry_prob, p_matrix[int(xi), int(yi)])
                        pmat[int(xi), int(yi)] = np.minimum(1, carry_prob)
        else:
            pmat = np.ones_like(p_matrix)

        return pmat

    p_line = p_matrix.copy()
    for _ in range(2):
        p_line /= p_line.sum()
        probs = p_line.ravel()
        node_i = npr.choice(node_args, size=1, replace=False, p=probs)
        nodes.append(node_i)
        R = 0.5 / np.sqrt(2)
        rbf = np.exp(
            -np.sum((coords - coords[node_i]) ** 2, axis=1) / (r_scale * R) ** 2
        )
        rbf = rbf.reshape(shape)
        p_matrix = p_matrix * (1 - rbf)
        p_line = p_line * (1 - rbf)
        pang = angle_filter(coords[node_i])
        p_line = p_line * pang * 100

    c1, c2 = coords[nodes]
    a = np.linspace(0, 1, 2 * Nx)
    interpolated = c1 * a[:, None] + (1.0 - a)[:, None] * c2
    interp_coords = np.zeros_like(interpolated)
    for i, p in enumerate(interpolated):
        closest = find_closest_grid_coord(p, coords)
        interp_coords[i] = closest

    interp_coords = np.unique(interp_coords, axis=0)

    for c in interp_coords:
        nodes.append(np.array([np.argwhere(np.all(coords == c, axis=1))[0][0]]))

    nodes = np.unique(np.array(nodes), axis=0)
    node_coords = coords[nodes]

    rbfs = rbff(coords, node_coords, r_scale)[:, 0, :]
    rbfs = np.max(rbfs, axis=0)

    p_matrix = p_matrix * (1 - rbfs).reshape(shape)
    return nodes, p_matrix


def random_problem(
    Nx=64,
    Ny=64,
    target_vf=0.5,
    r_scale=0.5,
    p_edge=100.0,
    distributed=False,
    n_distributed_poisson=2,
    min_n_distributed=1,
    distributed_bcs_angles=[0, np.pi / 2, np.pi / 4],
):
    """
    Generates a random problem with a random mask, random supports and
    random point loads

    Parameters
    ----------
    Nx : int, optional
        Nx of the domain. The default is 60.
    Ny : int, optional
        Ny of the domain. The default is 60.
    target_vf : float, optional
        Prescribed volume fraction

    r_scale: float, optional
        scale of the radial basis function, that controls how
        close the forces/supports can be to each other

    distributed: bool, optional
        if True, the boundary conditions are distributed along lines
    min_n_distributed: int, optional
        minimum number of distributed boundary conditions
    n_distributed_poisson: int, optional
        Poisson distribution parameter, tha is used to determine the number of
        distributed boundary conditions


    Note
    ----
    OOD dataset uses n_distributed_poisson=2,
    min_n_distributed=1,
    distributed_bcs_angles=[0, np.pi / 2, np.pi / 4],

    """

    # iniitalize arrays for Normals(supports) and Forces
    # non-zero value means that there is a support or force

    normals = np.zeros((Nx + 1, Ny + 1, 2))
    normals = normals.reshape(-1, 2)
    forces = np.zeros((Nx + 1, Ny + 1, 2))
    forces = forces.reshape(-1, 2)

    # Select one of the two options:
    sopt = npr.rand(1) < 0.5

    if sopt:
        n_xy_nodes = max(1, npr.poisson(1))
        n_x_or_y_nodes = max(1, npr.poisson(1))
        n_supports = n_xy_nodes + n_x_or_y_nodes
    else:
        nodes_dir1 = max(2, npr.poisson(1))
        nodes_dir2 = max(1, npr.poisson(1))
        n_supports = nodes_dir1 + nodes_dir2

    n_load = max(1, npr.poisson(2))
    n_bcs = n_supports + n_load

    if distributed:
        n_line_bcs = min(
            max(min_n_distributed, npr.poisson(n_distributed_poisson)),
            n_bcs,
        )
    else:
        n_line_bcs = 0

    is_distributed = npr.choice(np.arange(n_bcs), n_line_bcs, replace=False)

    p_matrix = np.ones((Nx + 1, Ny + 1)) * p_edge
    p_matrix[1:-1, 1:-1] = 1.0

    dist_p_counter = 0

    if sopt:
        for _ in range(n_xy_nodes):
            if dist_p_counter in is_distributed:
                nodes_fn = partial(
                    random_line, distributed_bcs_angles=distributed_bcs_angles
                )
            else:
                nodes_fn = random_nodes

            dist_p_counter += 1
            xy_node, p_matrix = nodes_fn(
                1,
                Ny,
                Nx,
                p_matrix=p_matrix,
                p_edge=p_edge,
                r_scale=r_scale,
            )

            normals[:, 0][xy_node] = 1.0
            normals[:, 1][xy_node] = 1.0

        for _ in range(n_x_or_y_nodes):
            if dist_p_counter in is_distributed:
                nodes_fn = partial(
                    random_line, distributed_bcs_angles=distributed_bcs_angles
                )
            else:
                nodes_fn = random_nodes

            dist_p_counter += 1
            x_or_y_node, p_matrix = nodes_fn(
                1,
                Ny,
                Nx,
                p_matrix=p_matrix,
                p_edge=p_edge,
                r_scale=r_scale,
            )

            if npr.rand(1) > 0.5:
                normals[:, 0][x_or_y_node] = 1.0
            else:
                normals[:, 1][x_or_y_node] = 1.0
    else:
        # at least 2 ux + uy or 2uy + ux
        dir1 = npr.choice([0, 1])

        for _ in range(nodes_dir1):
            if dist_p_counter in is_distributed:
                nodes_fn = partial(
                    random_line, distributed_bcs_angles=distributed_bcs_angles
                )
            else:
                nodes_fn = random_nodes

            dist_p_counter += 1
            node_dir1, p_matrix = nodes_fn(
                1,
                Ny,
                Nx,
                p_matrix=p_matrix,
                p_edge=p_edge,
                r_scale=r_scale,
            )

            normals[:, dir1][node_dir1] = 1.0

        for _ in range(nodes_dir2):
            if dist_p_counter in is_distributed:
                nodes_fn = partial(
                    random_line, distributed_bcs_angles=distributed_bcs_angles
                )
            else:
                nodes_fn = random_nodes
            dist_p_counter += 1

            node_dir2, p_matrix = nodes_fn(
                1,
                Ny,
                Nx,
                p_matrix=p_matrix,
                p_edge=p_edge,
                r_scale=r_scale,
            )

            normals[:, 1 - dir1][node_dir2] = 1.0

    # Random loads:

    for _ in range(n_load):
        random_angle = npr.uniform(0, 2 * np.pi)

        if dist_p_counter in is_distributed:
            nodes_fn = partial(
                random_line, distributed_bcs_angles=distributed_bcs_angles
            )
        else:
            nodes_fn = random_nodes

        dist_p_counter += 1
        load_nodes, p_matrix = nodes_fn(
            1,
            Ny,
            Nx,
            p_matrix=p_matrix,
            p_edge=p_edge,
            r_scale=r_scale,
        )

        forces[:, 0][load_nodes] = np.cos(random_angle)
        forces[:, 1][load_nodes] = np.sin(random_angle)

    normals = normals.reshape(Nx + 1, Ny + 1, 2)
    forces = forces.reshape(Nx + 1, Ny + 1, 2)

    # # create 5 subplots to visualize mask, forces and supports
    strain_energy = None
    id = None

    nb = np.any(normals > 0, axis=2)
    fb = np.any(forces > 0, axis=2)

    # Recursively call the function if the supports and forces overlap
    # [TODO] This could potentially be an infinite loop - add a guard
    overlaps = np.logical_and(fb, nb).any()

    if overlaps:
        return random_problem(
            Nx,
            Ny,
            target_vf,
            r_scale,
            p_edge,
            distributed,
            n_distributed_poisson,
            min_n_distributed,
            distributed_bcs_angles,
        )
    else:
        return {
            "normals": normals,
            "forces": forces,
            "target_vf": target_vf,
            "strain_energy": strain_energy,
            "id": id,
        }
