from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import markers
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pandas import DataFrame


plt.set_loglevel("ERROR")


MM = 0.0393701  # millimeter in inches (1 mm = 0.0393701 in)

SMALL_FIG = 80 * MM
MEDIUM_FIG = 120 * MM
BIG_FIG = 180 * MM


SMALL_SQUARE = (SMALL_FIG, SMALL_FIG)
MEDIUM_SQUARE = (MEDIUM_FIG, MEDIUM_FIG)
BIG_SQUARE = (BIG_FIG, BIG_FIG)

SMALL_GOLDEN = (SMALL_FIG, (10.0 / 16.0) * SMALL_FIG)
MEDIUM_GOLDEN = (MEDIUM_FIG, (10.0 / 16.0) * MEDIUM_FIG)
BIG_GOLDEN = (BIG_FIG, (10.0 / 16.0) * BIG_FIG)

DPI = 300


# Matplotlib settings
mpl.rcdefaults()

mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.labelsize"] = 7
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["pdf.fonttype"] = 42

# Set line widths
mpl.rcParams["axes.linewidth"] = 0.5
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["lines.linewidth"] = 1.0
mpl.rcParams["lines.markersize"] = 3

# Seaborn settings
palette = sns.color_palette("Dark2")
sns.set_palette(palette)


###############################################################################
def save_figure(fig: Figure, save_path: str, save_name: str, dpi: int = DPI):
    for ext in ["pdf", "png"]:
        path = Path(save_path) / f"{save_name}.{ext}"
        fig.savefig(path, dpi=dpi, format=ext, transparent=(ext == "pdf"))


###############################################################################
def plot_meta_losses(
    meta_losses: np.ndarray,
    *,  # Force keyword arguments for clarity
    window_size: int = 100,
    clip_percentile: int = 99,
    figsize: Tuple[float, float] = MEDIUM_SQUARE,
    title: Optional[str] = None,
    legend: Optional[bool] = True,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    dpi: int = DPI,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    else:
        fig = ax.figure

    colors = sns.color_palette()

    # Ensure meta_losses is at least a 2D array
    if meta_losses.ndim == 1:
        meta_losses = jnp.expand_dims(meta_losses, axis=0)

    n_iters = meta_losses.shape[1]

    def _moving_avg(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    for meta_loss, color in zip(meta_losses, colors):
        upper_bound = np.percentile(meta_loss, clip_percentile)
        meta_loss = np.clip(meta_loss, None, upper_bound)  # Filter out outliers
        ax.plot(
            meta_loss,
            label="Meta-loss",
            linewidth=1,
            alpha=0.5,
            color=color,
        )

        if n_iters >= 2 * window_size:
            # Calculate moving average
            ma_values = _moving_avg(meta_loss, window_size)
            ma_values = np.concatenate(([np.nan] * (window_size - 1), ma_values))

            ax.plot(
                ma_values,
                label=f"Moving average (window size: {window_size})",
                linewidth=2,
                color=color,
            )

    ax.set_xlim(0, len(meta_loss) - 1)
    ax.set_ylim(0.0, 1.1 * meta_loss[0])
    ax.set_xlabel("Outer loop iteration")
    ax.set_ylabel("Meta-loss")

    if title:
        ax.set_title(title)

    if legend:
        fig.legend(loc="outside lower center", ncol=2, frameon=False)

    if save_path and save_name:
        save_figure(fig, save_path, save_name)
    return fig, ax


###############################################################################
def calculate_performance_ratios(
    results_df: DataFrame,
    solver_names: List[str],
    metric: Literal["steps", "loss", "bw_loss"],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Calculate performance ratios for solvers.

    Args:
        results_df: DataFrame containing solver results
        solver_names: List of solver names to compare
        metric: Metric to analyze (e.g., 'steps', 'loss')

    Returns:
        unique_taus: Sorted unique tau values
        solver_taus: Performance profile values for each solver
        solvers: List of solver names found in the data
    """
    # Get relevant columns
    solvers = [
        col
        for col in results_df.columns
        if col.startswith(metric) and any(name in col for name in solver_names)
    ]

    # Sort alphabetically
    solvers = sorted(solvers)

    # Calculate ratios
    best_metrics = results_df[solvers].min(axis=1)
    performance_ratios = results_df[solvers].div(best_metrics, axis=0)

    # Calculate profile values
    unique_taus = np.sort(np.unique(performance_ratios.values))
    solver_taus = np.zeros((len(solvers), len(unique_taus)))

    for i, tau in enumerate(unique_taus):
        fraction_solved = (performance_ratios <= tau).mean(axis=0)
        solver_taus[:, i] = fraction_solved

    return unique_taus, solver_taus, solvers


###############################################################################
def plot_performance_profiles(
    results_df: DataFrame,
    solver_names: List[str],
    metric: Literal["steps", "loss", "bw_loss"],
    *,
    dataset_name: Optional[str] = None,
    x_limit: float = 2.0,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[float, float] = MEDIUM_SQUARE,
    ax: Optional[Axes] = None,
    legend: Optional[bool] = True,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(
            figsize=figsize, subplot_kw=dict(box_aspect=1), layout="constrained"
        )
    else:
        fig = ax.figure

    colors = sns.color_palette()

    unique_taus, solver_taus, solvers = calculate_performance_ratios(
        results_df, solver_names, metric
    )

    for i, solver in enumerate(solvers):
        ax.step(
            unique_taus,
            solver_taus[i, :],
            where="post",
            color=colors[i],
            linewidth=1.5,
            label=solver,
        )

    dataset_name = dataset_name if dataset_name else "dataset"

    ax.set_yticks(np.arange(0, 1 + 0.1, step=0.1), minor=False)
    # ax.set_xticks(np.arange(1, x_limit + 0.1, step=0.1), minor=False)
    ax.set_xlim(1.0, x_limit)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Tolerance", fontsize=7)
    ax.set_ylabel("Fraction of test problems", fontsize=7)

    if legend:
        fig.legend(loc="outside lower center", ncols=len(solvers), frameon=False)

    if title:
        ax.set_title(title)

    if save_path and save_name:
        save_figure(save_path, save_name)

    return fig, ax


###############################################################################
def _align_marker(marker, halign="center", valign="middle"):
    if isinstance(halign, (str)):
        halign = {
            "right": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "left": 1.0,
        }[halign]

    if isinstance(valign, (str)):
        valign = {
            "top": -1.0,
            "middle": 0.0,
            "center": 0.0,
            "bottom": 1.0,
        }[valign]

    bm = markers.MarkerStyle(marker)

    m_arr = bm.get_path().transformed(bm.get_transform()).vertices

    m_arr[:, 0] += halign / 2
    m_arr[:, 1] += valign / 2

    return mpl.path.Path(m_arr, bm.get_path().codes)


def plot_design(
    rho,
    problem: Dict[str, np.ndarray] = None,
    *,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[float, float] = BIG_SQUARE,
    ax: Optional[Axes] = None,
    legend: Optional[bool] = False,
    colorbar: Optional[bool] = True,
    title: Optional[str] = None,
    vmin: Optional[float] = 0,
    vmax: Optional[float] = 1,
    cmap: Optional[str] = "Grays",
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    colors = sns.color_palette()

    # Get design dimensions
    Ny, Nx = rho.shape

    # Coordinate grid
    x = np.arange(Nx + 1)
    y = np.arange(Ny + 1)

    # Plot the design
    im = ax.pcolormesh(
        x,
        y,
        rho,
        antialiased=True,
        vmin=vmin,
        vmax=vmax,
        zorder=0,
        cmap=cmap,
        edgecolors="None",
        rasterized=True,
    )

    ax.invert_yaxis()

    # Add margins around the figure

    if problem is not None:
        rect = plt.Rectangle(
            (0, 0), Nx, Ny, fill=False, color="black", linewidth=1, zorder=1
        )
        ax.add_patch(rect)
        ax.use_sticky_edges = False
        ax.margins(0.1)

        # Remove outer spines
        for spine in ax.spines.values():
            spine.set_visible(False)

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.3)

    if problem is None:
        # Legend is not needed when plotting the design only
        legend = False

    if problem is not None:
        forces = problem["forces"]
        normals = problem["normals"]

        # Design and boundary conditions might be defined on a different grid
        sy = Ny / (forces.shape[1] - 1)
        sx = Nx / (forces.shape[0] - 1)

        # Add forces using quiver
        force_indices = np.argwhere(np.any(forces != 0, axis=2))
        scaled_positions = force_indices * [sy, sx]

        if len(force_indices) > 0:
            Fx = forces[force_indices[:, 0], force_indices[:, 1], 0]
            Fy = forces[force_indices[:, 0], force_indices[:, 1], 1]

            F = jnp.sqrt(Fx**2 + Fy**2)

            ax.quiver(
                scaled_positions[:, 0],
                scaled_positions[:, 1],
                Fx / F.max(),
                Fy / F.max(),
                units="xy",
                scale=10 / Nx,  # inverse of fraction of the design grid
                width=0.75,
                color=colors[1],
                linewidths=0,
                zorder=3,
                joinstyle="round",
            )

        # Add supports as scatter points
        x_support_indices = np.argwhere(normals[:, :, 0] != 0)
        scaled_x_supports = x_support_indices * [sy, sx]
        y_support_indices = np.argwhere(normals[:, :, 1] != 0)
        scaled_y_supports = y_support_indices * [sy, sx]

        if len(x_support_indices) > 0:
            ax.scatter(
                scaled_x_supports[:, 0],
                scaled_x_supports[:, 1],
                marker=_align_marker(">", "right", "middle"),
                s=500,
                linewidths=0,
                color=colors[0],
                label="X supports",
                zorder=2,
                clip_on=False,
                alpha=0.9,
            )

        if len(y_support_indices) > 0:
            ax.scatter(
                scaled_y_supports[:, 0],
                scaled_y_supports[:, 1],
                marker=_align_marker("^", "middle", "top"),
                s=500,
                linewidths=0,
                color=colors[3],
                label="Y supports",
                zorder=2,
                clip_on=False,
                alpha=0.9,
            )

    # Customize plot appearance
    ax.grid(False)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.autoscale()

    if legend:
        fig.legend(loc="outside lower center", ncols=3, frameon=False)

    if title:
        ax.set_title(title)

    if save_path and save_name:
        save_figure(fig, save_path, save_name)

    return fig, ax


def plot_losses(
    losses,
    name: str,
    *,
    max_steps: int,
    save_path: Optional[str] = None,
    save_name: Optional[str] = None,
    figsize: Tuple[float, float] = BIG_SQUARE,
    ax: Optional[Axes] = None,
    legend: Optional[bool] = False,
    color: Optional[str] = "b",
    linestyle: Optional[str] = "-",
    title: Optional[str] = None,
    y_limit: Optional[float] = None,
    label: Optional[str] = None,
):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # When the losses stagnate, i.e. it means convergence
    ax.plot(
        losses,
        color=color,
        linestyle=linestyle,
        linewidth=2.0,
        label=label,
    )

    ax.scatter(
        len(losses) - 1,
        losses[-1],
        facecolor="none",
        edgecolor=color,
        clip_on=False,
    )

    # ax.set_box_aspect(1)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, max_steps)

    # Plot a point at the last iteration

    if title:
        ax.set_title(title)

    if legend:
        ax.legend(ncols=1, frameon=False)

    if save_path and save_name:
        save_figure(fig, save_path, save_name)

    if y_limit:
        ax.set_ylim(0, y_limit)

    return fig, ax


# TODO: Make this the same format as the rest of the plotting functions
def plot_sinusoids(X, Ys, Ys0, batch, n_plots, algo_name, n_steps):
    fig, axs = plt.subplots(
        ncols=n_plots,
        subplot_kw=dict(box_aspect=1),
        figsize=MEDIUM_SQUARE,
        layout="constrained",
    )
    colors = sns.color_palette()

    amp, phase = batch["amp"], batch["phase"]

    # Note - this is a little bit hacky - for validation batch, there should
    # be no split between train and test
    data, labels = batch["x"], batch["y"]

    sines = lambda x, amp, phase: amp * np.sin(x + phase)

    for (id, ax), color in zip(enumerate(axs), colors):
        ax.plot(X, Ys[id], label=f"{algo_name}", linewidth=2, color=color)
        ax.plot(
            X,
            Ys0[id],
            label="Standard",
            linewidth=2,
            color=color,
            linestyle="--",
        )
        ax.plot(
            X,
            sines(X, amp[id], phase[id]),
            linestyle="--",
            color="k",
            label="Ground Truth",
            linewidth=1,
        )
        ax.scatter(
            data[id],
            labels[id],
            color="k",
            s=20,
            label="Observations",
            zorder=3,
        )

        ax.set_title(f"Task {id + 1}")
        ax.set_xlim(-5.0, 5.0)
        ax.set_ylim(-5.5, 5.5)

    fig.legend(
        loc="lower center",
        ncols=4,
        frameon=False,
        labels=[
            f"{algo_name} ({n_steps} it.)",
            f"Standard  ({n_steps} it.)",
            "Ground Truth",
            "Observations",
        ],
        handles=[
            Line2D([0], [0], c="gray", lw=2),
            Line2D([0], [0], c="gray", ls="--", lw=2),
            Line2D([0], [0], c="k", ls="--", lw=1),
            Line2D([0], [0], marker="o", markersize=5, c="k", linestyle="None"),
        ],
    )

    plt.show()
