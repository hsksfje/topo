import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D


palette = sns.color_palette("Dark2")
sns.set_palette(palette)

MM = 0.0393701  # millimeter in inches (1 mm = 0.0393701 in)

FIGWIDTH_2 = 180 * MM
FIGWIDTH_1 = 90 * MM
FIGWIDTH_3 = 60 * MM

SMALL_FIGURE = (FIGWIDTH_1, FIGWIDTH_1)
LARGE_FIGURE = (FIGWIDTH_2, (10.0 / 16.0) * FIGWIDTH_2)

DPI = 300


def plot_sinusoids(X, Ys, Ys0, batch, n_plots, algo_name, n_steps):
    fig, axs = plt.subplots(
        ncols=n_plots,
        subplot_kw=dict(box_aspect=1),
        figsize=LARGE_FIGURE,
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
        ax.set_ylim(X.min(), 1.5 * X.max())

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
