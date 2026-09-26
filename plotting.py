import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLOR_KEEP = "#1746c9"
COLOR_DROP = "#e03131"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.grid": True, "grid.color": "#d9d9d9", "grid.linestyle": "--", "grid.alpha": 0.5,
    "axes.edgecolor": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    "figure.facecolor": "white", "axes.axisbelow": True,
})


def _axes(x, y, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(max(0, x.min() - 20), x.max() + 20)
    ax.set_ylim(0, float(y.max()))
    return fig, ax


def plot_ground_truth(day, path):
    clean, loss = day[~day["is_loss"]], day[day["is_loss"]]
    fig, ax = _axes(day["g"], day["Potencia"], "POA Irradiance (W/m$^2$)", "Power (kW)")
    ax.scatter(clean["g"], clean["Potencia"], s=6, alpha=0.6, color=COLOR_KEEP, edgecolors="none")
    ax.scatter(loss["g"], loss["Potencia"], s=16, alpha=0.3, color=COLOR_DROP, marker="x", linewidths=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_filter_result(day, outliers, folder):
    kept, removed = day[~outliers], day[outliers]
    folder.mkdir(parents=True, exist_ok=True)

    fig, ax = _axes(day["g"], day["Potencia"], "POA Irradiance (W/m$^2$)", "Power (kW)")
    ax.scatter(kept["g"], kept["Potencia"], s=6, alpha=0.6, color=COLOR_KEEP, edgecolors="none")
    ax.scatter(removed["g"], removed["Potencia"], s=16, alpha=0.3, color=COLOR_DROP, marker="x", linewidths=0.6)
    fig.savefig(folder / "before.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = _axes(day["g"], day["Potencia"], "POA Irradiance (W/m$^2$)", "Power (kW)")
    ax.scatter(kept["g"], kept["Potencia"], s=6, alpha=0.6, color=COLOR_KEEP, edgecolors="none")
    fig.savefig(folder / "after.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_normalized(x, y, path, color=COLOR_KEEP):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(x, y, s=6, alpha=0.6, color=color, edgecolors="none")
    ax.set_xlabel("Normalized POA Irradiance")
    ax.set_ylabel("Normalized Power")
    ax.set_xlim(max(0, x.min() - 0.02), x.max() + 0.02)
    ax.set_ylim(0, float(y.max()) + 0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
