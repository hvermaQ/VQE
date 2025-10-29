import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
from multiprocessing import Pool
import itertools

# ===========================================================
# PARAMETERS AND PREPROCESSING
# ===========================================================
delta = -7.712
kappa = 0.02
eps = 1e-5

gam1 = 0.78
omg1 = 0.58
E0INF = 40
E00 = 2

plt.rcParams.update({'font.size': 20})

t = np.linspace(1, 1001, 1000, dtype=int)
N = np.linspace(100, 2001, 1000, dtype=int)
contour_vals = [0.10, 0.25, 0.50]


# ===========================================================
# FUNCTION DEFINING METRIC
# ===========================================================
def final_energy(NN, tt):
    Einf1 = ((1 - eps)**NN) * (
        E0INF * np.exp(-kappa * NN)
        + delta
        + E00 * np.exp(-gam1 * tt * (NN**(-omg1)))
    ) - delta
    prod = NN * tt
    return [Einf1, prod]


# ===========================================================
# PARALLEL COMPUTATION
# ===========================================================
def main():
    spann = list(itertools.product(N, t))
    with Pool() as p:
        res_pool = p.starmap(final_energy, spann)
    metric = [it[0] for it in res_pool]
    energy = [it[1] for it in res_pool]
    return metric, energy


# ===========================================================
# MAIN EXECUTION
# ===========================================================
if __name__ == "__main__":
    metric_list, energy_list = main()

    met_arr = np.reshape(metric_list, (len(N), len(t)))
    eng_arr = np.reshape(energy_list, (len(N), len(t)))
    T, G = np.meshgrid(t, N)

    # ===========================================================
    # CONTOUR PLOT (Δ on log scale)
    # ===========================================================
    fig, ax = plt.subplots(figsize=(10, 8))

    # Safe min/max values (avoid zeros)
    min_val = np.nanmin(eng_arr[eng_arr > 0])
    max_val = np.nanmax(eng_arr)

    # Generate unique levels for contours (avoid duplicate edges)
    log_min, log_max = np.floor(np.log10(min_val)), np.ceil(np.log10(max_val))
    filled_levels = np.logspace(log_min, log_max, num=9)
    line_levels = np.logspace(log_min + 0.1, log_max - 0.1, num=7)  # offset to avoid overlap

    # --- Filled contours (background only) ---
    cf = ax.contourf(
        T, G, eng_arr,
        levels=filled_levels,
        cmap="cividis",
        norm=LogNorm(vmin=min_val, vmax=max_val)
    )

    # --- Single colorbar (clean ticks, no duplicates) ---
    cbar = plt.colorbar(cf, ax=ax, pad=0.02)
    cbar.set_label(r'$\Delta =~\text{Total algorithmic resources}$', rotation=270, labelpad=25)
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"$10^{{{int(np.log10(x))}}}$")
    )

    # --- Contour lines (offset levels prevent duplicates) ---
    c_lines = ax.contour(
        T, G, eng_arr,
        levels=line_levels,
        colors='black',
        linewidths=0.7,
        alpha=0.7
    )
    ax.clabel(
        c_lines,
        fmt=lambda x: rf"$10^{{{int(np.log10(x))}}}$",
        fontsize=12,
        inline=True,
        inline_spacing=8
    )

    # --- White dotted metric contours ---
    for contour_val in contour_vals:
        c = ax.contour(
            T, G, met_arr,
            levels=[contour_val],
            colors='white',
            linestyles='dotted',
            linewidths=2,
            alpha=0.9
        )
        # label near middle
        paths = c.collections[0].get_paths()
        if paths:
            mid = paths[len(paths)//2].vertices
            x_mid, y_mid = mid[len(mid)//2]
            ax.clabel(
                c,
                inline=True,
                inline_spacing=8,
                fontsize=16,
                fmt={contour_val: f"{contour_val:.2f}"},
                manual=[(x_mid, y_mid)]
            )

    # --- Axes and title ---
    ax.set_xlabel(r'$N_{it}$', fontsize=22)
    ax.set_ylabel(r'$N_g$', fontsize=22)
    ax.set_title(r'$\Delta$ Contours (log-scaled per order of magnitude)', fontsize=20)
    ax.grid(False)

    plt.tight_layout()
    plt.show()
