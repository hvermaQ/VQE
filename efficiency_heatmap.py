import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from multiprocessing import Pool
import itertools

# ===========================================================
# 1. PARAMETERS AND PREPROCESSING
# ===========================================================
# Model constants for simulated error and resource metrics
delta = -7.711545013271975
kappa = 0.01
eps = 1e-6

gam1 = 0.78    # ν₀
omg1 = 0.58    # δ
E0INF = 4     # β
E00 = 2        # α
off = 100 #offset discovered in emperical power law 

plt.rcParams.update({'font.size': 20})

# Define grid of iterations (t) and gate counts (N)
t = np.linspace(1, 1000, 1000, dtype=int)
N = np.linspace(1, 2000, 2000, dtype=int)

# ===========================================================
# 2. FUNCTION DEFINING THE METRIC
# ===========================================================
def final_energy(NN, tt):
    """
    Computes simulated error metric and resource cost.

    Args:
        NN : int or float
            Number of gates
        tt : int or float
            Number of iterations

    Returns:
        Einf1 : float
            Error metric value
        prod  : float
            Resource product NN * tt
    """
    Einf1 = ((1 - eps)**NN) * (
        E0INF * np.exp(-kappa * NN)
        + delta
        + E00 * np.exp(-gam1 * tt * ((NN-off)**(-omg1)))
    ) - delta

    prod = NN * tt
    return [Einf1, prod]


# ===========================================================
# 3. PARALLEL GRID COMPUTATION
# ===========================================================
def main():
    """
    Computes metric and resource arrays over full (N, t) grid using multiprocessing.
    """
    spann = list(itertools.product(N, t))
    with Pool() as p:
        res_pool = p.starmap(final_energy, spann)

    metric = [it[0] for it in res_pool]
    energy = [it[1] for it in res_pool]
    return metric, energy


# ===========================================================
# 4. MAIN EXECUTION
# ===========================================================
if __name__ == "__main__":

    # Compute metric and resource data
    metric_list, energy_list = main()

    # Reshape into 2D grids aligned with (N, t)
    met_arr = np.reshape(metric_list, (len(N), len(t)))
    eng_arr = np.reshape(energy_list, (len(N), len(t)))

    # Define efficiency: log10((1 - metric) / resources)
    eff_arr = np.log10(np.divide(1 - met_arr, eng_arr))

    # Metric contour levels
    contour_vals = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
# ===========================================================
# 5. RESOURCES HEATMAP (Δ)
# ===========================================================
fig1, ax1 = plt.subplots(figsize=(13, 10))

# Heatmap for total resources
sns.heatmap(
    eng_arr,
    cmap="Blues",
    ax=ax1,
    cbar_kws={
        'label': r'$\Delta =~\text{Total algorithmic resources}$',
        'format': ticker.ScalarFormatter(useMathText=True)
    }
)

ax1.spines['bottom'].set_visible(True)
ax1.spines['left'].set_visible(True)

rows, cols = met_arr.shape
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# -------------------------------
# Contours for given metric levels
# -------------------------------
for contour_val in contour_vals:
    c = ax1.contour(
        X, Y, met_arr,
        levels=[contour_val],
        colors='magenta',
        alpha=0.75,
        linestyles='dotted',
        linewidths=3
    )

    # -------------------------------------
    # Find minimal-resource point on contour
    # -------------------------------------
    paths = c.collections[0].get_paths()
    if paths:
        # Combine all vertices from all contour paths
        verts = np.vstack([p.vertices for p in paths])

        # Map from grid to actual N, t coordinates
        t_vals = np.interp(verts[:, 0], np.arange(len(t)), t)
        N_vals = np.interp(verts[:, 1], np.arange(len(N)), N)

        # Compute resources (Δ = N * t)
        resources = N_vals * t_vals
        min_idx = np.argmin(resources)

        # Coordinates in grid space for plotting the star
        x_star, y_star = verts[min_idx]

        # Plot the optimal point directly on the contour
        ax1.plot(x_star, y_star, 'r*', markersize=18, zorder=10)

        # Optional: place label slightly offset from contour midpoint
        v = verts[len(verts)//2]
        offset_x, offset_y = 20, 100
        label_pos = (v[0] + offset_x, v[1] + offset_y)

        ax1.clabel(
            c,
            inline=True,
            inline_spacing=0,
            fontsize=20,
            fmt={contour_val: f"{contour_val:.2f}"},
            manual=[label_pos]
        )

        # Force labels to stay horizontal (no rotation)
        for txt in ax1.texts:
            txt.set_rotation(0)

# -------------------------------
# (b) Brown dashed contours: Resource levels
# -------------------------------
resource_levels = [1e4, 1e5, 5e5, 1e6]
c4 = ax1.contour(
    X, Y, eng_arr,
    levels=resource_levels,
    colors='black',
    alpha=0.9,
    linestyles='dashed',
    linewidths=2,
)

# Label resource contours manually
paths_list = c4.collections
manual_positions = []
for i, level in enumerate(resource_levels):
    paths = paths_list[i].get_paths()
    if paths:
        mid_path = paths[len(paths)//2]
        v = mid_path.vertices
        mid_idx = len(v)//2
        x_mid, y_mid = v[mid_idx]
        manual_positions.append((x_mid, y_mid + 500))

def sci_tex(x):
    exp = int(np.log10(x))
    base = x / (10**exp)
    if abs(base - 1) < 1e-3:
        return rf"$10^{{{exp}}}$"
    else:
        return rf"${base:.1f}\times10^{{{exp}}}$"

ax1.clabel(
    c4,
    fmt=sci_tex,
    inline=True,
    inline_spacing=0,
    fontsize=18,
    manual=manual_positions
)

# Axis formatting
ax1.set_xticks(np.linspace(0, cols-1, 10))
ax1.set_xticklabels(np.linspace(1, t[-1], 10, dtype=int), rotation=0)
ax1.set_yticks(np.linspace(0, rows-1, 10))
ax1.set_yticklabels(np.linspace(N[0], N[-1], 10, dtype=int))
ax1.invert_yaxis()
ax1.set_xlabel(r'$N_{it}$', fontsize=24)
ax1.set_ylabel(r'$N_g$', fontsize=24)

plt.savefig('Figs/en_met_eps5_HVA.png', bbox_inches='tight', dpi=100)
plt.show()

#fit line for optimal points
import numpy as np
from scipy.optimize import curve_fit

# Extract blue star points from resource heatmap plot (ax1)
star_coords_res = []
for artist in ax1.get_children():
    if isinstance(artist, plt.Line2D):
        if artist.get_marker() == '*' and artist.get_color() == 'b' and len(artist.get_xdata()) == 1:
            x, y = artist.get_xdata()[0], artist.get_ydata()[0]
            t_val = np.interp(x, np.arange(len(t)), t)
            N_val = np.interp(y, np.arange(len(N)), N)
            star_coords_res.append([t_val, N_val])
star_coords_res = np.array(star_coords_res)

if len(star_coords_res) > 1:
    t_vals = star_coords_res[:, 0]
    N_vals = star_coords_res[:, 1]

    def fit_model(x, a, b, c):
        return a * (x**b) + c

    params, _ = curve_fit(fit_model, t_vals, N_vals)
    N_pred = fit_model(t_vals, *params)

    # Calculate R^2
    residuals = N_vals - N_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((N_vals - np.mean(N_vals))**2)
    r_squared = 1 - (ss_res / ss_tot)

    print(f'Resource heatmap fit: N_g = {params[0]:.3e} * N_it^{params[1]:.3f} + {params[2]:.2f}')
    print(f'R^2 = {r_squared:.4f}')

    # Plot fit curve on resource heatmap axes
    t_fine = np.linspace(np.min(t_vals), np.max(t_vals), 200)
    ax1.plot(np.searchsorted(t, t_fine), np.interp(fit_model(t_fine, *params), N, np.arange(len(N))),
             'r-', lw=2, label='Fit')
    ax1.legend()
else:
    print('Not enough blue stars found on resource heatmap plot to fit.')

plt.show()

# ===========================================================
# 6. EFFICIENCY HEATMAP (η)
# ===========================================================
fig2, ax2 = plt.subplots(figsize=(13, 10))

# Heatmap for efficiency
sns.heatmap(
    eff_arr,
    cmap=sns.color_palette("Greens", 12),
    ax=ax2,
    cbar_kws={'label': r'$\log_{10}(\eta)$'},
    vmin=-7, vmax=-4,
)

ax2.spines['bottom'].set_visible(True)
ax2.spines['left'].set_visible(True)

X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# -----------------------------------
# Blue contours for the metric levels
# -----------------------------------
for contour_val in contour_vals:
    c = ax2.contour(
        X, Y, met_arr,
        levels=[contour_val],
        colors='magenta',
        alpha=0.75,
        linestyles='dotted',
        linewidths=2
    )


    paths = c.collections[0].get_paths()
    if paths:
        # Merge all contour vertices
        verts = np.vstack([p.vertices for p in paths])

        # Map from grid to actual coordinates
        t_vals = np.interp(verts[:, 0], np.arange(len(t)), t)
        N_vals = np.interp(verts[:, 1], np.arange(len(N)), N)

        # Interpolate efficiency at those vertices
        # (convert to integer indices safely)
        x_idx = np.clip(np.round(verts[:, 0]).astype(int), 0, len(t)-1)
        y_idx = np.clip(np.round(verts[:, 1]).astype(int), 0, len(N)-1)
        eff_vals = eff_arr[y_idx, x_idx]

        # Find the vertex of maximum efficiency
        max_idx = np.argmax(eff_vals)
        x_star, y_star = verts[max_idx]

        # Plot the max-efficiency point
        ax2.plot(x_star, y_star, 'r*', markersize=18, zorder=10)

        # Add contour label near mid vertex
        v = verts[len(verts)//2]
        ax2.clabel(
            c,
            inline=True,
            inline_spacing=0,
            fontsize=20,
            fmt={contour_val: f"{contour_val:.2f}"},
            manual=[(v[0], v[1])]
        )

    # Force labels to stay horizontal (no rotation)
    for txt in ax2.texts:
        txt.set_rotation(0)

# Axis formatting
ax2.set_xticks(np.linspace(0, cols-1, 10))
ax2.set_xticklabels(np.linspace(1, t[-1], 10, dtype=int), rotation=0)
ax2.set_yticks(np.linspace(0, rows-1, 10))
ax2.set_yticklabels(np.linspace(N[0], N[-1], 10, dtype=int))
ax2.invert_yaxis()
ax2.set_xlabel(r'$N_{it}$', fontsize=24)
ax2.set_ylabel(r'$N_g$', fontsize=24)

plt.savefig('Figs/eff_met_eps5_HVA.png', bbox_inches='tight', dpi=100)
plt.show()
