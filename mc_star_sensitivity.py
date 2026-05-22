"""
mc_star_sensitivity.py

Monte Carlo sensitivity analysis for the optimal red-star locations
identified in efficiency_heatmap.py (Figure 1: Resources heatmap).

Red stars mark the (N_g, N_it) point with minimum resource cost
N_g * N_it on each metric contour.  This script:

  1. Recomputes baseline red-star locations from the nominal parameters.
  2. Runs a Monte Carlo simulation (N_MC samples) varying all parameters
     *except* delta and eps by ±10 % (independent uniform draws).
  3. Plots all MC-computed optimal points together on a single resource
     heatmap, coloured by contour level, with 68 % confidence ellipses
     and the nominal baseline stars overlaid.
  4. Produces supporting marginal and sensitivity figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib import ticker
import seaborn as sns

# ===========================================================
# 1.  NOMINAL PARAMETERS  (mirror efficiency_heatmap.py exactly)
# ===========================================================
delta = -7.711545013271975    # ground-state energy — NOT varied
kappa = 0.01
eps   = 1e-6
gam1  = 1.06     # ν₀
omg1  = 0.63     # exponent (δ_exp)
E0INF = 3.94      # β
E00   = 2.0      # α
off   = 110.0    # empirical gate offset

NOMINAL = dict(kappa=kappa, eps=eps, gam1=gam1, omg1=omg1,
               E0INF=E0INF, E00=E00, off=off)
# delta and eps are fixed — only the remaining 5 params are varied
PARAM_NAMES  = [k for k in NOMINAL if k != 'eps']
PARAM_LABELS = dict(
    kappa=r'$\kappa$',
    gam1 =r'$\nu_0$',
    omg1 =r'$\delta_{\!exp}$',
    E0INF=r'$\beta$',
    E00  =r'$\alpha$',
    off  =r'$N_{off}$',
)

contour_vals = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
CMAP         = plt.cm.turbo
COLORS       = CMAP(np.linspace(0.05, 0.95, len(contour_vals)))

plt.rcParams.update({'font.size': 14})

# ===========================================================
# 2.  GRIDS
# ===========================================================
t_arr = np.linspace(1, 1000, 1000, dtype=int)
N_arr = np.linspace(1, 2000, 2000, dtype=int)

N_2d = N_arr[:, np.newaxis].astype(float)   # (2000, 1)
t_2d = t_arr[np.newaxis, :].astype(float)   # (1, 1000)

rows, cols = len(N_arr), len(t_arr)
X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

# ===========================================================
# 3.  VECTORISED METRIC & RESOURCE COMPUTATION
# ===========================================================
def compute_metric(p, delta, N_2d, t_2d):
    """
    Returns met_arr (nN, nt) and eng_arr (nN, nt).
    Rows where N <= off yield NaN in met_arr.
    """
    dN = N_2d - p['off']
    with np.errstate(invalid='ignore', divide='ignore'):
        dN_pow = np.where(dN > 0, dN ** (-p['omg1']), np.nan)
        met = ((1 - p['eps']) ** N_2d) * (
            p['E0INF'] * np.exp(-p['kappa'] * N_2d)
            + delta
            + p['E00']  * np.exp(-p['gam1'] * t_2d * dN_pow)
        ) - delta
    eng = N_2d * t_2d
    return met, eng


# ===========================================================
# 4.  RED-STAR EXTRACTION
#     For each contour level: find (N_g, N_it) on the iso-metric
#     curve that minimises total resources N_g * N_it.
# ===========================================================
def extract_red_stars(met_arr, N_arr, t_arr, contour_vals):
    stars = {}
    for cv in contour_vals:
        below     = met_arr <= cv
        t_idx     = np.argmax(below, axis=1)          # first True per N row
        any_valid = below[np.arange(len(N_arr)), t_idx]

        if not np.any(any_valid):
            stars[cv] = None
            continue

        resources = np.where(any_valid, N_arr * t_arr[t_idx], np.inf)
        i_min     = np.argmin(resources)

        stars[cv] = (float(N_arr[i_min]), float(t_arr[t_idx[i_min]])) \
                    if any_valid[i_min] else None
    return stars


# ===========================================================
# 5.  BASELINE
# ===========================================================
print("Computing baseline …")
met_nom, eng_nom = compute_metric(NOMINAL, delta, N_2d, t_2d)
baseline_stars   = extract_red_stars(met_nom, N_arr, t_arr, contour_vals)

print("Baseline red stars  (N_g*, N_it*):")
for cv, s in baseline_stars.items():
    print(f"  E_inf={cv:.2f}  →  {s}")


# ===========================================================
# 6.  MONTE CARLO  (all params ±10 %, uniform, independent)
# ===========================================================
N_MC = 10000
rng  = np.random.default_rng(42)

mc_samples = {
    name: rng.uniform(val * 0.90, val * 1.10, N_MC)
    for name, val in NOMINAL.items()
}

mc_stars = {cv: {'N': [], 't': []} for cv in contour_vals}

print(f"\nRunning {N_MC} Monte Carlo samples …")
for i in range(N_MC):
    if i % 100 == 0:
        print(f"  {i}/{N_MC}")
    p        = dict(NOMINAL)   # carries fixed eps; varied keys overwrite below
    p.update({name: mc_samples[name][i] for name in PARAM_NAMES})
    met_i, _ = compute_metric(p, delta, N_2d, t_2d)
    stars_i  = extract_red_stars(met_i, N_arr, t_arr, contour_vals)
    for cv in contour_vals:
        if stars_i[cv] is not None:
            mc_stars[cv]['N'].append(stars_i[cv][0])
            mc_stars[cv]['t'].append(stars_i[cv][1])

for cv in contour_vals:
    mc_stars[cv]['N'] = np.array(mc_stars[cv]['N'])
    mc_stars[cv]['t'] = np.array(mc_stars[cv]['t'])

print("MC complete.")
print("\nMC summary  (N_g*: μ ± σ  |  N_it*: μ ± σ):")
for cv in contour_vals:
    Ng = mc_stars[cv]['N']
    tt = mc_stars[cv]['t']
    if len(Ng):
        print(f"  E_inf={cv:.2f}  N_g*: {np.mean(Ng):.0f} ± {np.std(Ng):.0f}"
              f"   N_it*: {np.mean(tt):.0f} ± {np.std(tt):.0f}"
              f"   ({len(Ng)} valid)")


# ===========================================================
# 7.  HELPER — 68 % CONFIDENCE ELLIPSE
# ===========================================================
def confidence_ellipse(ax, x, y, color, n_std=1.0, **kwargs):
    """Draw a covariance ellipse covering n_std standard deviations."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle  = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h   = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h,
                  angle=angle, edgecolor=color, fc='none',
                  lw=2, linestyle='--', **kwargs)
    ax.add_patch(ell)


# ===========================================================
# 8.  MAIN FIGURE — SINGLE RESOURCE HEATMAP WITH ALL MC STARS
# ===========================================================
fig, ax = plt.subplots(figsize=(14, 10))

# --- background heatmap ---
sns.heatmap(
    eng_nom,
    cmap='Blues',
    ax=ax,
    cbar_kws={
        'label' : r'$\Delta = N_g \times N_{it}$  (total resources)',
        'format': ticker.ScalarFormatter(useMathText=True),
        'shrink': 0.75,
    },
)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

# --- black dashed resource contours (matching efficiency_heatmap.py) ---
resource_levels = [1e4, 1e5, 5e5, 1e6]
c_res = ax.contour(X, Y, eng_nom, levels=resource_levels,
                   colors='black', alpha=0.7, linestyles='dashed', linewidths=1.5)

def sci_tex(x):
    exp  = int(np.log10(x))
    base = x / (10 ** exp)
    return rf"$10^{{{exp}}}$" if abs(base - 1) < 1e-3 \
           else rf"${base:.1f}\times10^{{{exp}}}$"

ax.clabel(c_res, fmt=sci_tex, inline=True, fontsize=11)

# --- per-contour: magenta contour, MC scatter, ellipse, baseline star ---
legend_handles = []

for ci, cv in enumerate(contour_vals):
    col    = COLORS[ci]
    Ng_mc  = mc_stars[cv]['N']
    t_mc   = mc_stars[cv]['t']

    # magenta metric contour
    ax.contour(X, Y, met_nom, levels=[cv],
               colors=['magenta'], alpha=0.55,
               linestyles='dotted', linewidths=2)

    if len(Ng_mc) == 0:
        continue

    # convert physical coords → grid (pixel) coords for plotting
    x_mc = np.interp(t_mc,  t_arr, np.arange(len(t_arr)))
    y_mc = np.interp(Ng_mc, N_arr, np.arange(len(N_arr)))

    # MC scatter
    ax.scatter(x_mc, y_mc, color=col, alpha=0.20, s=18, zorder=5)

    # 68 % confidence ellipse in grid coords
    confidence_ellipse(ax, x_mc, y_mc, color=col, n_std=1.0, zorder=6)

    # baseline star
    if baseline_stars[cv] is not None:
        Ng_b, t_b = baseline_stars[cv]
        x_b = np.interp(t_b,  t_arr, np.arange(len(t_arr)))
        y_b = np.interp(Ng_b, N_arr, np.arange(len(N_arr)))
        ax.plot(x_b, y_b, '*', color=col, markersize=20,
                markeredgecolor='k', markeredgewidth=0.8, zorder=10)

    # legend proxy: filled circle (MC) + star (baseline)
    proxy = mpatches.Patch(color=col, alpha=0.6, label=rf'$Error={cv:.2f}$')
    legend_handles.append(proxy)

# --- axes formatting ---
ax.set_xticks(np.linspace(0, cols - 1, 10))
ax.set_xticklabels(np.linspace(1, t_arr[-1], 10, dtype=int), rotation=0)
ax.set_yticks(np.linspace(0, rows - 1, 10))
ax.set_yticklabels(np.linspace(N_arr[0], N_arr[-1], 10, dtype=int))
ax.invert_yaxis()
ax.set_xlabel(r'$N_{it}$', fontsize=22)
ax.set_ylabel(r'$N_g$',    fontsize=22)
ax.set_title(
    r'MC optimal $(N_g^*, N_{it}^*)$ under $\pm10\%$ parameter perturbation'
    '\n'
    r'colored ★ = nominal optimum   ·   scatter + ellipse (68%) = MC ($N_{MC}=$'
    f'{N_MC})',
    fontsize=14,
)

# dummy entries for the annotation items in the legend
star_proxy  = plt.Line2D([0], [0], marker='*', color='grey',
                          markeredgecolor='k', markersize=14,
                          linestyle='None', label='Nominal optimum (★)')
scat_proxy  = plt.Line2D([0], [0], marker='o', color='grey',
                          alpha=0.4, markersize=8,
                          linestyle='None', label=r'MC sample ($\pm10\%$)')
ell_proxy   = mpatches.Patch(edgecolor='grey', facecolor='none',
                              linestyle='--', linewidth=2, label='68% ellipse')
cont_proxy  = plt.Line2D([0], [0], color='magenta', lw=2,
                          linestyle='dotted', label='Metric contour')

ax.legend(
    handles=legend_handles + [star_proxy, scat_proxy, ell_proxy, cont_proxy],
    loc='upper right', fontsize=11,
    title=r'$Error$ contour level',
    title_fontsize=11,
    ncol=2,
)

plt.tight_layout()
plt.savefig('Figs/mc_heatmap_all_stars.png', dpi=120, bbox_inches='tight')
plt.show()


# ===========================================================
# 9.  SUPPORTING FIGURE — MARGINALS + OAT SENSITIVITY
# ===========================================================

# --- OAT ---
oat = {cv: {name: {'dN': [], 'dt': []} for name in PARAM_NAMES}
       for cv in contour_vals}

for name in PARAM_NAMES:
    for factor in (0.90, 1.10):
        p = dict(NOMINAL)
        p[name] = NOMINAL[name] * factor
        met_p, _ = compute_metric(p, delta, N_2d, t_2d)
        sp = extract_red_stars(met_p, N_arr, t_arr, contour_vals)
        for cv in contour_vals:
            if sp[cv] is not None and baseline_stars[cv] is not None:
                oat[cv][name]['dN'].append(sp[cv][0] - baseline_stars[cv][0])
                oat[cv][name]['dt'].append(sp[cv][1] - baseline_stars[cv][1])

# --- layout: 6 rows × 3 cols (N_g hist | N_it hist | OAT bars) ---
n_cv  = len(contour_vals)
fig2, axes2 = plt.subplots(n_cv, 3, figsize=(18, 3.5 * n_cv))

labels = [PARAM_LABELS[n] for n in PARAM_NAMES]
y_pos  = np.arange(len(PARAM_NAMES))

for ci, cv in enumerate(contour_vals):
    col   = COLORS[ci]
    Ng_mc = mc_stars[cv]['N']
    t_mc  = mc_stars[cv]['t']
    ax_Ng, ax_t, ax_s = axes2[ci]

    # --- N_g* histogram ---
    if len(Ng_mc) > 1:
        ax_Ng.hist(Ng_mc, bins=30, color=col, edgecolor='k', alpha=0.75)
        if baseline_stars[cv] is not None:
            ax_Ng.axvline(baseline_stars[cv][0], color='k', lw=2,
                          ls='--', label='Nominal')
        ax_Ng.set_title(
            rf'$N_g^*$  $Error={cv:.2f}$'
            rf'   $\mu={np.mean(Ng_mc):.0f},\;\sigma={np.std(Ng_mc):.0f}$',
            fontsize=10)
        ax_Ng.set_xlabel(r'$N_g^*$'); ax_Ng.set_ylabel('Count')
        ax_Ng.legend(fontsize=9)
    else:
        ax_Ng.text(0.5, 0.5, 'no data', ha='center', va='center',
                   transform=ax_Ng.transAxes)

    # --- N_it* histogram ---
    if len(t_mc) > 1:
        ax_t.hist(t_mc, bins=30, color=col, edgecolor='k', alpha=0.75)
        if baseline_stars[cv] is not None:
            ax_t.axvline(baseline_stars[cv][1], color='k', lw=2,
                         ls='--', label='Nominal')
        ax_t.set_title(
            rf'$N_{{it}}^*$  $Error={cv:.2f}$'
            rf'   $\mu={np.mean(t_mc):.0f},\;\sigma={np.std(t_mc):.0f}$',
            fontsize=10)
        ax_t.set_xlabel(r'$N_{it}^*$'); ax_t.set_ylabel('Count')
        ax_t.legend(fontsize=9)
    else:
        ax_t.text(0.5, 0.5, 'no data', ha='center', va='center',
                  transform=ax_t.transAxes)

    # --- OAT sensitivity bar (signed mean shift) ---
    dN_m = [np.mean(oat[cv][n]['dN']) if oat[cv][n]['dN'] else 0.0
            for n in PARAM_NAMES]
    dt_m = [np.mean(oat[cv][n]['dt']) if oat[cv][n]['dt'] else 0.0
            for n in PARAM_NAMES]

    bar_colors = ['steelblue' if v >= 0 else 'tomato' for v in dN_m]
    ax_s.barh(y_pos - 0.2, dN_m, height=0.35,
              color=bar_colors, edgecolor='k', alpha=0.85, label=r'$\Delta N_g^*$')
    bar_colors_t = ['darkorange' if v >= 0 else 'purple' for v in dt_m]
    ax_s.barh(y_pos + 0.2, dt_m, height=0.35,
              color=bar_colors_t, edgecolor='k', alpha=0.85, label=r'$\Delta N_{it}^*$')

    ax_s.axvline(0, color='k', lw=0.8, ls='--')
    ax_s.set_yticks(y_pos); ax_s.set_yticklabels(labels)
    ax_s.set_xlabel('Signed shift from nominal')
    ax_s.set_title(rf'OAT sensitivity   $Error={cv:.2f}$', fontsize=10)
    ax_s.legend(fontsize=9, loc='lower right')

fig2.suptitle(
    r'Supporting analysis: marginal distributions and OAT sensitivity (±10 %)',
    fontsize=15,
)
plt.tight_layout()
plt.savefig('Figs/mc_supporting.png', dpi=120, bbox_inches='tight')
plt.show()

print("\nSaved:  Figs/mc_heatmap_all_stars.png")
print("        Figs/mc_supporting.png")
