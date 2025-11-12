import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fixed parameters (reuse your definitions for delta, kappa, etc.)
delta = -7.711545013271975
kappa = 0.01
gam1 = 0.78
omg1 = 0.58
E0INF = 4
E00 = 2
off = 100
gate_energy = 1e-7

N = np.linspace(20, 2000, 2000, dtype=int)
t = np.linspace(20, 1000, 1000, dtype=int)

def final_metric(NN, tt, eps):
    if NN <= off:
        return np.nan  # Exclude invalid points
    return ((1 - eps)**NN) * (
        E0INF * np.exp(-kappa * NN) + delta + E00 * np.exp(-gam1 * tt * ((NN - off)**(-omg1)))
    ) - delta

def energy_calc(NN, tt):
    dep = (NN - 15) / 150
    total_including_qpu_run = 96621.711 * np.exp(0.262 * dep) - 102781
    only_qpu_run = 18426.225 * dep + 19210.299
    flops_per_iteration = total_including_qpu_run - only_qpu_run
    green_500 = 72.733 * 10**9
    energy_per_iteration = flops_per_iteration / green_500
    energy = gate_energy * NN * energy_per_iteration * tt
    return energy

def fit_power(x, a, b, c):
    return a * x**b + c

def fit_exp(x, a, b, c):
    return a * np.exp(-b * x) + c

def get_optimal_points_for_eps(eps):
    metric_arr = np.full((len(N), len(t)), np.nan)
    energy_arr = np.full((len(N), len(t)), np.nan)
    for i in range(len(N)):
        if N[i] <= off:
            continue
        for j in range(len(t)):
            metric_arr[i, j] = final_metric(N[i], t[j], eps)
            energy_arr[i, j] = energy_calc(N[i], t[j])

    min_m = np.nanmin(metric_arr)
    max_m = np.nanmax(metric_arr)
    contour_vals = np.linspace(min_m + 0.001, max_m - 0.001, 25)

    optimal_points = []
    for val in contour_vals:
        if np.isnan(val):
            continue
        diff = np.abs(metric_arr - val)
        threshold = 0.05 * np.abs(val)
        mask = diff < threshold
        if np.count_nonzero(mask) == 0:
            continue
        idxs = np.where(mask)
        energies = energy_arr[idxs]
        min_idx = np.nanargmin(energies)
        opt_t = t[idxs[1][min_idx]]
        opt_N = N[idxs[0][min_idx]]
        optimal_points.append((opt_t, opt_N))

    return np.array(optimal_points)

epsilons = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

plt.figure(figsize=(10, 7))

for eps in epsilons:
    points = get_optimal_points_for_eps(eps)
    if points.shape[0] < 2:
        print(f"Epsilon={eps}: Not enough points for fit")
        continue
    t_vals = points[:, 0]
    N_vals = points[:, 1]

    # 1. Power law fit
    try:
        popt_pw, _ = curve_fit(fit_power, t_vals, N_vals, maxfev=5000, p0=[1e-6, 1.0, 0])
        N_pred_pw = fit_power(t_vals, *popt_pw)
        ss_res = np.sum((N_vals - N_pred_pw) ** 2)
        ss_tot = np.sum((N_vals - np.mean(N_vals)) ** 2)
        r2_pw = 1 - ss_res / ss_tot
    except Exception as e:
        popt_pw, r2_pw = None, -np.inf

    # 2. If R^2 < 0.8, try exponential fit
    if r2_pw < 0.8:
        try:
            popt_exp, _ = curve_fit(fit_exp, t_vals, N_vals, maxfev=5000, p0=[np.max(N_vals), 0.01, np.min(N_vals)])
            N_pred_exp = fit_exp(t_vals, *popt_exp)
            ss_res_exp = np.sum((N_vals - N_pred_exp) ** 2)
            r2_exp = 1 - ss_res_exp / ss_tot
            if r2_exp > r2_pw:
                print(f"Epsilon={eps:.0e}: Exponential fit N_g = {popt_exp[0]:.3e} * exp(-{popt_exp[1]:.3f} * N_it) + {popt_exp[2]:.2f} with R^2 = {r2_exp:.4f}")
                t_fine = np.linspace(np.min(t_vals), np.max(t_vals), 200)
                plt.plot(t_fine, fit_exp(t_fine, *popt_exp), label=f'eps={eps:.0e} (exp)')
                plt.scatter(t_vals, N_vals, s=40)
                continue
        except Exception as e:
            print(f"Epsilon={eps:.0e}: Exponential fit failed: {e}")

    # Default: plot and print power fit
    print(f"Epsilon={eps:.0e}: Power fit N_g = {popt_pw[0]:.3e} * N_it^{popt_pw[1]:.3f} + {popt_pw[2]:.2f} with R^2 = {r2_pw:.4f}")
    t_fine = np.linspace(np.min(t_vals), np.max(t_vals), 200)
    plt.plot(t_fine, fit_power(t_fine, *popt_pw), label=f'eps={eps:.0e} (pow)')
    plt.scatter(t_vals, N_vals, s=40)

plt.xlabel('N_it')
plt.ylabel('N_g')
plt.title('Optimal (N_it, N_g) fits for varying epsilons\n(fallback to exp if R2<0.8)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Fit functions and latex expressions for legend
def phys_fit(N_it):
    return 106 * N_it**0.386 - 403.85
phys_expr = r"$N_{g*} = 106 N_{it*}^{0.386} - 403.85$"

def algo_fit(N_it):
    return 31.59 * N_it**0.529 - 22.47
algo_expr = r"$N_{g*} = 31.59 N_{it*}^{0.529} - 22.47$"

t_fine = np.linspace(20, 2000, 300)
plt.figure(figsize=(10, 7))

plt.plot(t_fine, algo_fit(t_fine), '--m', linewidth=2, label='ALGO: ' + algo_expr)
plt.plot(t_fine, phys_fit(t_fine), '-.k', linewidth=2, label='PHYS: ' + phys_expr)

plt.xlabel(r'$N_{it*}$', fontsize=22)
plt.ylabel(r'$N_{g*}$', fontsize=22)
plt.legend(fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=18)
#plt.title('Comparison of ALGO and PHYS Fits', fontsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig('Figs/opt_params.png', bbox_inches='tight', dpi=100)
plt.show()
