from multiprocessing import Pool
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib.ticker as mticker
from functools import partial

from matplotlib.ticker import FuncFormatter

def pow10_formatter(x, pos):
    # format numbers as LaTeX 10^{exp} when coefficient == 1, otherwise coeff\times10^{exp}
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coeff = x / (10**exp)
    # round small floating noise
    if abs(coeff - 1.0) < 1e-9:
        return rf"$10^{{{exp}}}$"
    else:
        # show one decimal for mantissa (drop trailing .0)
        mant = f"{coeff:.1f}".rstrip('0').rstrip('.')
        return rf"${mant}\times10^{{{exp}}}$"

#delt = -7.711545013271975
delt = -7.711545013271975 # Adjusted delta value for consistency
#alp0 = 1.0589003542176467
#omega = 0.6709446472822442 
#E0 = 4.1054927116664945
kappa = 0.01
#E000 = 0.0643109474333867 #change for different noise
#E001 = 0.5706839345786814 #change for different noise
#eps = 10**(-4)

gam1 = 0.78 #nu_0
omg1 = 0.58 #delta
E0INF = 4 #beta
E00 = 2 #alpha

plt.rcParams.update({'font.size': 16})


D = np.linspace(1000, 1000000, 10000, dtype = int)
N = np.linspace(100, 2000, 2000, dtype = int)

def energy_fixed_metric(Ng, Delta, eps):
    """Computes the energy metric for given Ng and Delta."""
    met = ((1 - eps)**Ng) * (E00 * np.exp(-gam1 * (Delta/Ng) * 1/((Ng-100)**(omg1))) + E0INF * np.exp(-kappa * Ng) + delt) - delt
    return 1-met

#defining main
def main(aps):
    spann =  list(itertools.product(N, D))
    #print(spann)
    e_metric_fixed_eps = partial(energy_fixed_metric, eps = aps)
    with Pool() as p:
        res_pool = p.starmap(e_metric_fixed_eps, spann) 
    metric = [] # error
    for it in res_pool:
        metric.append(it)
    return(metric)

if __name__ == "__main__":
    # Define a list of epsilon values
    #eps_values = [1e-4, 1e-3, 1e-2, 1e-1]
    
    eps_values = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    # Container for minimum metrics per epsilon
    min_metrics = {}

    for epss in eps_values:
        # Update epsilon
        
        # Compute metrics
        metrics = main(epss)

        # Reshape the result
        met_arr = np.reshape(metrics, (len(N), len(D)))

        # Find minimum metric values for each Delta
        min_metric_per_delta = np.max(met_arr, axis=0)
        min_metrics[epss] = min_metric_per_delta

    fig, ax1 = plt.subplots(figsize=(7, 6))

    for eps, min_metric_per_delta in min_metrics.items():
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        ax1.plot(D, min_metric_per_delta, label=label)
    ax1.axhline(0.99, linestyle='--', color='k', label=r'$\mathcal{A} = 0.01$')
    ax1.set_ylim([0, 1.1])
    ax1.set_xlabel(r'$ \Delta = N_{it} \times N_g$')
    ax1.set_ylabel(r'$ \max_{N_g} (\mathcal{M}_\Delta )$')
    ax1.tick_params(axis='y')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    #for eps, min_metric_per_delta in min_metrics.items():
    #    ax2.plot(D, 1 - min_metric_per_delta, linestyle='--', alpha=0.7, color='tab:red')
    ax2.set_ylabel(r'$ \min_{N_g}~(E(N_g)_\Delta) - E_{gs} $')
    
    # Set secondary y-axis ticks to 1 - primary y-axis ticks
    def accuracy_ticks(x, pos):
        return f"{1-x:.2f}"

    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(accuracy_ticks))
    ax2.tick_params(axis='y')
    
    # After you create each axes, set the formatter. Example for the main figure with ax1/ax2 and inset_ax:
    ax1.xaxis.set_major_formatter(FuncFormatter(pow10_formatter))
    #ax1.yaxis.set_major_formatter(FuncFormatter(pow10_formatter))
    #ax2.yaxis.set_major_formatter(FuncFormatter(pow10_formatter))

    ax1.legend(ncol=3, bbox_to_anchor=(1.15,1.30))
    ax1.grid()
    #ax1.set_yscale('log')
    #plt.savefig("Figs/Error_vs_fixed_resources.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    # Compute and plot slope for each epsilon
    fig, ax = plt.subplots(figsize=(7, 6))
    for eps, min_metric_per_delta in min_metrics.items():
        # Compute slope (derivative) with respect to Delta
        slope = np.gradient(min_metric_per_delta, D)
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        plt.plot(D, slope, label=label)
    plt.ylim([10**(-11), 10**(-3.1)])
    plt.yscale('log')
    plt.xlabel(r'$ \Delta = N_{it} \times N_g$')
    plt.gca().yaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
    #plt.xscale('log')
    plt.ylabel(r' $\log(d\eta) = \log(d(\mathcal{M}_\Delta)/d\Delta)$')
    plt.legend(ncol=3, bbox_to_anchor=(1.15,1.25))
    plt.grid()
    #plt.title("Slope of Metric vs. Delta for Different $\epsilon$")
    #plt.savefig("Slope_vs_fixed_resources.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    # Compute and plot efficiency not slope for each epsilon
    fig, ax = plt.subplots(figsize=(7, 6))
    for eps, min_metric_per_delta in min_metrics.items():
        # Compute slope (derivative) with respect to Delta
        eff = np.divide(min_metric_per_delta, D)
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        plt.plot(D, eff, label=label)
    plt.yscale('log')
    plt.xlabel(r'$ \log(\Delta)$')
    plt.gca().yaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
    plt.xscale('log')
    plt.ylabel(r' $\log(\eta) = \log(\mathcal{M}_\Delta/\Delta)$')
    plt.legend(ncol=3, bbox_to_anchor=(1.15,1.25))
    plt.grid()
    plt.xlim([4e3, 1e6])
    #plt.title("Slope of Metric vs. Delta for Different $\epsilon$")
    #plt.savefig("Figs/Efficiency_vs_fixed_resources.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    # Optionally, plot efficiency as a function of metric (parametric plot)
    fig, ax = plt.subplots(figsize=(7, 6))
    for eps, min_metric_per_delta in min_metrics.items():
        eff = np.divide(min_metric_per_delta, D)
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        plt.plot(min_metric_per_delta, eff, label=label)
    plt.axvline(0.99, linestyle='--', color='k', label=r'$\mathcal{A} = 0.01$')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
    plt.xlim([0, 1.1])  # Adjust x-axis limits for better visibility
    #plt.xscale('log')
    plt.xlabel(r'$\mathcal{M}_\Delta$')
    plt.ylabel(r' $\log(\eta) = \log(\mathcal{M}_\Delta/\Delta)$')
    plt.legend(ncol=3, bbox_to_anchor=(0.5,1.25))
    plt.grid()
    #plt.title("Slope vs. Metric for Different $\epsilon$")
    #plt.savefig("Efficiency_vs_fixed_metric.pdf", bbox_inches='tight', dpi=300)
    plt.show()

    #final figure with inset
    fig, ax1 = plt.subplots(figsize=(7, 6))

    for eps, min_metric_per_delta in min_metrics.items():
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        ax1.plot(D, min_metric_per_delta, label=label)
    ax1.axhline(0.99, linestyle='--', color='k', label=r'$\mathcal{A} = 0.01$')
    #ax1.axhline(0.99, linestyle='--', color='k')
    #ax1.set_ylim([0, 1.1])
    ax1.set_xlabel(r'$ \Delta = N_{it} \times N_g$')
    ax1.set_ylabel(r'$ \max_{N_g} (\mathcal{M}_\Delta )$')
    ax1.tick_params(axis='y')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$ \min_{N_g}~(E(N_g)_\Delta) - E_{gs} $')

    def accuracy_ticks(x, pos):
        return f"{1-x:.2f}"

    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(accuracy_ticks))
    ax2.tick_params(axis='y')

    ax1.legend(ncol=3, bbox_to_anchor=(1.15,1.25))
    ax1.grid()

    # Inset: Slope vs Delta
    inset_ax = ax1.inset_axes([0.4, 0.1, 0.5, 0.5])  # [x0, y0, width, height] in axes fraction
    for eps, min_metric_per_delta in min_metrics.items():
        slope = np.gradient(min_metric_per_delta, D)
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\epsilon = {coef}$"
        else:
            label = f"$\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        inset_ax.plot(D, slope, label=label)
    inset_ax.set_yscale('log')
    inset_ax.yaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
    inset_ax.set_xlabel(r'$\Delta$', fontsize=12)
    inset_ax.set_ylabel(r'$\log(\mathrm{Slope})$', fontsize=12)
    inset_ax.tick_params(axis='both', which='major', labelsize=8)
    inset_ax.grid(True, alpha=0.5)
    # Optionally, add a legend for the inset
    # inset_ax.legend(fontsize=7)

# ...existing code...
    #final figure with inset
    fig, ax1 = plt.subplots(figsize=(7, 6))

    for eps, min_metric_per_delta in min_metrics.items():
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            label = f"$\\epsilon = {coef}$"
        else:
            label = f"$\\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        ax1.plot(D, min_metric_per_delta, label=label)
    ax1.axhline(0.99, linestyle='--', color='k', label=r'$\mathcal{A} = 0.01$')
    #ax1.axhline(0.99, linestyle='--', color='k')
    ax1.set_xlim([-1,2e5])
    ax1.set_xlabel(r'$ \Delta = N_{it} \times N_g$')
    ax1.set_ylabel(r'$ \max_{N_g} (\mathcal{M}_\Delta )$')
    ax1.tick_params(axis='y')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$ \min_{N_g}~(E(N_g)_\Delta) - E_{gs} $')

    def accuracy_ticks(x, pos):
        return f"{1-x:.2f}"

    ax2.set_ylim(ax1.get_ylim())
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(accuracy_ticks))
    ax2.tick_params(axis='y')

    # format main x axis with scientific formatter (optional)
    ax1.xaxis.set_major_formatter(FuncFormatter(pow10_formatter))

    ax1.legend(ncol=3, bbox_to_anchor=(1.15,1.25))
    ax1.grid()

    # Inset: Efficiency vs Delta (replaced slope inset)
    inset_ax = ax1.inset_axes([0.37, 0.12, 0.6, 0.6])  # [x0, y0, width, height] in axes fraction
    for eps, min_metric_per_delta in min_metrics.items():
        eff = np.divide(min_metric_per_delta, D)
        coef, exp = "{:.0e}".format(eps).split("e")
        if eps == 0:
            ilabel = f"$\\epsilon = {coef}$"
        else:
            ilabel = f"$\\epsilon = {coef} \\times 10^{{{int(exp)}}}$"
        inset_ax.plot(D, eff, label=ilabel)

    inset_ax.set_xscale('log')
    inset_ax.set_yscale('log')
    inset_ax.xaxis.set_major_formatter(FuncFormatter(pow10_formatter))
    inset_ax.yaxis.set_major_formatter(mticker.LogFormatterExponent(base=10))
    inset_ax.set_xlabel(r'$\Delta$', fontsize=16, labelpad=0.5)
    inset_ax.tick_params(axis='both', which='major', labelsize=16)
    inset_ax.set_ylabel(r'$log(\eta) = log(\mathcal{M}_\Delta / \Delta$)', fontsize=16)
    #inset_ax.tick_params(axis='both', which='major', labelsize=8)
    inset_ax.grid(True, alpha=0.75)
    # optional compact inset legend
    # inset_ax.legend(fontsize=7, loc='upper right')

    plt.savefig("Figs/Error_vs_fixed_resources_with_inset.pdf", bbox_inches='tight', dpi=300)
    plt.show()
# ...existing code...