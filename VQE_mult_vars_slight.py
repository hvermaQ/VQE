from multiprocessing import Pool
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import matplotlib.ticker as ticker


delta = -7.71
#alp0 = 1.0589003542176467
#omega = 0.6709446472822442 
#E0 = 4.1054927116664945
kappa = 0.02
#E000 = 0.0643109474333867 #change for different noise
#E001 = 0.5706839345786814 #change for different noise
eps = 10**(-5)

gam1 = 1.55 #nu_0
omg1 = 0.71 #delta
E0INF = 4.1 #beta
E00 = 2 #alpha

plt.rcParams.update({'font.size': 18})

#EXPRESSION FOR GENERAL SOLN OF METRIC AS A FUNCTION OF T AND NG
#def final_energy(NN, tt):
#    Einf1 = (E000 * (NN**(E001))) *np.exp(-(alp0 * (NN**(-omega))) * tt)+ ((1 - eps)**((NN/2) - 1)) * (E0*np.exp(-kappa*NN) + delta) - delta
#    prod = NN*tt
#    return([Einf1, prod])

def final_energy(NN, tt):
    Einf1 = ((1 - eps)**(NN)) * (E0INF*np.exp(-kappa*NN) + delta + E00 *np.exp(-gam1*tt*(NN**(-omg1)))) - delta
    prod = NN*tt
    return([Einf1, prod])

t = np.linspace(1, 1001, 100, dtype = int)
N = np.linspace(1, 1001, 100, dtype = int)

contour_vals = [0.05, 0.1, 0.25, 0.30 , 0.35,  0.4, 0.45, 0.48, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

#defining main
def main():
    spann =  list(itertools.product(N, t))
    #print(spann)
    with Pool() as p:
        res_pool = p.starmap(final_energy, spann) 
    metric = [] # error
    energy = [] # energy
    for it in res_pool:
        metric.append(it[0])
        energy.append(it[1])
    return(metric, energy)

if __name__=="__main__": 
    a, b = main()

    met_arr = np.reshape(a, (int(len(N)), int(len(t))))
    eng_arr = np.reshape(b, (int(len(N)), int(len(t))))
    eff_arr = np.log10(np.divide(met_arr, eng_arr))
    
    fmt = ticker.LogFormatterSciNotation()
    fmt.create_dummy_axis()
    # figsize=(6, 6) control width and height
    # dpi = 600, I 
    fig, ax = plt.subplots(figsize=(13, 10))
    #plt.figure(figsize=(13, 10), dpi = 600) 
    #labels = b
    sns.set(font_scale=2) 
    axz = sns.heatmap(eng_arr, cmap="rocket", cbar_kws={'label': 'Total algorithmic resources'})
    
    for contour_val in contour_vals:
        c = ax.contour(np.arange(.5, met_arr.shape[1]), np.arange(.5, met_arr.shape[0]), met_arr, [contour_val,], colors='yellow', alpha = 0.9,  linestyles = 'dotted')
        ax.clabel(c, [contour_val, ], inline=1, fontsize=15, manual = [(1,1)])
    

    c4 = ax.contour(np.arange(.5, eng_arr.shape[1]), np.arange(.5, eng_arr.shape[0]), eng_arr, [10000, 25000, 50000, 100000, 200000], colors='cyan', alpha = 0.5, linestyles = 'dashed')
    ax.clabel(c4, [10000, 25000, 50000, 100000, 200000], inline=1, fmt = fmt)
    
    #ax.set(xticklabels=t)
    ax.set_xticks(np.linspace(1, 100, 10))
    ax.set_xticklabels(np.linspace(1, 1001, 10, dtype = int))
    ax.set_yticks(np.linspace(1, 100, 10))
    ax.set_yticklabels(np.linspace(1, 1001, 10, dtype = int))
    
    #ax.set_yticks(range(1, 1001, 99))
    #ax.set_yticklabels(np.arange(1, 1001, 99, dtype = int))
    #ax.set(yticklabels=N)
    ax.invert_yaxis()
    #plt.xticks(a)
    #ax.yticks(b)
    #plt.title(r'$E = t \times N_g,~\epsilon = %s$'%eps)
    plt.title(r'$\epsilon = 10^{%s}$'%int(np.log10(eps)))
    plt.xlabel(r'$ t$')
    plt.ylabel(r'$ N_g$')
    #ax.legend(loc='upper left')
    # Add colorbar
    # Access colorbar and format the tick labels
    cbar = axz.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    #plt.savefig("en_met_eps5_HVA.pdf", dpi = 100)
    plt.show()

    energy = []
    for contour_val in contour_vals:
        indices = np.where(np.isclose(met_arr, contour_val, atol=1e-2))
        eng_at_contour = eng_arr[indices]
        min_eng = np.min(eng_at_contour)
        energy.append(min_eng)
    
    plt.plot(contour_vals, energy, marker='o')
    plt.show()