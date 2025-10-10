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
#eps = 10**(-4)

#base_expo = [[1,6], [1,5], [5,5], [1,4], [5,4], [1,3]]
base_expo = [[1,5]]
eps_arr = [b[0]*(10**(-b[1])) for b in base_expo]

gam1 = 1.55
omg1 = 0.84
E0INF = 4.1
E00 = 2

plt.rcParams.update({'font.size': 16})

def final_energy(NN, tt, eps):
    Einf1 = ((1 - eps)**(NN)) * (E0INF*np.exp(-kappa*NN) + delta + E00 *np.exp(-gam1*tt*(NN**(-omg1)))) - delta
    prod = NN*tt
    return([Einf1, prod])

#t = np.arange(1, 1001, 9, dtype = int)
#N = np.arange(1, 1001, 9, dtype = int)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
t = np.linspace(1, 1001, 100, dtype = int)
N = np.linspace(1, 1001, 100, dtype = int)

#defining main
def main():
    spann = list(itertools.product(N, t))
    results = {}  # Dictionary to store results for each eps_err

    for err in eps_arr:
        with Pool() as p:
            res_pool = p.starmap(final_energy, [(n, tt, err) for n, tt in spann])
        
        metric = []  # error in VQE energy
        energy = []  # energy
        efficiency = []
        
        for it in res_pool:
            metric.append(it[0])
            energy.append(it[1])
            efficiency.append(np.log10(np.divide(it[0], it[1])))
        
        # Reshape arrays for easier processing
        metric_arr = np.reshape(metric, (len(N), len(t)))
        energy_arr = np.reshape(energy, (len(N), len(t)))
        efficiency_arr = np.reshape(efficiency, (len(N), len(t)))
        
        # Dynamically select a set of metrics based on the minimum error metric
        #max_metric = np.min(metric_arr)
        
        #selected_metrics = np.linspace( max_metric, 10*max_metric, num=10)  # Select 10 evenly spaced metrics
        selected_metrics = [0.05, 0.1, 0.2, 0.25, 0.30, 0.35, 0.4]
        # Store results for each selected metric
        #max efficiency for each metric
        #minimimum resource for each metric (should be same as max efficiency, as metric is fixed)
        selected_results = []
        for fixed_metric in selected_metrics:
            # Use np.where to find indices close to the fixed metric
            indices = np.where(np.isclose(metric_arr, fixed_metric, atol=1e-3))
            #condition = np.abs(metric_arr - fixed_metric) < 1e-5
            #indices = np.argwhere(condition)
            if indices[0].size > 0:  # Ensure there are matching indices
                # Find the maximum efficiency for the selected metric
                efficiencies_at_metric = efficiency_arr[indices]
                max_efficiency = np.max(efficiencies_at_metric)
                
                #resources at metric
                resources_at_metric = energy_arr[indices]
                min_resources = np.min(resources_at_metric) #to cross check

                # Get the corresponding (N, t) arguments
                #max_eff_index = np.argmax(efficiencies_at_metric)
                #max_N = N[indices[0][max_eff_index]]
                #max_t = t[indices[1][max_eff_index]]

                # Append the result as [fixed_metric, max_efficiency, max_N, max_t]
                selected_results.append([fixed_metric, max_efficiency, min_resources])
        
        # Store the results for the current eps_err
        results[err] = selected_results
    
    return results

def plot_results(res):
    #plots needed
    #max efficiency vs metric of performance for different eps
    #max metric vs resource for different eps
    plt.figure(figsize=(10, 6))
    for eps_err, data in res.items():
        # Unpack the data
        data = np.array(data)
        metrics = data[:, 0]
        efficiencies = data[:, 1]
        # Plotting max efficiency vs fixed metric
        plt.plot(metrics, efficiencies, label=f'eps={eps_err:.1e}')
    plt.xlabel('Fixed Metric')
    plt.ylabel('log(Max Efficiency)')
    plt.title('Max Efficiency vs Fixed Metric')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    for eps_err, data in res.items():
        # Unpack the data
        data = np.array(data)
        #plotting min resources vs fixed metric
        metrics = data[:, 0]
        resources = data[:, 2]
        plt.plot(metrics, resources, label=f'eps={eps_err:.1e}')
    plt.xlabel('Fixed Metric')
    plt.ylabel('Min Resources')
    plt.yscale('log')
    plt.title('Min Resources vs Fixed Metric')
    plt.legend()
    plt.grid()
    plt.show()
    return

if __name__=="__main__": 
    ress = main()
    plot_results(ress)