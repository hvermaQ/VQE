# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 13:54:14 2025

@author: cqtv201
"""
#modification for multiple levels of noise required
#modification for circuit generation using Hamiltonian required

from qat.core import Observable, Term, Batch #Hamiltonian
from qat.qpus import get_default_qpu
from qat.fermion import SpinHamiltonian
import numpy as np
from multiprocessing import Pool, cpu_count
from circ_gen import gen_circ_RYA, gen_circ_HVA
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

from opto_gauss import Opto, GaussianNoise

import seaborn as sns
from scipy import stats
from scipy.constants import hbar
from tqdm import tqdm

plt.rcParams.update({'font.size': 16})  # Set global font size

optimizer = Opto()
qpu = get_default_qpu()

#gateset for counting algorithmic resources
def gateset():
    #gateset for counting gates to introduce noise through Gaussian noise plugin
    one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    two_qb_gateset = ['CNOT', 'CSIGN']  
    gates = one_qb_gateset + two_qb_gateset
    return gates

def create_observable(type, nqbts):
    if type == "Heisenberg":
        #Instantiation of Hamiltoniian
        heisen = Observable(nqbts)
        #Generation of Heisenberg Hamiltonian
        for q_reg in range(nqbts-1):
            heisen += Observable(nqbts, pauli_terms = [Term(1., typ, [q_reg,q_reg + 1]) for typ in ['XX','YY','ZZ']])  
        obs = heisen
    return obs

def exact_Result(obs):
    obs_class = SpinHamiltonian(nqbits=obs.nbqbits, terms=obs.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    g_energy = eigvals[0]
    return g_energy

def submit_job_wrapper(args):
    """Wrapper function for parallel job submission that recreates required objects"""
    nqbts, dep, rnd, ans, ham, n_params = args
    #everything based on i
    if ans == "RYA":
        circuit = gen_circ_RYA((nqbts, dep))
    elif ans == "HVA":
        circuit = gen_circ_HVA((nqbts, dep))
    obss = create_observable(ham, nqbts)
    job = circuit.to_job(observable=obss, nbshots=0)
    obs_mat = obss.to_matrix().A
    # Create fresh instances in worker process
    stack = optimizer | GaussianNoise(n_params[0], n_params[1], obs_mat) | qpu
    result = stack.submit(job)
    print(result.meta_data["n_steps"])
    return (nqbts, result.value, result.meta_data["n_steps"])

def run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params):
    print("Parallelizing")
    # Prepare arguments for parallel processing
    job_args = []
    for i in range(len(problem_set)):
        for rnd in range(rnds):
            # Only pass serializable data
            job_args.append((
                problem_set[i][0], 
                problem_set[i][1],
                rnd,
                ansatz,
                observe,
                noise_params
            ))
    
    num_processes = min(cpu_count(), len(job_args))
    print(f"Using {num_processes} processes")
    
    #with Pool(processes=num_processes, initializer=MB_benchmark.init_worker) as pool:
    with Pool(processes=num_processes) as pool:
        # Use imap with tqdm for progress tracking
        result_async = pool.map_async(submit_job_wrapper, job_args)
        results = result_async.get()
            
    print("Parallel done")
    
    # Process results
    avg_results_per_size = []
    variance_results_per_size = []
    n_iterations = []
    
    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [(val, iters) for nqb, val, iters in results if  nqb == problem_set[i][0]]
        values = [r[0] for r in size_results]
        iterations = [int(r[1]) for r in size_results]
        
        avg_results_per_size.append(np.mean(values))
        variance_results_per_size.append(np.var(values))
        n_iterations.append(np.sum(iterations))

    return (avg_results_per_size, variance_results_per_size, n_iterations)

def run_serial_jobs(problem_set, rnds, ansatz, observe, noise_params):
    print("Running jobs serially")
    #MB_benchmark.init_worker()
    
    results = []
    for i in range(len(problem_set)):
        for rnd in range(rnds):
            # Only pass serializable data
            args = (
                problem_set[i][0], 
                problem_set[i][1],
                rnd,
                ansatz,
                observe,
                noise_params
            )
            # Use existing static submit_job_wrapper
            result = submit_job_wrapper(args)
            results.append(result)
    
    print("Processing complete")
    
    # Process results
    avg_results_per_size = []
    variance_results_per_size = []
    n_iterations = []
    
    # Group results by problem size
    for i in range(len(problem_set)):
        size_results = [(val, iters) for nqb, val, iters in results if nqb == problem_set[i][0]]
        #print(size_results)
        values = [r[0] for r in size_results]
        iterations = [int(r[1]) for r in size_results]
        
        avg_results_per_size.append(np.mean(values))
        variance_results_per_size.append(np.var(values))
        n_iterations.append(np.sum(iterations))
    return (avg_results_per_size, variance_results_per_size, n_iterations)

def benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, known_size, hw):
    print("Benchmarking Main")
    problem_set = list(zip(nqbits, depths))
    sim_results, sim_variance, sim_iterations = run_parallel_jobs(problem_set, rnds, ansatz, observe, noise_params)

    # Perform linear regression and extrapolation
    projected_results = linear_extrapolation(nqbits, sim_results, sim_variance, target_size=known_size)
    projected_value = projected_results['extrapolated_value']

    # Calculate the known result
    obss = create_observable(observe, known_size)
    obs_class = SpinHamiltonian(nqbits=obss.nbqbits, terms=obss.terms)
    obs_mat = obs_class.get_matrix()
    eigvals, _ = np.linalg.eigh(obs_mat)
    known_res = eigvals[0]

    error = np.abs(projected_value - known_res)
    error_bars = projected_results['extrapolated_error']

    # Calculate algorithmic resources
    algo_resources = 0
    for i in range(len(problem_set)):
        obss = create_observable(observe, nqbits[i])
        if ansatz == "RYA":
            circuit = gen_circ_RYA((nqbits[i], depths[i]))
        elif ansatz == "HVA":
            circuit = gen_circ_HVA((nqbits[i], depths[i]))
        pauls = len(obss.terms)
        gates_count = sum([circuit.count(yt) for yt in gateset()])
        algo_resources += pauls * gates_count * sim_iterations[i] * nshots

    algo_eff = error / algo_resources
    if hw is not None:
        hw_res = hardware_resource(algo_resources)
        hw_eff = error / hw_res
        print("Hardware resources = %f" % hw_res)
        print("Hardware efficiency = %f" % hw_eff)
    else:
        hw_res = None
        hw_eff = None

    print("Algorithmic resources = %f" % algo_resources)
    print("Algorithmic efficiency = %f" % algo_eff)
    print("Projected value = %f" % projected_value)
    print("Error using the exact value = %f" % error)
    print("Exact value = %f" % known_res)

    # Pass the dynamically calculated degree to the plotting function
    plot_results(nqbits, sim_results, sim_variance, projected_results, known_size)
    return sim_results, sim_variance, projected_results, error, error_bars, projected_results


#benchmark(nqbits, depths, rnds, ansatz, observe, noise_params, nshots, thermal_size, thermodynamic_limit, hw):
#code to generate result. Especially required for using multiprocessing correctly
if __name__ == '__main__':
    nqbits = [3, 4, 5, 6, 7]
    deps = [3, 4, 5, 6, 7] #ensure the depths are same in the number as qubits
    rnd = 4 #random seeds
    noise_p = [0.0001, -1]
    therm_size = 8 #size of the system to be used for extrapolation
    a, b, c, d, e, f = benchmark(nqbits, deps, rnd, 'RYA', 'Heisenberg', noise_p, 1000, therm_size, 'supercond')
    print("Completed")