#appendages: noise modeller, scipy wrapper

#custom scipy optimization wrapper : Junction in Qaptiva
from qat.plugins import Junction
from scipy.optimize import minimize
from qat.core.plugins import AbstractPlugin
import numpy as np
from qat.core import Result

#gateset for counting gates to introduce noise through Gaussian noise plugin
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
two_qb_gateset = ['CNOT', 'CSIGN']  
gateset = one_qb_gateset + two_qb_gateset

#add id insertion mode

class Opto(Junction):
    def __init__(self, x0: np.ndarray = None, tol: float = 1e-8, maxiter: int = 25000, nbshots: int = 0,):
        super().__init__(collective=False)
        self.x0 = x0
        self.maxiter = maxiter
        self.nbshots = nbshots
        self.n_steps = 0
        self.energy_optimization_trace = []
        self.parameter_map = None
        self.energy = 0
        self.energy_result = Result()
        self.tol = tol
        self.c_steps = 0
        self.int_energy = []

    def run(self, job, meta_data):
        
        if self.x0 is None:
            self.x0 = 2*np.pi*np.random.rand(len(job.get_variables()))
            self.parameter_map = {name: x for (name, x) in zip(job.get_variables(), self.x0)}

        def compute_energy(x):
            job_bound =  job(** {v: xx for (v, xx) in zip(job.get_variables(), x)})
            self.energy = self.execute(job_bound)
            self.energy_optimization_trace.append(self.energy.value)
            self.n_steps += 1
            return self.energy.value

        def cback(intermediate_result):
            #fn =  compute_energy(intermediate_result)
            self.int_energy.append(intermediate_result.fun)
            self.c_steps += 1
            #return(fn)

        bnd = (0, 2*np.pi)
        bnds = tuple([bnd for i in range(len(job.get_variables()))])
        #res = minimize(compute_energy, x0 = self.x0, method='L-BFGS-B', bounds = bnds, callback = cback , options={'ftol': self.tol, 'disp': False, 'maxiter': self.maxiter})
        res = minimize(compute_energy, x0 = self.x0, method='COBYLA', bounds = bnds, options={'tol': self.tol, 'disp': False, 'maxiter': self.maxiter})
        en = res.fun
        self.parameter_map =  {v: xp for v, xp in zip(job.get_variables(), res.x)}
        self.energy_result.value = en
        self.energy_result.meta_data = {"optimization_trace": str(self.energy_optimization_trace), "n_steps": f"{self.n_steps}", "parameter_map": str(self.parameter_map), "c_steps" : f"{self.c_steps}", "int_energy": str(self.int_energy)}
        return (Result(value = self.energy_result.value, meta_data = self.energy_result.meta_data))


#custom gaussian noise plugin : Abstract plugin in qaptiva

class GaussianNoise(AbstractPlugin,):
    def __init__(self, p, hamiltonian_matrix):
        self.p = p
        self.hamiltonian_trace = np.trace(hamiltonian_matrix)/(np.shape(hamiltonian_matrix)[0])
        self.unsuccess = 0
        self.success = 0
        self.nb_pauli_strings = 0
        self.nbshots = 0
    
    @staticmethod
    def get_layer_duration(gate_name):
        """Get duration of a gate based on type"""
        if gate_name in one_qb_gateset:
            return 1
        return 2  # for two-qubit gates
    
    @staticmethod
    def can_add_to_layer(gate, current_layer, used_qubits):
        """Check if a gate can be added to current layer with improved parallelization"""
        gate_qubits = set(gate[2])
        gate_name = gate[0]
        
        # Quick check for qubit overlap
        if bool(gate_qubits & used_qubits):
            return False
        
        for existing_gate, _ in current_layer:
            existing_name = existing_gate[0]
            existing_qubits = set(existing_gate[2])
            
            # Check RZ-CNOT commutation more precisely
            if gate_name == 'RZ' and existing_name == 'CNOT':
                rz_qubit = list(gate_qubits)[0]
                control, target = existing_gate[2]
                # RZ commutes with CNOT if not on target
                if rz_qubit != target:
                    return True
                
            elif existing_name == 'RZ' and gate_name == 'CNOT':
                rz_qubit = list(existing_qubits)[0]
                control, target = gate[2]
                # RZ on target qubit doesn't commute with CNOT
                if rz_qubit == target:
                    return False
                # RZ on control or other qubits can be parallel
                continue
                
            # Improve CNOT-CNOT parallelization
            if gate_name == 'CNOT' and existing_name == 'CNOT':
                gate_control, gate_target = gate[2]
                existing_control, existing_target = existing_gate[2]
                # Allow parallel CNOTs if completely independent
                if not (set([gate_control, gate_target]) & 
                        set([existing_control, existing_target])):
                    return True
                return False
                
            # Handle H gates (don't commute with CNOT)
            if (gate_name == 'H' and existing_name == 'CNOT') or \
            (existing_name == 'H' and gate_name == 'CNOT'):
                if bool(gate_qubits & existing_qubits):
                    return False
                    
            # Handle parallel single-qubit gates
            if gate_name in ['RZ', 'H', 'X'] and existing_name in ['RZ', 'H', 'X']:
                continue
                
        return True

    @staticmethod
    def get_parallel_layers(circuit):
        """Group gates into parallel execution layers with improved compression"""
        gates = list(circuit.iterate_simple())
        layers = []
        current_layer = []
        used_qubits = set()
        
        for gate in gates:
            gate_name = gate[0]
            
            # Try to add to current layer
            if current_layer and GaussianNoise.can_add_to_layer(gate, current_layer, used_qubits):
                current_layer.append((gate, GaussianNoise.get_layer_duration(gate_name)))
                used_qubits.update(gate[2])
            else:
                # Try to add to previous layers
                added_to_previous = False
                for layer in reversed(layers):
                    layer_qubits = set().union(*[set(g[0][2]) for g in layer])
                    if GaussianNoise.can_add_to_layer(gate, layer, layer_qubits):
                        layer.append((gate, GaussianNoise.get_layer_duration(gate_name)))
                        added_to_previous = True
                        break
                
                if not added_to_previous:
                    # Start new layer if couldn't add to any existing layer
                    if current_layer:
                        layers.append(current_layer)
                    current_layer = [(gate, GaussianNoise.get_layer_duration(gate_name))]
                    used_qubits = set(gate[2])
        
        if current_layer:
            layers.append(current_layer)
        
        return layers

    #call total gates to provide identity in addition to circuit gates
    @staticmethod
    def id_gates(batch):
        # Get parallel execution layers
        circuit = batch.jobs[0].circuit
        parallel_layers = GaussianNoise.get_parallel_layers(circuit)
        n_layers = len(parallel_layers)
        nqbt = circuit.nbqbits
        # Calculate duration per qubit considering parallel execution
        qubit_durations = {i: 0.0 for i in range(nqbt)}
        
        # For each layer, update the duration for involved qubits
        for layer in parallel_layers:
            layer_duration = max(gate[1] for gate in layer)  # Max gate time in layer
            for gate, _ in layer:
                for qubit in gate[2]:  # gate[2] contains qubit indices
                    qubit_durations[qubit] += layer_duration
        
        # Find maximum duration across all qubits
        max_duration = max(qubit_durations.values())
        
        # Calculate required identity gates for each qubit
        ids = 0
        for qubit, duration in qubit_durations.items():
            time_difference = max_duration - duration
            if time_difference > 0:
                # Convert time difference to number of identity gates needed
                num_identities = int(np.ceil(time_difference))
                ids += num_identities
        return ids

    def compile(self, batch, _):
        self.nbshots =  batch.jobs[0].nbshots
        #nb_gates = batch.jobs[0].circuit.depth({'CNOT' : 2, 'RZ' : 1, 'H' : 1}, default = 1)
        nb_gates = sum([batch.jobs[0].circuit.count(yt) for yt in gateset]) + GaussianNoise.id_gates(batch)
        self.success = abs((1-self.p)**nb_gates)
        self.unsuccess = (1-self.success)*self.hamiltonian_trace
        return batch 
    
    def post_process(self, batch_result):
        if batch_result.results[0].value is not None:
            for result in batch_result.results:
                if self.nbshots == 0:
                    noise =  self.unsuccess
                else: 
                    noise =  np.random.normal(self.unsuccess, self.unsuccess/np.sqrt(self.nbshots))
                result.value = self.success*result.value + noise
        return batch_result
