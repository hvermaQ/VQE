from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np

nqbt = 5 # number of qubits

#complicated mathod, checks if gates can be parallelized

#gateset for counting gates
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'I']
two_qb_gateset = ['CNOT', 'CSIGN']  

dep = np.arange(2, 3, 1, dtype = int)

#gate counting routine
def gate_ct(in_circ):
    N_1qb = 0
    N_2qb = 0
    for tt1 in one_qb_gateset:
        N_1qb += in_circ.count(tt1)
    for tt2 in two_qb_gateset:
        N_2qb += in_circ.count(tt2)
    return(N_1qb, N_2qb)

def get_layer_duration(gate_name):
    """Get duration of a gate based on type"""
    if gate_name in one_qb_gateset:
        return 1
    return 2  # for two-qubit gates

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

def get_parallel_layers(circuit):
    """Group gates into parallel execution layers with improved compression"""
    gates = list(circuit.iterate_simple())
    layers = []
    current_layer = []
    used_qubits = set()
    
    for gate in gates:
        gate_name = gate[0]
        
        # Try to add to current layer
        if current_layer and can_add_to_layer(gate, current_layer, used_qubits):
            current_layer.append((gate, get_layer_duration(gate_name)))
            used_qubits.update(gate[2])
        else:
            # Try to add to previous layers
            added_to_previous = False
            for layer in reversed(layers):
                layer_qubits = set().union(*[set(g[0][2]) for g in layer])
                if can_add_to_layer(gate, layer, layer_qubits):
                    layer.append((gate, get_layer_duration(gate_name)))
                    added_to_previous = True
                    break
            
            if not added_to_previous:
                # Start new layer if couldn't add to any existing layer
                if current_layer:
                    layers.append(current_layer)
                current_layer = [(gate, get_layer_duration(gate_name))]
                used_qubits = set(gate[2])
    
    if current_layer:
        layers.append(current_layer)
    
    # Print layer information for debugging
    #print(f"\nCircuit layering information:")
    #for i, layer in enumerate(layers):
    #    print(f"Layer {i}: {[(g[0][0], g[0][2]) for g in layer]}")
    
    return layers

def display_parallel_circuit(parallel_layers, num_qubits):
    """Display circuit with parallel gates shown in columns"""
    
    # Initialize the display grid
    grid = [[' ' for _ in range(len(parallel_layers) * 4)] for _ in range(num_qubits)]
    lines = [['-' for _ in range(len(parallel_layers) * 4)] for _ in range(num_qubits)]
    
    # Gate symbols
    symbols = {
        'H': 'H',
        'X': 'X',
        'RZ': 'R',
        'CNOT': 'C'
    }
    
    # Fill the grid with gates
    for layer_idx, layer in enumerate(parallel_layers):
        col_idx = layer_idx * 4
        used_qubits = set()
        
        for gate, duration in layer:
            gate_name = gate[0]
            gate_qubits = gate[2]
            
            if gate_name == 'CNOT':
                control, target = gate_qubits
                grid[control][col_idx] = 'C'
                grid[target][col_idx] = '+'
                # Draw vertical line
                for q in range(min(control, target) + 1, max(control, target)):
                    grid[q][col_idx] = '│'
            else:
                qubit = gate_qubits[0]
                symbol = symbols.get(gate_name, gate_name[0])
                grid[qubit][col_idx] = symbol
            
            used_qubits.update(gate_qubits)
        
        # Mark layer boundary
        for q in range(num_qubits):
            if grid[q][col_idx] == ' ':
                grid[q][col_idx] = '·'
    
    # Print the circuit
    print("\nParallel Circuit Display:")
    print("-" * (len(parallel_layers) * 4))
    
    for q in range(num_qubits):
        # Print qubit line
        print(f"q{q}: ", end='')
        for col in range(len(parallel_layers) * 4):
            print(grid[q][col], end='')
        print()
    
    print("-" * (len(parallel_layers) * 4)
    print(f"Total layers: {len(parallel_layers)}")

for ct in dep:
    qprog = Program()
    qbits = qprog.qalloc(nqbt)
    #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
    ao = [qprog.new_var(float, 'ao_%s'%i) for i in range(ct)]
    bo = [qprog.new_var(float, 'bo_%s'%i) for i in range(ct)]
    co = [qprog.new_var(float, 'co_%s'%i) for i in range(ct)]
    ae = [qprog.new_var(float, 'ae_%s'%i) for i in range(ct)]
    be = [qprog.new_var(float, 'be_%s'%i) for i in range(ct)]
    ce = [qprog.new_var(float, 'ce_%s'%i) for i in range(ct)]
    for q_index in range(nqbt):
        X(qbits[q_index])
    for q_index in range(nqbt):
        if not q_index%2 and q_index <= nqbt-1:
            H(qbits[q_index])
    for q_index in range(nqbt):
        if not q_index%2 and q_index <= nqbt-2:
            CNOT(qbits[q_index],qbits[q_index+1])
    for it in range(ct):
        for q_index in range(nqbt): #odd Rzz
            if q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ao[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #odd Ryy
            if q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                #I(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                #I(qbits[q_index])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(bo[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                #I(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
                #I(qbits[q_index])
        for q_index in range(nqbt): #odd Rxx
            if q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(co[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
        for q_index in range(nqbt): #even Rzz
            if not q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ae[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #even Ryy
            if not q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                #I(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                #I(qbits[q_index])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(be[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                #I(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
                #I(qbits[q_index])
        for q_index in range(nqbt): #even Rxx
            if not q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ce[it-1]/2)(qbits[q_index+1])
                #I(qbits[q_index])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
    circuit = qprog.to_circ()
    print("Original circuit:")
    circuit.display()
    #circuit.display()
    depth = circuit.depth({'CNOT' : 2, 'RZ' : 1, 'H' : 1, 'X' : 1})
    print(f"QLM Depth = {depth}")
    print(f"1qb = {gate_ct(circuit)[0]}, 2qb = {gate_ct(circuit)[1]}, total = {gate_ct(circuit)[0] + gate_ct(circuit)[1]}")
    # Get parallel execution layers
    parallel_layers = get_parallel_layers(circuit)
    print("Compiled circuit:")
    display_parallel_circuit(parallel_layers, nqbt)
    n_layers = len(parallel_layers)
    print(f"Compiled Layers = {n_layers}")
    print(f"Gates based on compiled layers = {n_layers*nqbt}")
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
    
    print(f"For Ansatz Depth {ct}: {ids} identity gates needed after basic compilation")
    print(f"gates+ids = {gate_ct(circuit)[0] + gate_ct(circuit)[1] + ids}")