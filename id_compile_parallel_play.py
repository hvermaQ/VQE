from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN
import numpy as np

nqbt = 5
dep = np.arange(2, 3, 1, dtype=int)

one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'I']
two_qb_gateset = ['CNOT', 'CSIGN']

def gate_ct(in_circ):
    N_1qb = sum(in_circ.count(g) for g in one_qb_gateset)
    N_2qb = sum(in_circ.count(g) for g in two_qb_gateset)
    return N_1qb, N_2qb

def get_layer_duration(gate_name):
    return 1 if gate_name in one_qb_gateset else 2

def can_add_to_layer(gate, current_layer, used_qubits):
    gate_qubits = set(gate[2])
    gate_name = gate[0]

    if bool(gate_qubits & used_qubits):
        return False

    for existing_gate, _ in current_layer:
        existing_name = existing_gate[0]
        existing_qubits = set(existing_gate[2])

        # RZ-CNOT commutation rules
        if gate_name == 'RZ' and existing_name == 'CNOT':
            rz_qubit = list(gate_qubits)[0]
            control, target = existing_gate[2]
            if rz_qubit != target:
                continue
            else:
                return False
        if gate_name == 'CNOT' and existing_name == 'RZ':
            rz_qubit = list(existing_qubits)[0]
            control, target = gate[2]
            if rz_qubit == target:
                return False
        # Independent CNOTs can be parallel
        if gate_name == 'CNOT' and existing_name == 'CNOT':
            if set(gate[2]) & set(existing_gate[2]):
                return False
    return True

def get_parallel_layers(circuit):
    gates = list(circuit.iterate_simple())
    layers = []
    current_layer = []
    used_qubits = set()

    for gate in gates:
        gate_name = gate[0]
        if current_layer and can_add_to_layer(gate, current_layer, used_qubits):
            current_layer.append((gate, get_layer_duration(gate_name)))
            used_qubits.update(gate[2])
        else:
            added_to_prev = False
            for layer in reversed(layers):
                layer_qubits = set().union(*[set(g[0][2]) for g in layer])
                if can_add_to_layer(gate, layer, layer_qubits):
                    layer.append((gate, get_layer_duration(gate_name)))
                    added_to_prev = True
                    break
            if not added_to_prev:
                if current_layer:
                    layers.append(current_layer)
                current_layer = [(gate, get_layer_duration(gate_name))]
                used_qubits = set(gate[2])
    if current_layer:
        layers.append(current_layer)
    return layers

def count_compiled_gates(parallel_layers):
    """Count single- and two-qubit gates in compiled (parallelized) circuit."""
    n1 = 0
    n2 = 0
    for layer in parallel_layers:
        for gate, _ in layer:
            if gate[0] in one_qb_gateset and gate[0] != 'I':
                n1 += 1
            elif gate[0] in two_qb_gateset:
                n2 += 1
    return n1, n2

def display_parallel_circuit_with_ID(parallel_layers, num_qubits):
    num_layers = len(parallel_layers)
    grid = [['.' for _ in range(num_layers)] for _ in range(num_qubits)]
    id_row = [0 for _ in range(num_layers)]
    qubit_duration_row = [0 for _ in range(num_layers)]  # total duration per layer

    qubit_time = np.zeros((num_qubits, num_layers), dtype=int)

    # Place gates in grid and calculate per-qubit layer durations
    for l_idx, layer in enumerate(parallel_layers):
        max_dur = max(g[1] for g in layer)
        qubit_dur_in_layer = np.zeros(num_qubits, dtype=int)
        for gate, dur in layer:
            qubits = gate[2]
            symbol = gate[0][0] if gate[0] != 'CNOT' else 'C'
            if gate[0] == 'CNOT':
                c, t = qubits
                grid[c][l_idx] = 'C'
                grid[t][l_idx] = '+'
                for q in range(min(c, t)+1, max(c, t)):
                    grid[q][l_idx] = '│'
                qubit_dur_in_layer[c] = dur
                qubit_dur_in_layer[t] = dur
            else:
                qubit = qubits[0]
                grid[qubit][l_idx] = symbol
                qubit_dur_in_layer[qubit] = dur
        # Compute IDs per layer
        for q in range(num_qubits):
            ids = max_dur - qubit_dur_in_layer[q]
            if ids > 0:
                id_row[l_idx] += ids
        qubit_duration_row[l_idx] = max_dur

    # Print ASCII
    print("\nParallel Circuit Display (with IDs and durations):")
    print("-" * (num_layers*4))
    for q in range(num_qubits):
        print(f"q{q}: ", end='')
        for col in range(num_layers):
            print(grid[q][col], end='  ')
        print()
    # ID row
    print("ID: ", end='')
    for val in id_row:
        print(val, end='  ')
    print()
    # Duration row
    print("Du: ", end='')
    for val in qubit_duration_row:
        print(val, end='  ')
    print()
    print("-" * (num_layers*4))
    total_ids = sum(id_row)
    return total_ids

# Main loop
for ct in dep:
    qprog = Program()
    qbits = qprog.qalloc(nqbt)
    ao = [qprog.new_var(float, f'ao_{i}') for i in range(ct)]
    bo = [qprog.new_var(float, f'bo_{i}') for i in range(ct)]
    co = [qprog.new_var(float, f'co_{i}') for i in range(ct)]
    ae = [qprog.new_var(float, f'ae_{i}') for i in range(ct)]
    be = [qprog.new_var(float, f'be_{i}') for i in range(ct)]
    ce = [qprog.new_var(float, f'ce_{i}') for i in range(ct)]

    # Initial gates
    for q_index in range(nqbt):
        X(qbits[q_index])
    for q_index in range(0, nqbt, 2):
        H(qbits[q_index])
    for q_index in range(0, nqbt-1, 2):
        CNOT(qbits[q_index], qbits[q_index+1])

    # Ansatz layers
    for it in range(ct):
        for q_index in range(1, nqbt-1, 2):  # odd Rzz
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(ao[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
        for q_index in range(1, nqbt-1, 2):  # odd Ryy
            RZ(np.pi/2)(qbits[q_index])
            RZ(np.pi/2)(qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(bo[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])
            RZ(-np.pi/2)(qbits[q_index])
            RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(1, nqbt-1, 2):  # odd Rxx
            H(qbits[q_index])
            H(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(co[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])
        for q_index in range(0, nqbt-1, 2):  # even Rzz
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(ae[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
        for q_index in range(0, nqbt-1, 2):  # even Ryy
            RZ(np.pi/2)(qbits[q_index])
            RZ(np.pi/2)(qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(be[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])
            RZ(-np.pi/2)(qbits[q_index])
            RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(0, nqbt-1, 2):  # even Rxx
            H(qbits[q_index])
            H(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            RZ(ce[it]/2)(qbits[q_index+1])
            CNOT(qbits[q_index], qbits[q_index+1])
            H(qbits[q_index])
            H(qbits[q_index+1])

    circuit = qprog.to_circ()
    circuit.display()
    depth = circuit.depth({'CNOT':2, 'RZ':1, 'H':1, 'X':1})
    print(f"\nQLM Depth = {depth}")
    n1, n2 = gate_ct(circuit)
    print(f"Original circuit gate count: 1qb={n1}, 2qb={n2}, total={n1+n2}")

    # Compile to parallel layers and display
    parallel_layers = get_parallel_layers(circuit)
    total_ids = display_parallel_circuit_with_ID(parallel_layers, nqbt)
    compiled_n1, compiled_n2 = count_compiled_gates(parallel_layers)
    total_gates = compiled_n1 + compiled_n2 + total_ids

    print(f"Compiled circuit gate count (excluding IDs): 1qb={compiled_n1}, 2qb={compiled_n2}, total={compiled_n1+compiled_n2}")
    print(f"Total IDs inserted = {total_ids}")
    print(f"Total gates including IDs = {total_gates}")