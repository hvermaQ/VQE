from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np

"""
from qat.devices import AllToAll
from qat.plugins import NISQCompiler
from qat.hardware import DefaultGatesSpecification
from qat.plugins import Nnizer
nnizer = Nnizer()

nruns = 0 #nbshots for observable sampling

device = AllToAll(nqbt)

gate_times = {"H": 1., "CNOT": 10., "RX": 5., "RZ": 1.}
gates_spec = DefaultGatesSpecification(gate_times)

# define a limited LNN connectivity by using the qat.core library
from qat.core import HardwareSpecs, Topology, TopologyType
specs = HardwareSpecs(topology=Topology(type=TopologyType.ALL_TO_ALL))
"""

nqbt = 5 # number of qubits

#for ID insertion, only count the one and two qubit gates for each qubit
#in HVA, only NN connectivity needed, so topology does not matter
#minimum ID needed = difference in gate counts for each qubit

#gateset for counting gates
one_qb_gateset = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ', 'I']
two_qb_gateset = ['CNOT', 'CSIGN']  

dep = np.arange(1, 11, 1, dtype = int)

#gate counting routine
def gate_ct(in_circ):
    N_1qb = 0
    N_2qb = 0
    for tt1 in one_qb_gateset:
        N_1qb += in_circ.count(tt1)
    for tt2 in two_qb_gateset:
        N_2qb += in_circ.count(tt2)
    return(N_1qb, N_2qb)

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

    (N_1qb, N_2qb) = gate_ct(str(circuit))
    depth = circuit.depth({'CNOT' : 1, 'RZ' : 1, 'H' : 1, 'X' : 1})
    print("For ansatz depth = %s: "%ct)
    print("N_1qb = %s, N_2qb = %s, depth = %s" % (N_1qb, N_2qb, depth))
    print("Exact calculations:")
    