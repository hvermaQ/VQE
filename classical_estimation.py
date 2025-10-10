#estimation of flops and classical energy consumption
#isolate the flops for quantum and classical
#use iterations of the VQE to calculcate the per iterations flops
#scale with the noise of the QPU

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np
#from qat.plugins import ScipyMinimizePlugin
from opto_gauss import Opto, GaussianNoise
from pypapi import events, papi_high as high

    
nqbt = 5 # number of qubits
nruns = 0 #nbshots for observable sampling

#dep = np.arange(1, 11, 1, dtype = int)

ct = 2 #sample depth
#Instantiation of Hamiltoniian
heisen = Observable(nqbt)
#Generation of Heisenberg Hamiltonian
for q_reg in range(nqbt-1):
    heisen += Observable(nqbt, pauli_terms = [Term(1., typ, [q_reg,q_reg + 1]) for typ in ['XX','YY','ZZ']])

    #exact calculation for ground state
from qat.fermion import SpinHamiltonian

heisen_class = SpinHamiltonian(nqbits=heisen.nbqbits, terms=heisen.terms)
heisen_mat = heisen_class.get_matrix()

eigvals, eigvecs = np.linalg.eigh(heisen_mat)
g_energy = eigvals[0]
g_state = eigvecs[:,0]

#from qlmaas.qpus import LinAlg
from qat.qpus import get_default_qpu
#from qlmaas.qpus import MPO
#from qlmaas.qpus import NoisyQProc
#ideal processor
#qpu_ideal = LinAlg()
qpu_ideal = get_default_qpu()

base = 1
expo = 5

#infidelity
F = base*(10**(-expo))

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
job = circuit.to_job(observable = heisen, nbshots = 0)
#Ng1, Ng2 = gate_ct(circuit)
#Ng = Ng1 + Ng2
#profile noisy
high.start_counters([events.PAPI_FP_OPS,])
stack_noisy = Opto | GaussianNoise(F, heisen_mat) | qpu_ideal
res_noisy = stack_noisy.submit(job)
x1 = high.stop_counters()
#profile ideal
stack_ideal = Opto | qpu_ideal
res_ideal = stack_ideal.submit(job)
#profile only qpu job submission with random bindings of variational parameters
x0 = 2*np.pi*np.random.rand(len(job.get_variables()))
param_dict = {v: xx for v, xx in zip(job.get_variables(), x0)}
# Bind parameters
job_bound = job(**param_dict)
res_ref = qpu_ideal.submit(job_bound)

