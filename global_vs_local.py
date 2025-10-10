#run local depolarizing model for all depths
#use the data to ascertain the effective number of gates and noise param
#5 qubit heisenberg HVA

import netrc
info = netrc.netrc()

from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
from qat.core import Observable, Term, Batch #Hamiltonian
import numpy as np
from qat.plugins import ScipyMinimizePlugin
from qat.hardware import make_depolarizing_hardware_model
import pickle

nqbt = 5 # number of qubits
nruns = 0 #nbshots for observable sampling

base = 1
expo = 5

dep = np.arange(1, 11, 1, dtype = int)

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

#from qlmaas.plugins import SeqOptim
#optimizer_scipy = SeqOptim()
#from qlmaas.plugins import ScipyMinimizePlugin
#optimizer_scipy = ScipyMinimizePlugin(method="COBYLA", tol=1e-4, options={"maxiter": 10000})
from qat.vsolve.optimize import ScipyMinimizePlugin
optimizer_scipy = ScipyMinimizePlugin(method="COBYLA",
                                  tol=1e-6,
                                  options={"maxiter": 25000},
#                                  x0=np.zeros(nqbt)
                                     )

#infidelity
F = base*(10**(-expo))
custom_hardware = make_depolarizing_hardware_model(eps1=F, eps2=2*F, correl_type='single_qubit')

#from qlmaas.qpus import LinAlg
#from qat.qpus import get_default_qpu
from qlmaas.qpus import NoisyQProc
qpu_noisy = NoisyQProc(hardware_model = custom_hardware, fidelity=0.999)