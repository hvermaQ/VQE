#circuit generation routine
#add circ generation through hamiltonian terms for generalization
from qat.lang.AQASM import Program, H, X, CNOT, RX, I, RY, RZ, CSIGN #Gates
import numpy as np

def gen_circ_RYA(args):
    nqbt = args[0]
    ct = args[1]
    #Variational circuit can only be constructed using the program framework
    qprog = Program()
    qbits = qprog.qalloc(nqbt)
    #variational parameters used for generating gates (permutation of [odd/even, xx/yy/zz])
    #variational parameters used for generating gates
    angs = [qprog.new_var(float, 'a_%s'%i) for i in range(nqbt*(2*ct+1))]
    ang = iter(angs)
    #circuit
    for it in range(ct):
        for q_index in range(nqbt):
            RY(next(ang))(qbits[q_index])
        for q_index in range(nqbt):
            if not q_index%2 and q_index <= nqbt-2:
                CSIGN(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt):
            RY(next(ang))(qbits[q_index])
        for q_index in range(nqbt):
            if q_index%2 and q_index <= nqbt-2:
                CSIGN(qbits[q_index],qbits[q_index+1])
        CSIGN(qbits[0],qbits[nqbt-1])
        if it==(ct-1):
            for q_index in range(nqbt):
                RY(next(ang))(qbits[q_index])
    #circuit
    circuit = qprog.to_circ()
    return(circuit)

def gen_circ_HVA(args):
    nqbt = args[0]
    ct = args[1]
    #Variational circuit can only be constructed using the program framework
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
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #odd Ryy
            if q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(bo[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(nqbt): #odd Rxx
            if q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(co[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
        for q_index in range(nqbt): #even Rzz
            if not q_index%2 and q_index <= nqbt-2:
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ae[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
        for q_index in range(nqbt): #even Ryy
            if not q_index%2 and q_index <= nqbt-2:
                RZ(np.pi/2)(qbits[q_index])
                RZ(np.pi/2)(qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(be[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
                RZ(-np.pi/2)(qbits[q_index])
                RZ(-np.pi/2)(qbits[q_index+1])
        for q_index in range(nqbt): #even Rxx
            if not q_index%2 and q_index <= nqbt-2:
                H(qbits[q_index])
                H(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                RZ(ce[it-1]/2)(qbits[q_index+1])
                CNOT(qbits[q_index],qbits[q_index+1])
                H(qbits[q_index])
                H(qbits[q_index+1])
    #circuit
    circuit = qprog.to_circ()
    return(circuit)