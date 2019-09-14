#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:17:39 2019
@author: henrikdreyer
"""

from qiskit.aqua.algorithms import VQE, ExactEigensolver
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.aqua.components.variational_forms import RYRZ, VariationalForm
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP, AQGD
from qiskit import IBMQ, BasicAer, Aer, execute
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import noise
from qiskit.aqua import QuantumInstance
from qiskit.quantum_info import Pauli
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator
backend = Aer.get_backend("qasm_simulator")

def createPlot1(bondLengthMin=0.5,bondLengthMax=1.5,numberOfPoints=10,initialParameters=None,numberOfParameters=16,shotsPerPoint=1000,registerSize = 12,map_type='jordan_wigner'):
   if initialParameters is None:
       initialParameters = np.random.rand(numberOfParameters)
   global qubitOp
   global qr_size
   global shots
   shots=shotsPerPoint
   qr_size = registerSize
   optimizer = COBYLA(maxiter=20)
   bondLengths = []
   values = []
   delta = (bondLengthMax-bondLengthMin)/numberOfPoints
   for i in range(numberOfPoints):
       bondLengths.append(bondLengthMin+i*delta)
   for bondLength in bondLengths:
       driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(bondLength), unit=UnitsType.ANGSTROM,
                            charge=0, spin=0, basis='sto3g')
       molecule = driver.run()
       repulsion_energy = molecule.nuclear_repulsion_energy
       num_spin_orbitals = molecule.num_orbitals * 2
       num_particles = molecule.num_alpha + molecule.num_beta
       ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
       qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
       sol_opt = optimizer.optimize(numberOfParameters, energy_opt, gradient_function=None,
                                    variable_bounds=None, initial_point=initialParameters)
       values.append(sol_opt[1]+repulsion_energy)
   filename ='Energy - BondLengths'
   with open(filename, 'wb') as f:
       pickle.dump([bondLengths,values], f)
   plt.plot(bondLengths,values)
   plt.ylabel('Ground State Energy')
   plt.xlabel('Bond Length')
   plt.show()

def createPlot2(exactGroundStateEnergy=-1.14,numberOfIterations=1000,bondLength=0.735,initialParameters=None,numberOfParameters=16,shotsPerPoint=1000,registerSize = 12,map_type='jordan_wigner'):
    if initialParameters is None:
        initialParameters = np.random.rand(numberOfParameters)
    global qubitOp
    global qr_size
    global shots
    global values
    global plottingTime
    plottingTime= True
    shots = shotsPerPoint
    qr_size = registerSize
    optimizer = COBYLA(maxiter=numberOfIterations)
    iterations = []
    values = []
    for i in range(numberOfIterations):
        iterations.append(i+1)

    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(bondLength), unit=UnitsType.ANGSTROM,
                         charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_spin_orbitals = molecule.num_orbitals * 2
    num_particles = molecule.num_alpha + molecule.num_beta
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)
    sol_opt = optimizer.optimize(numberOfParameters, energy_opt, gradient_function=None,
                                 variable_bounds=None, initial_point=initialParameters)

    #Calculate Energy Error
    for i in range(len(values)):
        values[i] = values[i]+ repulsion_energy - exactGroundStateEnergy

    #Saving and Plotting Data
    filename = 'Energy Error - Iterations'
    with open(filename, 'wb') as f:
        pickle.dump([iterations, values], f)
    plt.plot(iterations, values)
    plt.ylabel('Ground State Energy')
    plt.xlabel('Iterations')
    plt.show()


def opToCircs (circuit = QuantumCircuit, operator = WeightedPauliOperator, qr_size = int):
    if(qr_size < operator.num_qubits):
        raise Exception('Error: Not enough qubits, enter at least QubitOp.num_qubits qubits.')
    qr = []
    cr = []
    for i in range(len(operator.paulis)):
        qr.append(QuantumRegister(operator.num_qubits))
        cr.append(ClassicalRegister(operator.num_qubits))
    meascircuits = operator.construct_evaluation_circuit(circuit, statevector_mode=False,
                                                        qr=None, cr=None, use_simulator_operator_mode=False,
                                                        circuit_name_prefix='')
    paulis_per_register = math.floor(qr_size/operator.num_qubits)
    numRegisters = math.ceil(len(operator.paulis)/paulis_per_register)
    output_circ = []
    for j in range(numRegisters):
        l = j*paulis_per_register
        for k in range(paulis_per_register-1):
            if(l+k+1<len(qr)):
                meascircuits[l].add_register(qr[l+k+1],cr[l+k+1])
                meascircuits[l].append(meascircuits[l+k+1].to_instruction(), qr[l+k+1], cr[l+k+1])
        output_circ.append(meascircuits[l].decompose())
    return output_circ

def energy_opt(parameters):
    var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
    #var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,
    #                 active_occupied=None, active_unoccupied=None, initial_state=HF_state,
    #                 qubit_mapping="jordan_wigner", two_qubit_reduction = False, num_time_slices = 1, shallow_circuit_concat = True, z2_symmetries = None)
    circuit = var_form.construct_circuit(parameters)
    energy = E(circuit, var_form, qubitOp, qr_size)
    if plottingTime:
        print('bla')
        values.append(energy)
    print(energy)
    return energy



#Optimization
def E(circuit = QuantumCircuit, var_form = VariationalForm, qubitOp = WeightedPauliOperator, qr_size = int):
   #get circuits of parameter
   energy=0
   pauli_size = qubitOp.num_qubits
   paulis_per_register = math.floor(qr_size/pauli_size)
   output_circuits = opToCircs(circuit, qubitOp,pauli_size*paulis_per_register)
   counter = 0
   for circuit in output_circuits:
       job = execute(circuit, backend, shots=shots, optimization_level=3)
       result = job.result()
       #result=quantum_instance.execute(circuit)
       counts = result.get_counts(circuit)
       sep_counts = []
       for key in counts:
           string = []
           for i in range(paulis_per_register):
               if (i*(pauli_size + 1) + pauli_size <= len(key)):
                   string.append(key[i * (pauli_size + 1):i * (pauli_size + 1) + pauli_size])
           string.append(counts.get(key))
           sep_counts.append(string)
       for i in range(len(sep_counts[0])-1):
           newdict = {}
           for j in range(2**pauli_size):
               b ='{0:b}'.format(j)
               b = b.zfill(pauli_size)
               newdict[b]=0
           for k in range(len(sep_counts)):
               newdict[sep_counts[k][i]] += sep_counts[k][-1]
           energy += qubitOp.paulis[counter*paulis_per_register+i][0] * sum_binary(newdict, qubitOp.paulis[counter*paulis_per_register+i][1])
       counter += 1
   return energy


def sum_binary(counts, pauli = Pauli):
    sum = 0
    total = 0
    countOperator =  list(np.logical_or(pauli.x, pauli.z))
    for key in counts:
        parity = 0
        counter = 0
        for i in key:
            if int(i) == 1 and countOperator[counter]:
                parity += 1
            counter += 1
        sum += (-1) ** parity * counts[key]
        total += counts[key]
    return sum / total


global plottingTime
plottingTime= False

# Noisy Backend :-)
provider = IBMQ.load_account()
#backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibmq_16_melbourne")
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(device.properties())
quantum_instance = QuantumInstance(backend=backend, shots=1000,
                                  noise_model=noise_model,
                                  coupling_map=coupling_map)


#HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type) \\we need this for UCCSD

createPlot1()
#createPlot2(numberOfIterations=20)

'''
r=np.random.rand(16)
r= np.array([ 0.1887294 ,  1.6412292 ,  1.28960801,  0.02820515, -0.14967703,
        0.44681611,  1.46267043, -0.35240468,  1.76537533,  1.55680412,
        1.70234826,  0.50467851,  0.08815427,  0.12147291,  2.07086526,
        1.27208018]) \\good starting parameters for RYRZ
optimizer = COBYLA(maxiter=100)
sol_opt = optimizer.optimize(16, energy_opt, gradient_function=None,
                variable_bounds=None, initial_point=r)
print(sol_opt)
'''

