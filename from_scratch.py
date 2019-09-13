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

def opToCircs (circuit = QuantumCircuit, operator = WeightedPauliOperator, qr_size = int):
    if(qr_size < operator.num_qubits):
        #print("Error: Not enough qubits, enter at least QubitOp.num_qubits qubits.")
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
    #var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
    var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,
                     active_occupied=None, active_unoccupied=None, initial_state=HF_state,
                     qubit_mapping="jordan_wigner", two_qubit_reduction = False, num_time_slices = 1, shallow_circuit_concat = True, z2_symmetries = None)
    circuit = var_form.construct_circuit(parameters)
    energy = E(circuit, var_form, qubitOp, qr_size)
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
       job = execute(circuit, backend, shots=1000)
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
       #print(sep_counts)
       for i in range(len(sep_counts[0])-1):
           newdict = {}
           for j in range(2**pauli_size):
               b ='{0:b}'.format(j)
               b = b.zfill(pauli_size)
               newdict[b]=0
           #print(newdict)
           for k in range(len(sep_counts)):
               newdict[sep_counts[k][i]] += sep_counts[k][-1]
           energy += qubitOp.paulis[counter*paulis_per_register+i][0] * sum_binary(newdict, qubitOp.paulis[counter*paulis_per_register+i][1])
       counter += 1
   #print(energy)
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
        # print(" {} \t {} \t {}".format(key, (-1) ** parity, counts[key]))
        sum += (-1) ** parity * counts[key]
        total += counts[key]
    # print(total)
    return sum / total


shift = 0
dist=0.735
qr_size = 16
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM,
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()
num_spin_orbitals=molecule.num_orbitals*2
num_particles=molecule.num_alpha+molecule.num_beta

map_type='jordan_wigner'
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)

print(qubitOp.print_details())


#backend = Aer.get_backend("qasm_simulator")
#quantum_instance = QuantumInstance(backend=backend, shots=10)

optimizer = SPSA(max_trials=100)
HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type)


r=np.random.rand(3)
#r= np.array([ 0.1887294 ,  1.6412292 ,  1.28960801,  0.02820515, -0.14967703,
#        0.44681611,  1.46267043, -0.35240468,  1.76537533,  1.55680412,
#        1.70234826,  0.50467851,  0.08815427,  0.12147291,  2.07086526,
#        1.27208018])
#print(circuit.decompose())

optimizer = COBYLA(maxiter=100)

sol_opt = optimizer.optimize(3, energy_opt, gradient_function=None,
                variable_bounds=None, initial_point=r)
print(sol_opt)




#vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis") //operator_mode alte version
#vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer)

#for circ in output_circ:
#    print(circ.draw())

#energy = vqe_instance.run(backend)['energy'] + shift
#print(energy)





#print(sum_binary(counts_dicts[0]))

# Noisy Backend :-)

#provider = IBMQ.load_account()
#backend = Aer.get_backend("qasm_simulator")
#device = provider.get_backend("ibmqx4")
#coupling_map = device.configuration().coupling_map
#noise_model = noise.device.basic_device_noise_model(device.properties())
quantum_instance = QuantumInstance(backend=backend, shots=10)  # ,#)#,
# noise_model=noise_model,
# coupling_map=coupling_map)#,
# measurement_error_mitigation_cls=CompleteMeasFitter,
# cals_matrix_refresh_period=30,)

#result = quantum_instance.execute(meascircuits[0])
#print(result.get_counts(meascircuits[0]))

#sol_opt = optimizer.optimize(var_form.num_parameters, E, gradient_function=None,
                             #variable_bounds=None, initial_point=r)
#print(sol_opt)

# print(result.counts(meascircuits[0]))


# exact_solution = ExactEigensolver(qubitOp).run()
# print("Exact Result:", exact_solution['energy'])
# optimizer = SPSA(max_trials=100)
# var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
# vqe = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis")
# ret = vqe.run(quantum_instance)
# print("VQE Result:", ret['energy'])


##qr1=QuantumRegister(4)
##c=QuantumCircuit(4)
##qr1=QuantumRegister(4)
##cr1=ClassicalRegister(4)
# c2=qubitOp.construct_evaluation_circuit(wave_function=c,statevector_mode=False,
#                                     qr=None, cr=None, use_simulator_operator_mode=False,
#                                     circuit_name_prefix='')
# print(c2[0].draw())
##energy = vqe_instance.run(backend)['energy'] + shift
##print(energy)
#
## Execute the circuit on the qasm simulator
# job = execute(c2[0], backend, shots=1000)
#
## Grab results from the job
# result = job.result()
# print(result)
#
## Returns counts
# counts = result.get_counts(c2[0])
# print("\nTotal count for 00 and 11 are:",counts)
#
## Draw the circuit
##circuit.draw()