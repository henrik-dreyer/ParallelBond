#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 17:06:56 2019

@author: henrikdreyer
"""

import numpy as np

import qiskit
from qiskit import IBMQ
from qiskit import Aer

from qiskit import QuantumRegister

from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.algorithms import VQE, ExactEigensolver
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.operators import Z2Symmetries
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.circuit.quantumcircuit import QuantumCircuit
#from qiskit.aqua.algorithms.VQE import _energy_evaluation


from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock

import matplotlib.pyplot as plt

driver = PySCFDriver(atom='Li 0.0 0.0 0.0; H 0.0 0.0 1.6', unit=UnitsType.ANGSTROM, charge=0, spin=0, basis='sto3g')
molecule = driver.run()
                     
freeze_list=[0]
remove_list=[-3, -2]
map_type='parity'

h1=molecule.one_body_integrals
h2=molecule.two_body_integrals
nuclear_repulsion_energy=molecule.nuclear_repulsion_energy

num_particles=molecule.num_alpha+molecule.num_beta
num_spin_orbitals=molecule.num_orbitals*2
print("HF Energy: {}".format(molecule.hf_energy-molecule.nuclear_repulsion_energy))
print("# of electrons: {}".format(num_particles))
print("# of spin orbitals: {}".format(num_spin_orbitals))

remove_list=[x % molecule.num_orbitals for x in remove_list]
freeze_list=[x % molecule.num_orbitals for x in freeze_list]

remove_list=[x-len(freeze_list) for x in remove_list]
remove_list+=[x+molecule.num_orbitals - len(freeze_list) for x in remove_list]
freeze_list+=[x+molecule.num_orbitals for x in freeze_list]

energy_shift=0.0
qubit_reduction = True if map_type == 'parity' else False

ferOp=FermionicOperator(h1=h1, h2=h2)
if len(freeze_list)>0:
    ferOp, energy_shift=ferOp.fermion_mode_freezing(freeze_list)
    num_spin_orbitals -= len(freeze_list)
    num_particles -= len(freeze_list)
if len(remove_list) >0:
    ferOp = ferOp.fermion_mode_elimination(remove_list)
    num_spin_orbitals -= len(remove_list)

qubitOp=ferOp.mapping(map_type='parity', threshold=0.0000001)
qubitOp=Z2Symmetries.two_qubit_reduction(qubitOp, num_particles) if qubit_reduction else qubitOp
qubitOp.chop(10**-10)
print(qubitOp.print_details())

backend=Aer.get_backend('statevector_simulator')
max_eval=200
cobyla=COBYLA(maxiter=max_eval)
#SPSA=SPSA()#maxiter=max_eval)

HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type, qubit_reduction)
var_form = UCCSD(qubitOp.num_qubits,depth=1, 
                 num_orbitals=num_spin_orbitals, num_particles=num_particles, 
                 active_occupied=[0], active_unoccupied=[0,1], initial_state=HF_state,
                 qubit_mapping=map_type, two_qubit_reduction=qubit_reduction, num_time_slices=1)

q1=QuantumRegister(4)
q2=QuantumRegister(4)
qc1=var_form.construct_circuit([1,2,3,4,5,6,7,8],q1)
qc2=var_form.construct_circuit([1,2,3,4,5,6,7,8],q2)
qc=qc1+qc2
print(qc.draw())
print(qc.count_ops())
print(qc.depth())

from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.providers.aer import noise
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

provider = IBMQ.load_account()
backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibmqx4")
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(device.properties())
quantum_instance = QuantumInstance(backend=backend, shots=1000, 
                                   noise_model=noise_model, 
                                   coupling_map=coupling_map,
                                   measurement_error_mitigation_cls=CompleteMeasFitter,
                                   cals_matrix_refresh_period=30,)

exact_solution = ExactEigensolver(qubitOp).run()
print("Exact Result:", exact_solution['energy'])
optimizer = SPSA(max_trials=100)
var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
vqe = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis")

parameters=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
#E=VQE._energy_evaluation(vqe,parameters)

ret = vqe.run(quantum_instance)
print("VQE Result:", ret['energy'])


ret_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    operator, TPBGroupedWeightedPauliOperator.sorted_grouping)

ret_op.construct_evaluation_circuit(operator_mode=None, input_circuit=None, backend=None, qr=None, cr=None, use_simulator_operator_mode=False, wave_function=None, statevector_mode=None, circuit_name_prefix='')

#vqe = VQE(qubitOp, var_form, SPSA, 'paulis')
#results = vqe.run(backend)['energy'] + shift
#vqe_energies.append(results)

#backend = Aer.get_backend("qasm_simulator")
#device = provider.get_backend("ibmqx4")
#coupling_map = device.configuration().coupling_map
#noise_model = noise.device.basic_device_noise_model(device.properties())
#quantum_instance = QuantumInstance(backend=backend, shots=1000, 
#                                   noise_model=noise_model, 
#                                   coupling_map=coupling_map,
#                                   measurement_error_mitigation_cls=CompleteMeasFitter,
#                                   cals_matrix_refresh_period=30,)

#vqe=VQE(qubitOp,var_form,cobyla)
#quantum_instance=QuantumInstance(backend=backend)
#results=vqe.run(quantum_instance)
#print('The computed ground state energy is: {}'.format(results['eigvals'][0]))
#print('The total ground state energy is {}'.format(results['eigvals'][0]+energy_shift+nuclear_repulsion_energy))
#print("Parameters:{}".format(results['opt_params']))