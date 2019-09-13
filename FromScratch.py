#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:17:39 2019

@author: henrikdreyer
"""

from qiskit.aqua.algorithms import VQE, ExactEigensolver
import matplotlib.pyplot as plt
import numpy as np
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit import IBMQ, BasicAer, Aer
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ
from qiskit.providers.aer import noise
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
backend = Aer.get_backend("qasm_simulator")

dist=0.5
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()
num_spin_orbitals=molecule.num_orbitals*2
num_particles=molecule.num_alpha+molecule.num_beta

map_type='jordan_wigner'
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)

print(qubitOp.print_details())

optimizer = SPSA(max_trials=100)
HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type)
var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis")

energy = vqe_instance.run(backend)['energy'] + shift
print(energy)