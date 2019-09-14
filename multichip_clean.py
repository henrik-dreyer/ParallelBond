#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:17:39 2019

@author: henrikdreyer

"""

from qiskit.aqua.algorithms import VQE, ExactEigensolver
#from qiskit.aqua.algorithms.adaptive import VQEParallel
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumRegister,ClassicalRegister,QuantumCircuit
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP
from qiskit import IBMQ, BasicAer, Aer, execute
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit.providers.aer import noise
from qiskit.aqua import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
import pickle
import time


experiments=4
ds_min=0.5
ds_max=2.4
steps=15
opt_iter=20
energies_w_P=[]
energies_n_P=[]
stds_w_P=[]
stds_n_P=[]
ds_build=[]
exacts=[]
ds=range(steps)


for dist in ds:
    dist_step=dist
    dist=ds_min+dist*(ds_max-ds_min)/steps
    ds_build.append(dist)
    driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                         charge=0, spin=0, basis='sto3g')
    molecule = driver.run()
    repulsion_energy = molecule.nuclear_repulsion_energy
    num_spin_orbitals=molecule.num_orbitals*2
    num_particles=molecule.num_alpha+molecule.num_beta
    
    map_type='jordan_wigner'
    ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
    qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)

    
    exact_solution = ExactEigensolver(qubitOp).run()['energy']
    exact_solution=exact_solution+repulsion_energy

    #optimizer = SLSQP(maxiter=5)
    optimizer = COBYLA(maxiter=opt_iter)
    HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type)
    #var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
    var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,
                     active_occupied=None, active_unoccupied=None, initial_state=HF_state,
                     qubit_mapping='jordan_wigner', two_qubit_reduction=False, num_time_slices=1,
                     shallow_circuit_concat=True, z2_symmetries=None)
    
    
    #No parallelization
    energies_exp_n=[]
    for n in range(experiments):
        t=time.time()
        #Without parallelisation
        n_shots=50
        n_instances=1
        instances=[]
        for i in range(n_instances):
            instances.append(QuantumInstance(backend=BasicAer.get_backend("qasm_simulator"), shots=n_shots))
        
        if 'parameters_n_P' in locals():
            vqe_instance = VQE(qubitOp, var_form, initial_point=parameters_n_P, optimizer=optimizer,threads=len(instances))#, operator_mode="grouped_paulis")
        else:
            vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer,threads=len(instances))#, operator_mode="grouped_paulis")
    
        #result=vqe_instance.run()
        result_n_P=vqe_instance.run(instances)
        energy_n_P=result_n_P['energy']+repulsion_energy
        energies_exp_n.append(energy_n_P)
        delta_time=time.time()-t
        print('time is' + str(delta_time))
        
    mean=sum(energies_exp_n)/experiments
    std=sum(np.array(energies_exp_n)**2)/experiments-mean**2
    parameters_n_P=result_n_P['opt_params']
    energy_n_P=mean
    std_n_p=std
       

    #With parallelization
    energies_exp_w=[]
    for n in range(experiments):
        t=time.time()
        #WITH parallelisation
        n_shots=50
        n_instances=20
        instances=[]
        for i in range(n_instances):
            instances.append(QuantumInstance(backend=BasicAer.get_backend("qasm_simulator"), shots=n_shots))
        
        if 'parameters_w_P' in locals():
            vqe_instance = VQE(qubitOp, var_form, initial_point=parameters_w_P, optimizer=optimizer,threads=len(instances))#, operator_mode="grouped_paulis")
        else:
            vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer,threads=len(instances))#, operator_mode="grouped_paulis")
    
        #result=vqe_instance.run()
        result_w_P=vqe_instance.run(instances)
        energy_w_P=result_w_P['energy']+repulsion_energy 
        energies_exp_w.append(energy_w_P)
        delta_time=time.time()-t
        print('time is' + str(delta_time))
        
    mean=sum(energies_exp_w)/experiments
    std=sum(np.array(energies_exp_w)**2)/experiments-mean**2
    energy_w_P=mean
    std_w_p=std
    parameters_w_P=result_w_P['opt_params']
    
    
    filename='data_d'+str(dist_step)

    with open(filename, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([parameters_n_P ,energy_n_P, std_n_p, parameters_w_P ,energy_w_P, std_w_p, exact_solution, energies_exp_n, energies_exp_w, dist], f)
        

    energies_w_P.append(energy_w_P)
    energies_n_P.append(energy_n_P)
    stds_w_P.append(std_w_p)
    stds_n_P.append(std_n_p)
    exacts.append(exact_solution)
    
    plt.errorbar(ds_build, energies_w_P, yerr=stds_w_P, label='with parallelization')
    plt.errorbar(ds_build, energies_n_P, yerr=stds_n_P, label='without parallelization')

    plt.plot(ds_build,exacts)
    plt.show()
