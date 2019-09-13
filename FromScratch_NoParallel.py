#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:17:39 2019

@author: henrikdreyer
"""


from qiskit.aqua.algorithms import VQE, ExactEigensolver
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
backend = Aer.get_backend("qasm_simulator")

shift=0.0
dist=0.735
driver = PySCFDriver(atom="H .0 .0 .0; H .0 .0 " + str(dist), unit=UnitsType.ANGSTROM, 
                     charge=0, spin=0, basis='sto3g')
molecule = driver.run()
num_spin_orbitals=molecule.num_orbitals*2
num_particles=molecule.num_alpha+molecule.num_beta

map_type='jordan_wigner'
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
qubitOp = ferOp.mapping(map_type=map_type, threshold=0.00000001)

print(qubitOp.print_details())

optimizer = SPSA(max_trials=10)
HF_state=HartreeFock(qubitOp.num_qubits, num_spin_orbitals, num_particles, map_type)
#var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
var_form = UCCSD(qubitOp.num_qubits, depth=1, num_orbitals=num_spin_orbitals, num_particles=num_particles,
                 active_occupied=None, active_unoccupied=None, initial_state=HF_state,
                 qubit_mapping='jordan_wigner', two_qubit_reduction=False, num_time_slices=1,
                 shallow_circuit_concat=True, z2_symmetries=None)

#vqe_instance = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis")

r=np.random.rand(var_form.num_parameters)
circuit=var_form.construct_circuit(r)


meascircuits = qubitOp.construct_evaluation_circuit(wave_function=circuit, statevector_mode=False,
                                    qr=None, cr=None, use_simulator_operator_mode=False,
                                    circuit_name_prefix='')
print(meascircuits)
print(len(meascircuits))
counts_dicts=[]
for circuit in meascircuits:
   job = execute(circuit, backend, shots=10)
   result = job.result()
   counts = result.get_counts(circuit)
   counts_dicts.append(counts)
   print("\nTotal count for 00 and 11 are:",counts)
   print(circuit.draw())
   
#Optimization
def E(parameters):
    #get circuits of parameter  
    energy=0
    circuit=var_form.construct_circuit(parameters)
    meascircuits = qubitOp.construct_evaluation_circuit(wave_function=circuit, statevector_mode=False,
                                    qr=None, cr=None, use_simulator_operator_mode=False,
                                    circuit_name_prefix='')
    
    for circuit in meascircuits:
        job = execute(circuit, backend, shots=10)
        result=quantum_instance.execute(circuit)
        counts=result.get_counts(circuit)
        energy+=sum_binary(counts)
    
    return energy
    
   
    

def sum_binary(counts):
  sum = 0
  total = 0
  for key in counts:
      parity = 0
      for i in key:
          if int(i) == 1:
              parity += 1
      #print(" {} \t {} \t {}".format(key, (-1) ** parity, counts[key]))
      sum += (-1)** parity * counts[key]
      print(sum)
      total += counts[key]
  #print(total)
  return sum / total
print(sum_binary(counts_dicts[0]))




#Noisy Backend :-)

provider = IBMQ.load_account()
backend = Aer.get_backend("qasm_simulator")
device = provider.get_backend("ibmqx4")
coupling_map = device.configuration().coupling_map
noise_model = noise.device.basic_device_noise_model(device.properties())
quantum_instance = QuantumInstance(backend=backend, shots=10)#,#)#, 
                                   #noise_model=noise_model, 
                                   #coupling_map=coupling_map)#,
                                   #measurement_error_mitigation_cls=CompleteMeasFitter,
                                   #cals_matrix_refresh_period=30,)

result=quantum_instance.execute(meascircuits[0])
print(result.get_counts(meascircuits[0]))

sol_opt = optimizer.optimize(var_form.num_parameters, E, gradient_function=None,
                 variable_bounds=None, initial_point=r)
print(sol_opt)



#print(result.counts(meascircuits[0]))



#exact_solution = ExactEigensolver(qubitOp).run()
#print("Exact Result:", exact_solution['energy'])
#optimizer = SPSA(max_trials=100)
#var_form = RYRZ(qubitOp.num_qubits, depth=1, entanglement="linear")
#vqe = VQE(qubitOp, var_form, optimizer=optimizer, operator_mode="grouped_paulis")
#ret = vqe.run(quantum_instance)
#print("VQE Result:", ret['energy'])




##qr1=QuantumRegister(4)
##c=QuantumCircuit(4)
##qr1=QuantumRegister(4)
##cr1=ClassicalRegister(4)
#c2=qubitOp.construct_evaluation_circuit(wave_function=c,statevector_mode=False,
#                                     qr=None, cr=None, use_simulator_operator_mode=False,
#                                     circuit_name_prefix='')
#print(c2[0].draw())
##energy = vqe_instance.run(backend)['energy'] + shift
##print(energy)
#
## Execute the circuit on the qasm simulator
#job = execute(c2[0], backend, shots=1000)
#
## Grab results from the job
#result = job.result()
#print(result)
#
## Returns counts
#counts = result.get_counts(c2[0])
#print("\nTotal count for 00 and 11 are:",counts)
#
## Draw the circuit
##circuit.draw()
