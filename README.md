# ParallelBond

We implemented 

a) a many-chip version of the VQE that sends out prepare-and-measure circuits during each step of the optimization to different quantum instances with different quantum backends. MultiChip_clean computes data for different parallelization settings 

b) a parallelized version of VQEs that is run on one quantum device that provides a larger number of qubits than required to evaluate the individual Pauli operators. We studied how the results were affected by the number of qubits and determined the optimal ratio between the number of classical optimization steps and the number of iterations of the quantum circuit. Moreover, we studied noise models and analyzed how they affected the optimal ground state value of molecular hydrogen. With a little modification one could generalize this parallelization and apply it to other circuits that factorize into uninteracting blocks of circuits.
