# ParallelBond

We implemented two things:

a) A many-chip version of the VQE that sends out prepare-and-measure circuits during each step of the optimization to different quantum instances with different quantum backends. To this end, we modified Qiskit Aqua's built-in VQE function. Instead of passing a single quantum instance, now a list of quantum instances can be passed to the VQE object. Under the hood, during the energy evaluation, the same circuit is sent to the list of backends associated with each quantum instance. The script test_multichip.py is a small test program that computes the energy of molecular hydrogen close to the optimal bond length and compares it with the exact energy. 
The script multichip_clean.py computes data for different parallelization settings and has been used to produce the plot plot_multichip.png.
Note that, in order to run either of those, you have to replace Qiskit Aqua's VQE class with the vqe.py file contained in the repo.

Actually, during our work we found a slight inconvenience in Qiskit Aqua's QuantumInstance class. This class contains an execute function. Qiskit itself also contains an execute function. However, their functionality is slightly different! While the 'root' execute function returns a job handle and is compatible with scheduling, the QuantumInstance.execute method immediately runs the job and one has to wait to retrieve the results. It then returns a results object instead of a job object. Therefore, we think that it would be a good idea to rewrite the QuantumInstance.execute() method to allow for scheduling :-)

b) A parallelized version of VQEs that is run on one quantum device that provides a larger number of qubits than required to evaluate the individual Pauli operators. We studied how the results were affected by the number of qubits and determined the optimal ratio between the number of classical optimization steps and the number of iterations of the quantum circuit. Moreover, we studied noise models and analyzed how they affected the optimal ground state value of molecular hydrogen. With a little modification one could generalize this parallelization and apply it to other circuits that factorize into uninteracting blocks of circuits.
E_time(1).png contains a benchmark of the time-to-solution for a fixed number of shots. While developing we noticed that we have a systematic bias in our parallelisation routing: while we get close to the correct energy and converge just as fast as without parallelization, we systematically overestimate the energy.
E(r) shows that noise kills the process with or without parallelization.
There is also a test file.
