# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
The Variational Quantum Eigensolver algorithm.

See https://arxiv.org/abs/1304.3061

Changes for parallelization:
    Now takes a list of QuantumInstances rather than a single one (a list with
    a single QuantumInstance may be passed for the original functionality).
    Does not support parameter sets, statevector simulation and the calculation
    of the standard deviation is not correct. Also, currently it is assumed
    that all backends are the same.
    
    During each energy evaluation the same circuits us sent to as many quantum
    instances for evaluation. However, currently the qiskit.aqua.QuantumInstance
    class' execute function is different from the root Qiskit execute function
    in the following way: The root function execute returns a job handle and
    in principle the program can go on while the result is being computed 
    over the cloud. The execute method of QuantumInstance waits for the result
    to come in and returns a result object. For scheduling and for clarity,
    it would probably a good idea to make those functions more similar :-)
"""

import logging
import functools

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.aqua import QuantumInstance

from qiskit.aqua.algorithms.adaptive.vq_algorithm import VQAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.operators import (TPBGroupedWeightedPauliOperator, WeightedPauliOperator,
                                   MatrixOperator, op_converter)
from qiskit.aqua.utils.backend_utils import is_aer_statevector_backend, is_statevector_backend

logger = logging.getLogger(__name__)


class VQE(VQAlgorithm):
    """
    The Variational Quantum Eigensolver algorithm.

    See https://arxiv.org/abs/1304.3061
    """

    CONFIGURATION = {
        'name': 'VQE',
        'description': 'VQE Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'vqe_schema',
            'type': 'object',
            'properties': {
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'L_BFGS_B'
                },
            },
            {
                'pluggable_type': 'variational_form',
                'default': {
                    'name': 'RYRZ'
                },
            },
        ],
    }

    def __init__(self, operator, var_form, optimizer,
                 initial_point=None, max_evals_grouped=1, aux_operators=None, callback=None,
                 auto_conversion=True, threads=1):
        """Constructor.

        Args:
            operator (BaseOperator): Qubit operator
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
            initial_point (numpy.ndarray): optimizer initial point.
            max_evals_grouped (int): max number of evaluations performed simultaneously
            aux_operators (list[BaseOperator]): Auxiliary operators to be evaluated
                                                at each eigenvalue
            callback (Callable): a callback that can access the intermediate data
                                 during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion (bool): an automatic conversion for operator and aux_operators into
                                    the type which is
                                    most suitable for the backend.
                                    - non-aer statevector_simulator: MatrixOperator
                                    - aer statevector_simulator: WeightedPauliOperator
                                    - qasm simulator or real backend:
                                        TPBGroupedWeightedPauliOperator
        """
        self._operators=[]
        self.validate(locals())
        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point)
        self._use_simulator_operator_mode = None
        self._ret = None
        self._eval_time = None
        self._optimizer.set_max_evals_grouped(max_evals_grouped)
        self._callback = callback
        self._threads = threads
        if initial_point is None:
            self._initial_point = var_form.preferred_init_points
        self._operator = operator
        self._eval_count = 0
        self._aux_operators = []
        if aux_operators is not None:
            aux_operators = \
                [aux_operators] if not isinstance(aux_operators, list) else aux_operators
            for aux_op in aux_operators:
                self._aux_operators.append(aux_op)
        self._auto_conversion = auto_conversion
        logger.info(self.print_settings())

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance

        Returns:
            VQE: vqe object
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        vqe_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        initial_point = vqe_params.get('initial_point')
        max_evals_grouped = vqe_params.get('max_evals_grouped')

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, var_form, optimizer,
                   initial_point=initial_point, max_evals_grouped=max_evals_grouped,
                   aux_operators=algo_input.aux_ops)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(
            self.configuration['name'])
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._var_form.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def _config_the_best_mode(self, operator, backend):

        if not isinstance(operator, (WeightedPauliOperator, MatrixOperator,
                                     TPBGroupedWeightedPauliOperator)):
            logger.debug("Unrecognized operator type, skip auto conversion.")
            return operator

        ret_op = operator
        if not is_statevector_backend(backend):  # assume qasm, should use grouped paulis.
            if isinstance(operator, (WeightedPauliOperator, MatrixOperator)):
                logger.debug("When running with Qasm simulator, grouped pauli can "
                             "save number of measurements. "
                             "We convert the operator into grouped ones.")
                ret_op = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    operator, TPBGroupedWeightedPauliOperator.sorted_grouping)
        else:
            if not is_aer_statevector_backend(backend):
                if not isinstance(operator, MatrixOperator):
                    logger.info("When running with non-Aer statevector simulator, "
                                "represent operator as a matrix could "
                                "achieve the better performance. We convert "
                                "the operator to matrix.")
                    ret_op = op_converter.to_matrix_operator(operator)
            else:
                if not isinstance(operator, WeightedPauliOperator):
                    logger.info("When running with Aer statevector simulator, "
                                "represent operator as weighted paulis could "
                                "achieve the better performance. We convert "
                                "the operator to weighted paulis.")
                    ret_op = op_converter.to_weighted_pauli_operator(operator)
        return ret_op

    def construct_circuit(self, parameter, statevector_mode=False,
                          use_simulator_operator_mode=False, circuit_name_prefix=''):
        """Generate the circuits.

        Args:
            parameter (numpy.ndarray): parameters for variational form.
            statevector_mode (bool, optional): indicate which type of simulator are going to use.
            use_simulator_operator_mode (bool, optional): is backend from AerProvider,
                            if True and mode is paulis, single circuit is generated.
            circuit_name_prefix (str, optional): a prefix of circuit name

        Returns:
            list[QuantumCircuit]: the generated circuits with Hamiltonian.
        """
        wave_function = self._var_form.construct_circuit(parameter)
        circuits = self._operator.construct_evaluation_circuit(
            wave_function, statevector_mode,
            use_simulator_operator_mode=use_simulator_operator_mode,
            circuit_name_prefix=circuit_name_prefix)
        return circuits

    def _eval_aux_ops(self, threshold=1e-12, params=None):
        if params is None:
            params = self.optimal_params
        wavefn_circuit = self._var_form.construct_circuit(params)
        circuits = []
        values = []
        params = []
        for idx, operator in enumerate(self._aux_operators):
            if not operator.is_empty():
                temp_circuit = QuantumCircuit() + wavefn_circuit
                circuit = operator.construct_evaluation_circuit(
                    wave_function=temp_circuit,
                    statevector_mode=self._quantum_instance.is_statevector,
                    use_simulator_operator_mode=self._use_simulator_operator_mode,
                    circuit_name_prefix=str(idx))
                if self._use_simulator_operator_mode:
                    params.append(operator.aer_paulis)
            else:
                circuit = None
            circuits.append(circuit)

        if circuits:
            to_be_simulated_circuits = \
                functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
            if self._use_simulator_operator_mode:
                extra_args = {
                    'expectation':
                        {
                            'params': params,
                            'num_qubits': self._operator.num_qubits
                        }
                }
            else:
                extra_args = {}
            result = self._quantum_instance.execute(to_be_simulated_circuits, **extra_args)

            for idx, operator in enumerate(self._aux_operators):
                if operator.is_empty():
                    mean, std = 0.0, 0.0
                else:
                    mean, std = operator.evaluate_with_result(
                        result=result, statevector_mode=self._quantum_instance.is_statevector,
                        use_simulator_operator_mode=self._use_simulator_operator_mode,
                        circuit_name_prefix=str(idx))

                mean = mean.real if abs(mean.real) > threshold else 0.0
                std = std.real if abs(std.real) > threshold else 0.0
                values.append((mean, std))

        if values:
            aux_op_vals = np.empty([1, len(self._aux_operators), 2])
            aux_op_vals[0, :] = np.asarray(values)
            self._ret['aux_ops'] = aux_op_vals

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            dict: Dictionary of results

        Raises:
            AquaError: wrong setting of operator and backend.
        """
        
        #Threading currently only works if aux_operators is empty
        if isinstance(self._quantum_instance,QuantumInstance):
            if self._auto_conversion:
                self._operator = \
                    self._config_the_best_mode(self._operator, self._quantum_instance.backend)
                for i in range(len(self._aux_operators)):
                    if not self._aux_operators[i].is_empty():
                        self._aux_operators[i] = \
                            self._config_the_best_mode(self._aux_operators[i],
                                                       self._quantum_instance.backend)
        else:
            if self._auto_conversion:
                for j in range(len(self._quantum_instance)):
                    self._operators.append(self._config_the_best_mode(self._operator, self._quantum_instance[j].backend))
                    for i in range(len(self._aux_operators)):
                        if not self._aux_operators[i].is_empty():
                            self._aux_operators[i] = \
                                self._config_the_best_mode(self._aux_operators[i],
                                                           self._quantum_instance.backend)

        # sanity check
        if isinstance(self._operator, MatrixOperator) and not self._quantum_instance.is_statevector:
            raise AquaError("Non-statevector simulator can not work "
                            "with `MatrixOperator`, either turn ON "
                            "auto_conversion or use the proper "
                            "combination between operator and backend.")

        if isinstance(self._quantum_instance,QuantumInstance):
            self._use_simulator_operator_mode = \
                is_aer_statevector_backend(self._quantum_instance.backend) \
                and isinstance(self._operator, (WeightedPauliOperator, TPBGroupedWeightedPauliOperator))
                
            self._quantum_instance.circuit_summary = True
        else:
            self._use_simulator_operator_mode=False
            for j in range(len(self._quantum_instance)):
                self._quantum_instance[j].circuit_summary = True
        

        self._eval_count = 0
        self._ret = self.find_minimum(initial_point=self.initial_point,
                                      var_form=self.var_form,
                                      cost_fn=self._energy_evaluation,
                                      optimizer=self.optimizer)
        if self._ret['num_optimizer_evals'] is not None and \
                self._eval_count >= self._ret['num_optimizer_evals']:
            self._eval_count = self._ret['num_optimizer_evals']
        self._eval_time = self._ret['eval_time']
        logger.info('Optimization complete in %s seconds.\nFound opt_params %s in %s evals',
                    self._eval_time, self._ret['opt_params'], self._eval_count)
        self._ret['eval_count'] = self._eval_count

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self.get_optimal_cost()])
        self._ret['eigvecs'] = np.asarray([self.get_optimal_vector()])
        self._eval_aux_ops()
        return self._ret

    # This is the objective function to be passed to the optimizer that is uses for evaluation
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            Union(float, list[float]): energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        circuits = []
        parameter_sets = np.split(parameters, num_parameter_sets)
        mean_energy = []
        std_energy = []

        for idx, _ in enumerate(parameter_sets):
            parameter = parameter_sets[idx]
            circuit = self.construct_circuit(
                parameter,
                statevector_mode=self._quantum_instance[0].is_statevector,
                use_simulator_operator_mode=self._use_simulator_operator_mode,
                circuit_name_prefix=str(idx))
            circuits.append(circuit)

        to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)
        if self._use_simulator_operator_mode:
            extra_args = {
                'expectation':
                    {
                        'params': [self._operator.aer_paulis],
                        'num_qubits': self._operator.num_qubits
                    }
            }
        else:
            extra_args = {}
            
        results=[]
        jobs=[]
        
        #Here we distribute the energy evaluation over a list of quantum 
        #instances. For a comment see top.
        
        for i in range(self._threads):
            jobs.append(self._quantum_instance[i].execute(to_be_simulated_circuits, optimization_level=3, **extra_args))
        
        for i in range(self._threads):
            results.append(jobs[i])
  
        
        means=[]
        for i in range(self._threads):
            mean_energy = []
            std_energy = []
            for idx, _ in enumerate(parameter_sets):
                mean, std = self._operator.evaluate_with_result(
                    result=results[i], statevector_mode=self.quantum_instance[i].is_statevector,
                    use_simulator_operator_mode=self._use_simulator_operator_mode,
                    circuit_name_prefix=str(idx))
                mean_energy.append(np.real(mean))
                std_energy.append(np.real(std))
                self._eval_count += 1
                if self._callback is not None:
                    self._callback(self._eval_count, parameter_sets[idx], np.real(mean), np.real(std))
                logger.info('Energy evaluation %s returned %s', self._eval_count, np.real(mean))
                
                #HERE
            means.append(mean_energy)
        mean_energy=np.sum(means)/self._threads

        return mean_energy #if len(mean_energy) > 1 else mean_energy[0]

    def get_optimal_cost(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot return optimal cost before running the "
                            "algorithm to find optimal params.")
        return self._ret['min_val']

    def get_optimal_circuit(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal circuit before running the "
                            "algorithm to find optimal params.")
        return self._var_form.construct_circuit(self._ret['opt_params'])

    def get_optimal_vector(self):
        from qiskit.aqua.utils.run_circuits import find_regs_by_name

        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal vector before running the "
                            "algorithm to find optimal params.")
        qc = self.get_optimal_circuit()
        if self._quantum_instance[0].is_statevector:
            ret = self._quantum_instance.execute(qc)
            self._ret['min_vector'] = ret.get_statevector(qc)
        else:
            c = ClassicalRegister(qc.width(), name='c')
            q = find_regs_by_name(qc, 'q')
            qc.add_register(c)
            qc.barrier(q)
            qc.measure(q, c)
            tmp_cache = self._quantum_instance[0].circuit_cache
            self._quantum_instance[0]._circuit_cache = None
            ret = self._quantum_instance[0].execute(qc)
            self._quantum_instance[0]._circuit_cache = tmp_cache
            self._ret['min_vector'] = ret.get_counts(qc)
        return self._ret['min_vector']

    @property
    def optimal_params(self):
        if 'opt_params' not in self._ret:
            raise AquaError("Cannot find optimal params before running the algorithm.")
        return self._ret['opt_params']
