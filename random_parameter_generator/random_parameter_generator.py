from random_parameter_generator import generate_costs, generate_qalys
from random_parameter_generator.generate_transition_probability_matrices import TransitionProbabilityMatrices


class RandomParameters:
    def __init__(self, num_of_simulations: int, imported_data):
        self.societal_cost_data = imported_data.societal_cost_data
        self.qaly_expected_values = imported_data.qaly_expected_values
        self.qaly_standard_deviations = imported_data.qaly_standard_deviations

        self.cost_matrices = generate_costs.generate_random_cost_matrix(num_of_simulations, self.societal_cost_data)
        self.qaly_matrices = generate_qalys.generate_random_qaly_matrix(num_of_simulations, self.qaly_expected_values, self.qaly_standard_deviations)

        self.transition_probability_matrices = TransitionProbabilityMatrices(num_of_simulations, imported_data).generate_transition_probability_matrices_for_all_simulations()