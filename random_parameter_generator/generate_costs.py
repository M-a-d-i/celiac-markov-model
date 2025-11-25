import numpy as np

from settings import settings
from utilities import utils


def generate_random_cost_matrix(number_of_simulations: int, cost_data) -> np.ndarray:
    number_of_states = cost_data[next(iter(cost_data))]['expected_values'].shape[1]
    randomized_cost_matrix = np.zeros((number_of_states, settings.number_of_years_to_simulate, number_of_simulations))

    for _, cost_details in cost_data.items():
        unit_cost = cost_details['unit_cost']
        expected_values = cost_details['expected_values']
        standard_deviations = cost_details['standard_deviations']
        for state_index, state in enumerate(expected_values.keys()):
            randomized_costs_for_state = np.empty((settings.number_of_years_to_simulate, number_of_simulations))
            for age in range(settings.number_of_years_to_simulate):
                expected_value = expected_values[state][age]
                standard_deviation = standard_deviations[state][age]
                if expected_value == 0:
                    randomized_costs_for_state[age] = np.zeros(number_of_simulations)
                else:
                    alpha, beta = utils.solve_gamma_distribution_parameters(expected_value, standard_deviation)
                    randomized_costs_for_state[age] = np.random.gamma(shape = alpha, scale = 1/beta, size = number_of_simulations)
            randomized_cost_matrix[state_index] += randomized_costs_for_state * unit_cost

    return randomized_cost_matrix