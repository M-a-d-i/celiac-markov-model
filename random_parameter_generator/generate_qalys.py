import numpy as np

from utilities import utils


def generate_random_qaly_matrix(number_of_simulations: int, qaly_expected_values, qaly_standard_deviations) -> np.ndarray:
    number_of_states, number_of_ages = qaly_expected_values.shape

    qaly_matrix = np.empty((number_of_states, number_of_ages, number_of_simulations))

    for state in range(number_of_states):
        for age in range(number_of_ages):
            alpha, beta = utils.solve_beta_distribution_parameters(qaly_expected_values[state, age], qaly_standard_deviations[state, age])
            qaly_matrix[state, age] = np.random.beta(alpha, beta, size=number_of_simulations)

    return qaly_matrix