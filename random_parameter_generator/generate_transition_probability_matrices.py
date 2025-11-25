import numpy as np

from random_parameter_generator.generate_transition_probabilities import TransitionProbabilities
from settings import settings


class TransitionProbabilityMatrices:
    def __init__(self, number_of_simulations: int, imported_data):
        self.number_of_simulations = number_of_simulations
        self.mortalities = imported_data.mortalities
        self.randomized_transition_probabilities = TransitionProbabilities(number_of_simulations, imported_data).transition_probabilities

    def generate_transition_probability_matrices_for_all_simulations(self):
        transition_probability_matrices = np.array([
            self._generate_transition_probability_matrices_for_a_single_simulation(iteration_number)
            for iteration_number in range(self.number_of_simulations)
        ])
        return transition_probability_matrices

    def _generate_transition_probability_matrices_for_a_single_simulation(self, iteration_number):
        transition_probability_matrices = np.zeros(
            shape=(settings.number_of_years_to_simulate, len(settings.states), len(settings.states))
        )
        for age in range(settings.number_of_years_to_simulate):
            transition_probability_matrices[age] = self._get_transition_probabilities(age, iteration_number)
        return transition_probability_matrices

    def _get_transition_probabilities(self, age, iteration_number):
        aa, ab, ac, ad = self._get_probabilities("from healthy", iteration_number, age)
        bb, bc, bd = self._get_probabilities("from asymptomatic", iteration_number, age)
        cc, cd, ce, cf = self._get_probabilities("from symptomatic", iteration_number, age)
        dd, de, df = self._get_probabilities("from clinical evaluation", iteration_number, age)
        ee = self._get_probabilities("gluten free compliance rates", iteration_number, age)
        mr = self.mortalities[age]
        mr_hr = self._get_probabilities("mortality hazard ratios", iteration_number, age)
        p = [
            [aa - mr,           ab,                 ac,                 ad,                 0,                     0,                           mr],
            [0,                 bb - mr_hr * mr,    bc,                 bd,                 0,                     0,                           mr_hr * mr],
            [0,                 0,                  cc - mr_hr * mr,    cd,                 ce,                    cf,                          mr_hr * mr],
            [0,                 0,                  0,                  dd - mr_hr * mr,    de,                    df,                          mr_hr * mr],
            [0,                 0,                  0,                  0,                  ee * (1 - mr),         (1 - ee) * (1 - mr),         mr],
            [0,                 0,                  0,                  0,                  ee * (1 - mr_hr * mr), (1 - ee) * (1 - mr_hr * mr), mr_hr * mr],
            [0,                 0,                  0,                  0,                  0,                     0,                           1]
        ]

        return p

    def _get_probabilities(self, key, iteration_number, age):
        probabilities = self.randomized_transition_probabilities.get(key)
        if probabilities is None:
            raise ValueError(f"{key} not found")
        return probabilities[iteration_number][age]
