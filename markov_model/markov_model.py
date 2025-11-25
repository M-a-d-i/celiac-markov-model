import numpy as np

from markov_model.screening import Screening
from settings import settings
from utilities import utils


class MarkovModel:
    def __init__(self, imported_data):
        self.imported_data = imported_data
        self.screening_parameters = self.imported_data.screening_parameters
        self.screening = Screening(self.screening_parameters)

    def simulate_markov_chain(self, screening_strategy, transition_probability_matrix):
        screening_ages, test_for_hla = screening_strategy
        # Initial distribution at age = 0, everyone is born a healthy baby
        initial_distribution = np.array([1,0,0,0,0,0,0])
        population_distribution = initial_distribution
        total_screening_costs, screening_round_number, biopsy_disutility = 0, 0, 0
        population_distribution_over_time = np.zeros((settings.number_of_years_to_simulate, len(settings.states)))
        for age in range(settings.number_of_years_to_simulate):
            if age in screening_ages:
                screening_round_number += 1
                population_distribution, screening_costs, biopsy_disutility = self._screen_population(population_distribution, test_for_hla, screening_round_number)
                discount_coefficient = (1+settings.screening_cost_discount_rate)**(-age)
                total_screening_costs += screening_costs * discount_coefficient
            population_distribution_over_time[age] = population_distribution
            transition_probabilities = transition_probability_matrix[age]
            population_distribution = population_distribution @ transition_probabilities
        return population_distribution_over_time, total_screening_costs, biopsy_disutility

    def get_population_level_statistics_of_a_single_simulation(self, screening_strategy, transition_probability_matrix):
        screening_ages, test_for_hla = screening_strategy
        # Initial distribution at age = 0, everyone is born a healthy baby
        initial_distribution = np.array([1,0,0,0,0,0,0])
        population_distribution = initial_distribution
        total_screening_costs, screening_round_number = 0, 0

        # Initialize data containers for calculating statistics
        annual_healthy_deaths = np.zeros(settings.number_of_years_to_simulate)
        annual_celiac_disease_deaths = np.zeros(settings.number_of_years_to_simulate)
        annual_number_of_new_diagnoses = np.zeros(settings.number_of_years_to_simulate)
        percentage_of_celiac_disease_that_is_diagnosed = np.zeros(settings.number_of_years_to_simulate)
        clinical_prevalence_per_1000 = np.zeros(settings.number_of_years_to_simulate)

        for age in range(settings.number_of_years_to_simulate):
            if age in screening_ages:
                screening_round_number += 1
                population_distribution, screening_costs, biopsy_disutility = self._screen_population(population_distribution, test_for_hla, screening_round_number)
            transition_probabilities = transition_probability_matrix[age]
            # Calculate how many people transition from a state with celiac disease to death and save result to annual_celiac_disease_deaths
            annual_healthy_deaths[age] = self._get_transitions_between_given_states(population_distribution, transition_probabilities, ['HEALTHY'], ['DEAD'])
            annual_celiac_disease_deaths[age] = self._get_transitions_between_given_states(population_distribution, transition_probabilities, ['ASYMPTOMATIC', 'SYMPTOMATIC', 'CLINICALEVALUATION', 'COMPLIANT', 'NONCOMPLIANT'], ['DEAD'])
            annual_number_of_new_diagnoses[age] = self._get_transitions_between_given_states(population_distribution, transition_probabilities, ['ASYMPTOMATIC', 'SYMPTOMATIC', 'CLINICALEVALUATION'], ['COMPLIANT', 'NONCOMPLIANT'])
            healthy_patients = np.sum(population_distribution[0:1])
            all_celiac_disease_patients = np.sum(population_distribution[1:6])
            diagnosed_celiac_disease_patients = np.sum(population_distribution[4:6])
            # Correct clinical prevalence for the fact that some people are dead
            clinical_prevalence_per_1000[age] = diagnosed_celiac_disease_patients * 1000 / (healthy_patients + all_celiac_disease_patients)
            if all_celiac_disease_patients > 0:
                percentage_of_celiac_disease_that_is_diagnosed[age] = diagnosed_celiac_disease_patients / all_celiac_disease_patients
            else:
                percentage_of_celiac_disease_that_is_diagnosed[age] = 0
            population_distribution = population_distribution @ transition_probabilities

        lifetime_risk_of_being_diagnosed = np.sum(annual_number_of_new_diagnoses) * 100
        cumulative_healthy_deaths = np.cumsum(annual_healthy_deaths) * 100
        cumulative_celiac_disease_deaths = np.cumsum(annual_celiac_disease_deaths) * 100
        mean_age_at_diagnosis = utils.age_weighted_average(annual_number_of_new_diagnoses)
        life_expectancy_of_population = utils.age_weighted_average(annual_celiac_disease_deaths + annual_celiac_disease_deaths)

        return {
            'Mean age at diagnosis': mean_age_at_diagnosis,
            'Lifetime risk of a celiac disease diagnosis': lifetime_risk_of_being_diagnosed,
            'Cumulative death of healthy people': cumulative_healthy_deaths,
            'Cumulative death with celiac disease': cumulative_celiac_disease_deaths,
            'Annual number of new celiac disease diagnoses': annual_number_of_new_diagnoses,
            'Life expectancy of population': life_expectancy_of_population,
            'Percentage of diagnosed celiac disease cases': percentage_of_celiac_disease_that_is_diagnosed,
            'Clinical prevalence per 1000': clinical_prevalence_per_1000,
        }

    def _get_transitions_between_given_states(self, population_distribution, transition_probability_matrix, from_states, to_states):
        from_state_indices = [i for i, x in enumerate(settings.states) if x in from_states]
        to_state_indices   = [i for i, x in enumerate(settings.states) if x in to_states]
        all_transitons_to_to_state = population_distribution * transition_probability_matrix[:,to_state_indices].T
        transition_between_states = all_transitons_to_to_state[:,from_state_indices]
        total_transitions = np.sum(transition_between_states)
        return total_transitions

    def _screen_population(self, population_distribution, test_for_hla, screening_round_number):
        # Assert that the initial sum of population_distribution is close to 1
        assert np.isclose(np.sum(population_distribution), 1.0), "Initial population distribution does not sum to 1"

        prevalence_of_cd_in_screening_participants = np.sum(population_distribution[1:4]) / np.sum(population_distribution[0:4])
        screening_sensitivity = self.screening.get_screening_sensitivity(test_for_hla)
        screening_specificity = self.screening.get_screening_specificity(test_for_hla, prevalence_of_cd_in_screening_participants)
        percentage_of_population_with_undiagnosed_celiac_disease = np.sum(population_distribution[1:4])
        percentage_of_population_healthy = population_distribution[0]
        percentage_of_population_participating_in_screening = percentage_of_population_with_undiagnosed_celiac_disease + percentage_of_population_healthy
        # Redistribute population according to how the screening went
        false_positives = percentage_of_population_healthy * (1 - screening_specificity) 
        percentage_of_population_with_new_celiac_disease_diagnosis = percentage_of_population_with_undiagnosed_celiac_disease * screening_sensitivity + false_positives
        population_distribution[0] *= screening_specificity
        population_distribution[1:4] *= (1 - screening_sensitivity)
        population_distribution[4] += percentage_of_population_with_new_celiac_disease_diagnosis * self.imported_data.compliance_after_diagnosis
        population_distribution[5] += percentage_of_population_with_new_celiac_disease_diagnosis * (1 - self.imported_data.compliance_after_diagnosis)
        screening_costs, biopsy_disutility = self.screening.calculate_screening_cost_and_disutility(percentage_of_population_participating_in_screening, percentage_of_population_with_undiagnosed_celiac_disease, screening_round_number, test_for_hla)

        # Assert that the initial sum of population_distribution is close to 1
        assert np.isclose(np.sum(population_distribution), 1.0), "Population distribution after screening does not sum to 1"

        return population_distribution, screening_costs, biopsy_disutility