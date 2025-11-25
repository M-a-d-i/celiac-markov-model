import numpy as np
import pandas as pd
import yaml

from settings import settings


class ImportData:
    def __init__(self, filename: str, path: str = 'data/parameters/'):
        self.path = path
        self.filename = filename
        self.load_data()

    def load_data(self):
        with open(self.path + self.filename, 'r') as file:
            self.model_config = yaml.load(file, Loader=yaml.FullLoader)

        self.screening_parameters = self.model_config['screening_parameters']

        self.transition_probabilities = self.model_config['transition_probabilities']
        self.prevalence_data = self.model_config['transition_probabilities']['celiac_disease_prevalence']
        self.initial_celiac_disease_transitions = self.model_config['transition_probabilities']['initial_celiac_disease_transitions']
        self.asymptomatic_transitions = self.model_config['transition_probabilities']['asymptomatic_transitions']
        self.diagnostic_delays = self.model_config['transition_probabilities']['diagnostic_delays']
        self.gluten_free_diet_compliance = self.model_config['transition_probabilities']['gluten_free_diet_compliance']
        self.compliance_after_diagnosis = self.gluten_free_diet_compliance['compliance_after_diagnosis']

        self.societal_costs = self.model_config['societal_costs']['cost_sources']
        self.healthcare_visit_unit_cost = self.societal_costs['healthcare_visit']['unit_cost']
        self.hospitalization_day_unit_cost = self.societal_costs['hospitalization_day']['unit_cost']
        self.sick_day_unit_cost = self.societal_costs['sick_day']['unit_cost']

        self.qaly_expected_values, self.qaly_standard_deviations = self.expand_qaly_data_to_all_ages(self.model_config['quality_of_life'])
        self.societal_cost_data = self.expand_societal_cost_data_to_all_ages(self.model_config['societal_costs'])
        self.mortalities = self.get_mortalities(self.model_config['transition_probabilities']['mortality'])
        self.mortality_hazard_ratio = self.model_config['transition_probabilities']['mortality']['hazard_ratio']

        self.discounting = self.model_config['discounting']

    def expand_value_range_to_all_ages(self, age_data, property_data):
        result_array = np.zeros(settings.number_of_years_to_simulate)
        for age_range, rate in zip(age_data, property_data):
            start_age, end_age = age_range.split('-')
            start_age = int(start_age)
            end_age = int(end_age) if end_age else settings.number_of_years_to_simulate - 1
            result_array[start_age:end_age+1] = rate
        return result_array

    def get_mortalities(self, mortality_data):
        age_data = mortality_data['age']
        mortality_data = mortality_data['mortality_rate']
        mortalities = self.expand_value_range_to_all_ages(age_data, mortality_data)
        return mortalities

    def expand_qaly_data_to_all_ages(self, qaly_data):
        age_ranges = qaly_data['age']
        expected_values = qaly_data['expected_values']
        standard_deviations = qaly_data['standard_deviations']

        number_of_years = settings.number_of_years_to_simulate
        number_of_states = len(expected_values)

        expected_values_array = np.zeros((number_of_states, number_of_years))
        standard_deviations_array = np.zeros((number_of_states, number_of_years))

        for state_index, state in enumerate(expected_values):
            expected_values_array[state_index] = self.expand_value_range_to_all_ages(age_ranges, expected_values[state])
            standard_deviations_array[state_index] = self.expand_value_range_to_all_ages(age_ranges, standard_deviations[state])
        
        return expected_values_array, standard_deviations_array

    def expand_societal_cost_data_to_all_ages(self, societal_cost_data):
        age_ranges = societal_cost_data['age']
        cost_sources = societal_cost_data['cost_sources']

        number_of_years = settings.number_of_years_to_simulate

        cost_data = {}

        for cost_source, cost_details in cost_sources.items():
            unit_cost = cost_details['unit_cost']
            expected_values = cost_details['expected_values']
            standard_deviations = cost_details['standard_deviations']

            expected_values_df = pd.DataFrame(index=range(number_of_years))
            standard_deviations_df = pd.DataFrame(index=range(number_of_years))

            for state in expected_values:
                expected_values_df[state] = self.expand_value_range_to_all_ages(age_ranges, expected_values[state])
                standard_deviations_df[state] = self.expand_value_range_to_all_ages(age_ranges, standard_deviations[state])

            cost_data[cost_source] = {
                'unit_cost': unit_cost,
                'expected_values': expected_values_df,
                'standard_deviations': standard_deviations_df
            }

        return cost_data