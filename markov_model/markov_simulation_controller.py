import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from data.import_data import ImportData
from markov_model.markov_model import MarkovModel
from markov_model.process_results import SimulationResults
from random_parameter_generator.random_parameter_generator import RandomParameters
from settings import settings
from utilities import utils


class MarkovSimulationController(MarkovModel):
    def __init__(self, num_of_simulations: int, willingness_to_pay_threshold: int, filename: str, path: str = 'data/parameters/'):
        # Import data
        self.imported_data = ImportData(filename, path)
        MarkovModel.__init__(self, self.imported_data)
        self.discount_rate = self.imported_data.discounting['rate']
        self.first_discount_year = self.imported_data.discounting['first_discount_year']
        # Number of cores used for parallel processing
        self.num_cores: int = multiprocessing.cpu_count()
        # Pre-generate random parameters 
        self.num_of_simulations = num_of_simulations
        self.willingness_to_pay_threshold = willingness_to_pay_threshold
        self.random_parameters = RandomParameters(self.num_of_simulations, self.imported_data)
        self.qaly_matrices = self.random_parameters.qaly_matrices
        self.cost_matrices = self.random_parameters.cost_matrices
        self.transition_probability_matrices = self.random_parameters.transition_probability_matrices
        # Store and process simulation results
        self.population_distribution_over_time_without_screening, self.population_level_statistics_without_screening = self.get_population_level_statistics(settings.no_screening)
        self.simulation_results = SimulationResults(self.willingness_to_pay_threshold, self.population_distribution_over_time_without_screening, self.population_level_statistics_without_screening, self.num_cores)

    def reinitialize_markov_model(self):
        MarkovModel.__init__(self, self.imported_data)

    def simulate_screening_strategies(self, screening_age_range):
        all_screening_strategies = utils.generate_all_screening_strategies(screening_age_range)
        already_simulated_screening_strategies = set(self.simulation_results.delta_qalys.keys())
        new_screening_strategies = list(set(all_screening_strategies).difference(already_simulated_screening_strategies))
        results = np.array(
            Parallel(n_jobs=self.num_cores)(
            delayed(self._monte_carlo_simulation)(new_screening_strategies[i]) 
            for i in tqdm(range(len(new_screening_strategies)), desc='Number of screening strategies simulated')), dtype=object
        )
        screening_strategies = self.simulation_results.calculate_statistics_and_save_results(new_screening_strategies, results)
        return screening_strategies

    def simulate_single_screening_strategy(self, screening_strategy):
        # Check if the strategy has already been simulated
        if screening_strategy in self.simulation_results.delta_qalys.keys():
            return screening_strategy
        results = self._parallel_monte_carlo_simulation(screening_strategy)
        screening_strategies = self.simulation_results.calculate_statistics_and_save_results([screening_strategy], np.array([results]))
        return screening_strategies[0]

    def get_population_level_statistics(self, screening_strategy):
        def single_simulation(screening_strategy, transition_probability_matrix):
            # Without screening
            population_distribution_over_time_without_screening, _, _ = self.simulate_markov_chain(screening_strategy, transition_probability_matrix) 
            # Calculate population level statistics used for tuning the model
            population_level_statistics = self.get_population_level_statistics_of_a_single_simulation(screening_strategy, transition_probability_matrix)
            return population_distribution_over_time_without_screening, population_level_statistics
        desc = 'Simulating Markov model'
        if screening_strategy == settings.no_screening:
            desc += ' without screening'
        else:
            desc += ' with screening'
            genotyping_desc = ' with genotyping' if screening_strategy[1] == True else ' without genotyping'
            screening_ages_desc = ' at age(s): ' + ', '.join([str(age) for age in screening_strategy[0]])
            desc += screening_ages_desc
            desc += genotyping_desc
        results = np.array(
            Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(single_simulation)(screening_strategy, self.transition_probability_matrices[iteration_number]) 
            for iteration_number in tqdm(range(self.num_of_simulations), desc=desc)), dtype=object
        )
        population_distribution_over_time = results[:,0]
        unformatted_population_level_statistics = results[:,1]
        formatted_population_level_statistics = {}
        for key in unformatted_population_level_statistics[0]:
            formatted_population_level_statistics[key] = np.array([elem[key] for elem in unformatted_population_level_statistics])
        return population_distribution_over_time, formatted_population_level_statistics 
    
    def _parallel_monte_carlo_simulation(self, screening_strategy):
        results = np.array(
            Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(self._simulation)(screening_strategy, iteration_number)
            for iteration_number in tqdm(range(self.num_of_simulations), desc='Simulating single screening strategy')), dtype=object
        )
        delta_costs, delta_qalys = results[:,0], results[:,1]
        return delta_costs, delta_qalys

    def _monte_carlo_simulation(self, screening_strategy):
        results = np.array([
            self._simulation(screening_strategy, iteration_number) 
            for iteration_number in range(self.num_of_simulations)
        ], dtype=object)
        delta_costs, delta_qalys = results[:,0], results[:,1]
        return delta_costs, delta_qalys

    def _simulation(self, screening_strategy, iteration_number):
        # Make sure we use the same transition probability matrix for the same iteration number
        population_without_screening = self.population_distribution_over_time_without_screening[iteration_number]
        transition_probability_matrix = self.transition_probability_matrices[iteration_number]
        cost_matrix = self.cost_matrices[:,:,iteration_number]
        qaly_matrix = self.qaly_matrices[:,:,iteration_number]
        # Lifetime costs and QALYs without screening
        costs_without_screening = utils.calculate_lifetime_costs_or_qalys(population_without_screening, cost_matrix, self.discount_rate, self.first_discount_year)
        qalys_without_screening = utils.calculate_lifetime_costs_or_qalys(population_without_screening, qaly_matrix, self.discount_rate, self.first_discount_year)
        # Lifetime costs and QALYs with screening
        population_with_screening, screening_costs, biopsy_disutility = self.simulate_markov_chain(screening_strategy, transition_probability_matrix)
        costs_with_screening = utils.calculate_lifetime_costs_or_qalys(population_with_screening, cost_matrix, self.discount_rate, self.first_discount_year) + screening_costs
        qalys_with_screening = utils.calculate_lifetime_costs_or_qalys(population_with_screening, qaly_matrix, self.discount_rate, self.first_discount_year) - biopsy_disutility
        # Differences in lifetime costs and QALYs
        delta_costs = costs_with_screening - costs_without_screening
        delta_qalys = qalys_with_screening - qalys_without_screening
        return delta_costs, delta_qalys