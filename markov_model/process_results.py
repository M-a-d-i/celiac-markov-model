import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from settings import settings


class SimulationResults:
    def __init__(self, willingness_to_pay_threshold, population_distribution_over_time_without_screening, formated_population_level_statistics, num_cores):
        # Number of cores used for parallel processing
        self.num_cores = num_cores
        # Willingness to pay threshold for cost-effectiveness analysis
        self.willingness_to_pay_threshold = willingness_to_pay_threshold
        # Simulation results without screening
        self.population_distribution_over_time_without_screening = population_distribution_over_time_without_screening
        self.population_level_statistics_without_screening = formated_population_level_statistics
        # Simulation results with screening
        self.population_distribution_over_time_with_screening = {}
        # Simulation result statistics
        self.delta_costs, self.delta_qalys = {}, {}
        self.delta_costs_expected_value = {}
        self.delta_qalys_expected_value = {}
        self.icer_expected_value, self.icer_lower_bound, self.icer_upper_bound = {}, {}, {}
        self.icer_quadrant, self.icer_bootstrap_samples = {}, {}
        self.inhb_all, self.inhb_expected_value, self.inhb_lower_bound, self.inhb_upper_bound = {}, {}, {}, {}
        self.inmb_all, self.inmb_expected_value, self.inmb_lower_bound, self.inmb_upper_bound = {}, {}, {}, {}
        self.willingness_to_pay_vector = np.arange(0, 120000, 500)
        self.alternative_acceptability_curves = {}
        self.acceptability_curves, self.probabilities_at_willingness_threshold = {}, {}
        #self.probability_of_cost_effectiveness_at_threshold_bootstrap_samples = {}
        #self.mean_probability_of_cost_effectiveness_at_threshold = {}

        # Population groups
        self.undiagnosed_population_over_time_without_screening = np.array([
            [np.sum(elem[1:4]) for elem in population_distribution_over_time]
            for population_distribution_over_time in self.population_distribution_over_time_without_screening
        ])
        self.diagnosed_population_over_time_without_screening = np.array([
            [np.sum(elem[4:6]) for elem in population_distribution_over_time]
            for population_distribution_over_time in self.population_distribution_over_time_without_screening
        ])
        self.healthy_population_over_time_without_screening = np.array([
            [np.sum(elem[0:1]) for elem in population_distribution_over_time]
            for population_distribution_over_time in self.population_distribution_over_time_without_screening
        ])
        self.alive_population_over_time_without_screening = self.undiagnosed_population_over_time_without_screening + self.diagnosed_population_over_time_without_screening + self.healthy_population_over_time_without_screening
        self.population_with_celiac_disease_over_time_without_screening = self.undiagnosed_population_over_time_without_screening + self.diagnosed_population_over_time_without_screening

    def calculate_statistics_and_save_results(self, screening_strategies, results) -> list:
        if len(results) == 0:
            raise ValueError("Results array is empty. Cannot calculate statistics.")
        # Extract results from the input array
        delta_costs = results[:, 0].astype(np.float64)
        delta_qalys = results[:, 1].astype(np.float64)
        # Calculate statistics using helper functions
        alternative_acceptability_curves = self._calculate_alternative_acceptability_curves(delta_costs, delta_qalys)
        acceptability_curves, probabilities_at_willingness_threshold = self._calculate_probability_of_cost_effectiveness_for_all_screening_strategies(delta_costs, delta_qalys)
        #mean_probability_of_cost_effectiveness_at_threshold, probability_of_cost_effectiveness_at_threshold_bootstrap_samples = self._calculate_probability_of_cost_effectiveness_at_threshold_for_all_screening_strategies(delta_costs, delta_qalys)
        icer_means, icer_quadrants, icer_lower_bounds, icer_upper_bounds, icer_bootstrap_samples = self._calculate_icer_bootstrap_statistics_for_all_screening_strategies(delta_costs, delta_qalys)
        # Save calculated statistics for each screening strategy
        for i, strategy in tqdm(enumerate(screening_strategies), total=len(screening_strategies), desc='Saving results and calculating statistics'):
            self.delta_costs[strategy] = delta_costs[i]
            self.delta_qalys[strategy] = delta_qalys[i]
            self.delta_costs_expected_value[strategy] = np.mean(delta_costs[i])
            self.delta_qalys_expected_value[strategy] = np.mean(delta_qalys[i])

            self.icer_quadrant[strategy] = icer_quadrants[i]
            self.icer_expected_value[strategy] = icer_means[i]
            self.icer_lower_bound[strategy] = icer_lower_bounds[i]
            self.icer_upper_bound[strategy] = icer_upper_bounds[i]
            self.icer_bootstrap_samples[strategy] = icer_bootstrap_samples[i]

            self.inhb_all[strategy] = np.array([self._calculate_incremental_net_health_benefit(delta_costs[i][j], delta_qalys[i][j]) for j in range(len(delta_costs[i]))])
            self.inhb_expected_value[strategy] = self._calculate_incremental_net_health_benefit(self.delta_costs_expected_value[strategy], self.delta_qalys_expected_value[strategy])
            self.inhb_lower_bound[strategy], self.inhb_upper_bound[strategy] = self._net_health_benefit_95_percent_confidence_intervals(delta_costs[i], delta_qalys[i])

            self.inmb_all[strategy] = np.array([self._calculate_incremental_net_monetary_benefit(delta_costs[i][j], delta_qalys[i][j]) for j in range(len(delta_costs[i]))])
            self.inmb_expected_value[strategy] = self._calculate_incremental_net_monetary_benefit(self.delta_costs_expected_value[strategy], self.delta_qalys_expected_value[strategy])
            self.inmb_lower_bound[strategy], self.inmb_upper_bound[strategy] = self._incremental_net_monetary_benefit_95_percent_confidence_intervals(delta_costs[i], delta_qalys[i])

            self.alternative_acceptability_curves[strategy] = alternative_acceptability_curves[i]

            self.acceptability_curves[strategy] = acceptability_curves[i]
            self.probabilities_at_willingness_threshold[strategy] = probabilities_at_willingness_threshold[i]
            #self.probability_of_cost_effectiveness_at_threshold_bootstrap_samples[strategy] = probability_of_cost_effectiveness_at_threshold_bootstrap_samples[i]
            #self.mean_probability_of_cost_effectiveness_at_threshold[strategy] = mean_probability_of_cost_effectiveness_at_threshold[i]
        return screening_strategies

    def get_best_strategies(self, statistic):
        match statistic:
            case 'icer':
                # Be careful with this
                sorted_strategies = sorted(self.icer_expected_value.items(), key=lambda x: x[1], reverse=False)
            case 'inhb':
                sorted_strategies = sorted(self.inhb_expected_value.items(), key=lambda x: x[1], reverse=True)
            case 'inmb':
                sorted_strategies = sorted(self.inmb_expected_value.items(), key=lambda x: x[1], reverse=True)
            #case 'probability at willingness threshold':
            #    sorted_strategies = sorted(self.mean_probability_of_cost_effectiveness_at_threshold.items(), key=lambda x: x[1], reverse=True)
            case _:
                raise ValueError(f'Invalid statistic {statistic}')
        best_strategies = [strategy[0] for strategy in sorted_strategies]
        return best_strategies
    
    def _calculate_icer(self, delta_costs: np.float64, delta_qalys: np.float64) -> tuple[np.float64, str]:
        # Return both icer and quadrant based on the sign of delta_costs and delta_qalys
        try:
            icer = delta_costs / delta_qalys
        except ZeroDivisionError:
            raise ZeroDivisionError(f"Division by zero: delta_costs = {delta_costs}, delta_qalys = {delta_qalys}")

        if delta_costs > 0 and delta_qalys > 0:
            return icer, 'NE'
        if delta_costs < 0 and delta_qalys > 0:
            return icer, 'SE'
        if delta_costs < 0 and delta_qalys < 0:
            return icer, 'SW'
        if delta_costs > 0 and delta_qalys < 0:
            return icer, 'NW'
        return icer, 'Unknown'

    def _calculate_expected_value_of_icer(self, delta_costs: np.ndarray, delta_qalys: np.ndarray):
        mean_delta_costs = np.mean(delta_costs, dtype=np.float64)
        mean_delta_qalys = np.mean(delta_qalys, dtype=np.float64)
        icer, quadrant = self._calculate_icer(mean_delta_costs, mean_delta_qalys)
        return icer, quadrant

    def _calculate_icer_bootstrap_statistics_for_a_single_strategy(self, delta_costs: np.ndarray, delta_qalys: np.ndarray):
        # Draw 1000 samples with replacement
        delta_costs_bootstrap_sample = np.array([np.random.choice(delta_costs, len(delta_costs)) for _ in range(1000)])
        delta_qalys_bootstrap_sample = np.array([np.random.choice(delta_qalys, len(delta_qalys)) for _ in range(1000)])
        # Calculate total mean and quadrant for the bootstrap sample
        delta_costs_mean = np.mean(delta_costs_bootstrap_sample, dtype=np.float64)
        delta_qalys_mean = np.mean(delta_qalys_bootstrap_sample, dtype=np.float64)
        icer_mean, quadrant = self._calculate_icer(delta_costs_mean, delta_qalys_mean)
        # Calculate bootstrap ICERS
        icer_boostrap_samples = np.array([
            self._calculate_expected_value_of_icer(delta_costs_bootstrap_sample[i], delta_qalys_bootstrap_sample[i])[0]
            for i in range(len(delta_costs_bootstrap_sample))
        ])
        icer_lower_bound = np.quantile(icer_boostrap_samples, 0.025)
        icer_upper_bound = np.quantile(icer_boostrap_samples, 0.975)
        return icer_mean, quadrant, icer_lower_bound, icer_upper_bound, icer_boostrap_samples

    def _calculate_icer_bootstrap_statistics_for_all_screening_strategies(self, delta_costs: np.ndarray, delta_qalys: np.ndarray):
        number_of_screening_strategies = len(delta_costs)
        statistics = np.array(
            Parallel(n_jobs=self.num_cores)(
            delayed(self._calculate_icer_bootstrap_statistics_for_a_single_strategy)(delta_costs[i], delta_qalys[i]) 
            for i in tqdm(range(number_of_screening_strategies), desc='Calculating ICER bootstraps')), dtype=object
        )
        icer_mean = statistics[:, 0]
        quadrant = statistics[:, 1]
        icer_lower_bound = statistics[:, 2]
        icer_upper_bound = statistics[:, 3]
        icer_bootstrap_samples = statistics[:, 4]
        return icer_mean, quadrant, icer_lower_bound, icer_upper_bound, icer_bootstrap_samples

    def _calculate_incremental_net_health_benefit(self, delta_costs: np.float64, delta_qalys: np.float64) -> np.float64:
        return delta_qalys - delta_costs / self.willingness_to_pay_threshold

    def _net_health_benefit_95_percent_confidence_intervals(self, delta_costs: np.ndarray, delta_qalys: np.ndarray) -> tuple[float, float]:
        n = len(delta_costs)
        nhb_variance = (np.var(delta_qalys) + (np.var(delta_costs) / self.willingness_to_pay_threshold** 2) - (2 * np.cov(delta_costs, delta_qalys)[0][1] / self.willingness_to_pay_threshold))/ n
        nhb_mean = self._calculate_incremental_net_health_benefit(np.mean(delta_costs, dtype=np.float64), np.mean(delta_qalys, dtype=np.float64))
        nhb_lower_bound = nhb_mean - 1.96 * np.sqrt(nhb_variance)
        nhb_upper_bound = nhb_mean + 1.96 * np.sqrt(nhb_variance)
        return nhb_lower_bound, nhb_upper_bound

    def _calculate_incremental_net_monetary_benefit(self, delta_costs: np.float64, delta_qalys: np.float64) -> np.float64:
        return delta_qalys * self.willingness_to_pay_threshold - delta_costs

    def _incremental_net_monetary_benefit_95_percent_confidence_intervals(self, delta_costs: np.ndarray, delta_qalys: np.ndarray) -> tuple[float, float]:
        n = len(delta_costs)
        nmb_variance = ((self.willingness_to_pay_threshold ** 2 * np.var(delta_qalys)) + np.var(delta_costs) - (2 * self.willingness_to_pay_threshold * np.cov(delta_costs, delta_qalys)[0][1])) / n
        nmb_mean = self._calculate_incremental_net_monetary_benefit(np.mean(delta_costs, dtype=np.float64), np.mean(delta_qalys, dtype=np.float64))
        nmb_lower_bound = nmb_mean - 1.96 * np.sqrt(nmb_variance)
        nmb_upper_bound = nmb_mean + 1.96 * np.sqrt(nmb_variance)
        return nmb_lower_bound, nmb_upper_bound

    def _calculate_probability_of_cost_effectiveness(self, delta_costs, delta_qalys):
        number_of_simulations = len(delta_costs)
        # Define indices where differences are positive and negative for costs and qalys
        delta_costs_positive_indices = np.where(delta_costs > 0)[0]
        delta_costs_negative_indices = np.where(delta_costs < 0)[0]
        delta_q_positive_indices = np.where(delta_qalys > 0)[0]
        # Number of results where screening dominates, calculate in cost-effectiveness
        number_of_screening_domination = len(np.intersect1d(delta_costs_negative_indices, delta_q_positive_indices))
        # Result indices where paying money results in more qalys
        pay_money_for_more_qaly_indices = np.intersect1d(delta_costs_positive_indices, delta_q_positive_indices)
        icer_values_of_tradeoff = delta_costs[pay_money_for_more_qaly_indices] / delta_qalys[pay_money_for_more_qaly_indices]
        acceptability_curve = np.array([
            np.sum(icer_values_of_tradeoff < i) / number_of_simulations for i in self.willingness_to_pay_vector
        ])
        acceptability_curve = acceptability_curve + (number_of_screening_domination / number_of_simulations)
        acceptability_curve *= 100

        index_of_willingness_threshold = np.where(self.willingness_to_pay_vector == self.willingness_to_pay_threshold)[0][0]
        probability_at_willingness_threshold = acceptability_curve[index_of_willingness_threshold]

        return acceptability_curve, probability_at_willingness_threshold

    def _calculate_probability_of_cost_effectiveness_for_all_screening_strategies(self, delta_costs, delta_qalys):
        number_of_screening_strategies = len(delta_costs)
        probabilities = np.array(
            Parallel(n_jobs=self.num_cores, prefer="threads")(
            delayed(self._calculate_probability_of_cost_effectiveness)(delta_costs[i], delta_qalys[i]) 
            for i in tqdm(range(number_of_screening_strategies), desc='Calculating acceptability curves')), dtype=object
        )
        acceptability_curves = probabilities[:, 0]
        probabilities_at_willingness_threshold = probabilities[:, 1]
        return acceptability_curves, probabilities_at_willingness_threshold

    # Functionality not complete (not utilized)
    def _calculate_alternative_acceptability_curves(self, delta_costs, delta_qalys):
        number_of_simulations = len(delta_costs[0])
        number_of_strategies = len(delta_costs)
        
        inmb_matrix = np.zeros((number_of_strategies, number_of_simulations, len(self.willingness_to_pay_vector)))
        
        for i, wtp in enumerate(self.willingness_to_pay_vector):
            for j in range(number_of_strategies):
                inmb_matrix[j, :, i] = delta_qalys[j] * wtp - delta_costs[j]
        
        best_strategy_counts = np.zeros((number_of_strategies, len(self.willingness_to_pay_vector)))
        
        for i in range(len(self.willingness_to_pay_vector)):
            best_strategy_indices = np.argmax(inmb_matrix[:, :, i], axis=0)
            for j in range(number_of_strategies):
                best_strategy_counts[j, i] = np.sum(best_strategy_indices == j)
        
        alternative_acceptability_curves = best_strategy_counts / number_of_simulations * 100
        
        return alternative_acceptability_curves

    def _calculate_probability_of_cost_effectiveness_at_threshold_for_a_single_screening_strategy(self, delta_costs, delta_qalys):
        # Create an array with 1's and 0's, that tells if the screening was cost-effective or not, calculate the convergence from that
        def is_cost_effective(delta_cost, delta_qaly):
            if delta_cost < 0 and delta_qaly > 0:
                return 1
            if delta_cost > 0 and delta_qaly > 0 and delta_cost / delta_qaly < self.willingness_to_pay_threshold:
                return 1
            return 0
        # This function should be applied to all elements of the delta_costs and delta_qalys arrays, the result should be an array of 1's and 0's
        vectorized_is_cost_effective = np.vectorize(is_cost_effective)
        # Draw 1000 samples with replacement
        delta_costs_bootstrap_sample = np.array([np.random.choice(delta_costs, len(delta_costs)) for _ in range(1000)])
        delta_qalys_bootstrap_sample = np.array([np.random.choice(delta_qalys, len(delta_qalys)) for _ in range(1000)])
        # Calculate bootstraps
        probability_of_cost_effectiveness_at_threshold_bootstrap = np.array([
            np.mean(vectorized_is_cost_effective(delta_costs_bootstrap_sample[i], delta_qalys_bootstrap_sample[i]))
            for i in range(len(delta_costs_bootstrap_sample))
        ])
        mean_probability_of_cost_effectiveness_at_threshold = np.mean(probability_of_cost_effectiveness_at_threshold_bootstrap)

        return mean_probability_of_cost_effectiveness_at_threshold, probability_of_cost_effectiveness_at_threshold_bootstrap

    def _calculate_probability_of_cost_effectiveness_at_threshold_for_all_screening_strategies(self, delta_costs, delta_qalys):
        number_of_screening_strategies = len(delta_costs)
        probability_of_cost_effectiveness_at_threshold_bootstrap = np.array(
            Parallel(n_jobs=self.num_cores)(
            delayed(self._calculate_probability_of_cost_effectiveness_at_threshold_for_a_single_screening_strategy)(delta_costs[i], delta_qalys[i]) 
            for i in tqdm(range(number_of_screening_strategies), desc='Calculating bootstraps for cost-effectiveness probability')), dtype=object
        )
        mean_probability_of_cost_effectiveness_at_threshold = probability_of_cost_effectiveness_at_threshold_bootstrap[:, 0]
        probability_of_cost_effectiveness_at_threshold_bootstrap_samples = probability_of_cost_effectiveness_at_threshold_bootstrap[:, 1]
        return mean_probability_of_cost_effectiveness_at_threshold, probability_of_cost_effectiveness_at_threshold_bootstrap_samples

    def export_to_csv(self, directory: str, filename: str):
        # Sort the dictionary keys
        sorted_keys = sorted(self.delta_costs.keys(), key=lambda x: (x[0], x[1]))
        
        data = {
            'strategy': sorted_keys,
            'icer_expected_value': [self.icer_expected_value[key] for key in sorted_keys],
            'icer_lower_bound': [self.icer_lower_bound[key] for key in sorted_keys],
            'icer_upper_bound': [self.icer_upper_bound[key] for key in sorted_keys],
            'icer_quadrant': [self.icer_quadrant[key] for key in sorted_keys],
            'delta_costs_expected_value': [self.delta_costs_expected_value[key] for key in sorted_keys],
            'delta_qalys_expected_value': [self.delta_qalys_expected_value[key] for key in sorted_keys],
            'inhb expected value': [self.inhb_expected_value[key] for key in sorted_keys],
            'inhb lower bound': [self.inhb_lower_bound[key] for key in sorted_keys],
            'inhb upper bound': [self.inhb_upper_bound[key] for key in sorted_keys],
            'inmb expected value': [self.inmb_expected_value[key] for key in sorted_keys],
            'inmb lower bound': [self.inmb_lower_bound[key] for key in sorted_keys],
            'inmb upper bound': [self.inmb_upper_bound[key] for key in sorted_keys],
            'delta_costs': [self.delta_costs[key] for key in sorted_keys],
            'delta_qalys': [self.delta_qalys[key] for key in sorted_keys],
            'icer_bootstrap_samples': [self.icer_bootstrap_samples[key] for key in sorted_keys],
            'acceptability_curves': [
                [(wtp, acc) for wtp, acc in zip(self.willingness_to_pay_vector, self.acceptability_curves[key])]
                for key in sorted_keys
            ],
        }
        df = pd.DataFrame(data)
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)