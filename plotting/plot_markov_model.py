import os

import matplotlib
import matplotlib.pylab as plt
from tqdm import tqdm

from plotting.box_plots import BoxPlots
from plotting.heatmap_plots import HeatmapPlots
from plotting.histogram_plots import HistogramPlots
from plotting.line_plots import LinePlots
from plotting.scatter_plots import ScatterPlots

matplotlib.use('webAgg')


class PlotMarkovModel(BoxPlots, ScatterPlots, LinePlots, HistogramPlots, HeatmapPlots):
    def __init__(self, simulation_results, willingness_to_pay_threshold):
       BoxPlots.__init__(self, simulation_results, willingness_to_pay_threshold)
       ScatterPlots.__init__(self, simulation_results)
       LinePlots.__init__(self, simulation_results, willingness_to_pay_threshold)
       HistogramPlots.__init__(self, simulation_results )
       HeatmapPlots.__init__(self, simulation_results)

    def main_results(self, screening_strategies):
        self.plot_cost_effectiveness_heatmap(shared_colorbar=True, use_transparency=False, metric='icer')
        self.plot_cost_effectiveness_heatmap(shared_colorbar=True, use_transparency=False, metric='inmb')
        self.acceptability_curves(screening_strategies)
        self.scatter_all_screening_strategies_costs_and_qalys()
        #self.alternative_acceptability_curves()

    def model_calibration_statistics(self):
        self.prevalence()
        self.clinical_prevalence_per_1000_as_a_function_of_age()
        self.percentage_of_celiac_disease_diagnosed()
        self.percentage_of_12_year_olds_with_symptomatic_or_clinical_evaluation_cd_histogram()
        self.mean_age_at_diagnosis_histogram()
        self.lifetime_risk_of_diagnosis_histogram()

    def supplementary_figures(self, markov_model, number_of_top_strategies_to_plot, screening_strategy = ((11,), False)):
        screening_strategy = markov_model.simulate_single_screening_strategy(screening_strategy)
        self.population_distribution_over_time()
        self.compare_population_distribution_with_and_without_screening(markov_model, screening_strategy)
        self.celiac_disease_population_over_time()
        self.prevalence_and_cumulative_risk(markov_model.imported_data.prevalence_data)
        self.percentage_of_population_with_cd_diagnosis()
        self.percentage_of_asymptomatic_cd_without_diagnosis()
        self.annual_percentage_of_undiagnosed_cd_diagnosed()
        self.lifetime_probability_of_not_getting_diagnosed()
        self.scatter_single_screening_strategy_costs_and_qalys(screening_strategy)
        self.acceptability_curves([screening_strategy])
        #self.scatter_best_cost_effectiveness_probability_and_expected_value_costs_and_qalys()
        #self.best_acceptability_curves(number_of_top_strategies_to_plot)
        self.box_plot_best_icers(number_of_top_strategies_to_plot)
        self.box_plot_screening_age_against_icers()
        self.box_plot_screening_age_against_icer_with_best_repeated_screening_strategies()
        #self.box_plot_best_cost_effectiveness_probabilities_at_threshold(number_of_top_strategies_to_plot)
        self.box_plot_all_repeated_screening_icers()
        self.box_plot_all_repeated_screening_icers_ordered()
        self.box_plot_all_repeated_screening_icers_by_time_between_screenings()

    def save_figures_and_export_results(self, screening_age_range, num_of_simulations, filename, show_figures=True):
        filename_without_extension = os.path.splitext(filename)[0]
        subfolder_name = os.path.join('results', f'{filename_without_extension}', f'ages {screening_age_range} - {num_of_simulations} simulations')
        figure_subfolder_name = os.path.join(subfolder_name, 'figures')
        
        counter = 1
        original_subfolder_name = subfolder_name
        while os.path.exists(subfolder_name) or os.path.exists(figure_subfolder_name):
            subfolder_name = f"{original_subfolder_name}_{counter}"
            figure_subfolder_name = os.path.join(subfolder_name, 'figures')
            counter += 1
        
        os.makedirs(subfolder_name)
        os.makedirs(figure_subfolder_name)

        figure_labels = plt.get_figlabels()
        for figure_label in tqdm(figure_labels, desc="Saving figures to pdf"):
            figure_path_name = os.path.join(figure_subfolder_name, f'{figure_label}.pdf')
            plt.figure(figure_label)
            plt.savefig(figure_path_name, bbox_inches='tight')

        # Export simulation results to CSV
        csv_filename = 'simulation_results.csv'
        self.simulation_results.export_to_csv(subfolder_name, csv_filename)

        if show_figures:
            plt.show()

    def clear_figures(self):
        plt.close('all')
