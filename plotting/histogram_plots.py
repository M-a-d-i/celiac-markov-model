import matplotlib.pylab as plt
import numpy as np

from plotting import big_figure, medium_figure
from utilities import utils


class HistogramPlots:
    def __init__(self, simulation_results):
        self.simulation_results = simulation_results
        self.bins = 20

    def _initialize_plot(self, title, size):
        fig, ax = plt.subplots(1, 1, num=title, constrained_layout=True)
        fig.set_size_inches(size)
        return fig, ax

    def _plot_histogram(self, data, title, xlabel, ylabel, size, title_suffix = '', mean_unit = ''):
        mean, lower_bound, upper_bound = utils.iid_95_confidence_intervals_and_mean(data)
        fig, ax = self._initialize_plot(title, size)
        ax.hist(data, bins=self.bins)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=f'{title} \nMean: {mean:.2f}{mean_unit} (95% CI: {lower_bound:.2f}-{upper_bound:.2f}), {title_suffix}')
        ax.grid()

    def mean_age_at_diagnosis_histogram(self):
        title = 'Mean age at diagnosis'
        mean_age_at_diagnosis = self.simulation_results.population_level_statistics_without_screening.get(title)
        self._plot_histogram(mean_age_at_diagnosis, title, 'Age (years)', 'Frequency', medium_figure, 'target: 32')

    def lifetime_risk_of_diagnosis_histogram(self):
        # plot the distribution of Lifetime risk of a celiac disease diagnosis
        title = 'Lifetime risk of a celiac disease diagnosis'
        lifetime_risk_of_being_diagnosed = self.simulation_results.population_level_statistics_without_screening.get(title)
        self._plot_histogram(lifetime_risk_of_being_diagnosed, title, 'Percentage of population reaching a diagnosed state (%)', 'Frequency', medium_figure, 'target: 1.8%', '%')

    def percentage_of_12_year_olds_with_symptomatic_or_clinical_evaluation_cd_histogram(self):
        # plot the percentage of 12 year olds with undiagnosed celiac disease that have symptomatic or clinical evaluation celiac disease
        title = 'Undiagnosed celiac disease and reduced quality of life at age 12'
        undiagnosed_celiac_disease_population_at_age_12 = self.simulation_results.undiagnosed_population_over_time_without_screening[:, 12]
        percentage_of_population_in_symptomatic_or_clinical_evaluation = np.array([elem[12, 2] + elem[12, 3] for elem in self.simulation_results.population_distribution_over_time_without_screening])
        percentage_of_undiagnosed_cd_in_symptomatic_or_clinical_evaluation = 100 * percentage_of_population_in_symptomatic_or_clinical_evaluation / undiagnosed_celiac_disease_population_at_age_12
        self._plot_histogram(percentage_of_undiagnosed_cd_in_symptomatic_or_clinical_evaluation, title, 'Undiagnosed celiac disease cases with reduced quality of life (%)', 'Frequency', medium_figure, 'target: 7.2%', '%')

    def lifetime_probability_of_not_getting_diagnosed(self):
        # plot the distribution of average underdiagnosis percentage
        title = 'Lifetime probability of not getting diagnosed'
        population_with_celiac_disease_over_time_without_screening = np.copy(self.simulation_results.population_with_celiac_disease_over_time_without_screening)
        population_with_celiac_disease_over_time_without_screening[population_with_celiac_disease_over_time_without_screening == 0] = np.nan
        percentage_of_diagnosed_celiac_disease = 100 * self.simulation_results.diagnosed_population_over_time_without_screening / population_with_celiac_disease_over_time_without_screening
        percentage_of_diagnosed_celiac_disease = np.nan_to_num(percentage_of_diagnosed_celiac_disease)
        percentage_of_diagnosed_celiac_disease_averages = np.average(percentage_of_diagnosed_celiac_disease, axis=1, weights=self.simulation_results.population_with_celiac_disease_over_time_without_screening)
        self._plot_histogram(percentage_of_diagnosed_celiac_disease_averages, title, 'Percentage of celiac disease patients diagnosed (%)', 'Frequency', medium_figure, '%', '%')