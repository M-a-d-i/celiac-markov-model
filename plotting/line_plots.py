import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from plotting import big_figure, huge_figure, medium_figure, wide_figure
from random_parameter_generator.generate_risk_of_celiac_disease import RiskOfCeliacDisease
from settings import settings
from utilities import utils


class LinePlots:
    def __init__(self, simulation_results, willingness_to_pay_threshold):
        self.simulation_results = simulation_results
        self.willingness_to_pay_threshold = willingness_to_pay_threshold
    
    def _plot_mean_with_confidence_intervals(self, ax, data, title, x_label, y_label, mean_label='Mean'):
        mean, lower_bound, upper_bound = utils.iid_95_confidence_intervals_and_mean(data)
        ax.plot(mean, label=mean_label)
        ax.plot(lower_bound, label='95% confidence interval', color='gray', linestyle='dashed')
        ax.plot(upper_bound, color='gray', linestyle='dashed')
        ax.set(xlabel=x_label, ylabel=y_label, title=title)
        ax.grid(visible=True)
        ax.legend()

    def population_distribution_over_time(self):
        title = 'Population distribution over time'
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, num = title)

        population_distribution_over_time = np.mean(self.simulation_results.population_distribution_over_time_without_screening, axis=0) * 100
        celiac_disease_population = np.array([elem[1:6] for elem in population_distribution_over_time])
        ax1.plot(population_distribution_over_time, label = settings.states)
        ax2.plot(celiac_disease_population, label = settings.states[1:6])

        for ax in [ax1, ax2]:
            ax.set_xlabel('Age (years)', fontsize=20, weight='bold')
            ax.set_ylabel('% of population', fontsize=20, weight='bold')
            ax.set_xlim(left=0, right = settings.number_of_years_to_simulate)
            ax.set_ylim(bottom=0)
            ax.legend()

        fig.set_size_inches(*medium_figure)
        fig.tight_layout()

    def compare_population_distribution_with_and_without_screening(self, markov_model, screening_strategy, num_years = settings.number_of_years_to_simulate):
        title = 'Population distribution over time with and without screening'
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, num = title)

        population_without_screening = np.mean(self.simulation_results.population_distribution_over_time_without_screening, axis=0) * 100
        celiac_disease_population_without_screening = np.array([elem[1:6] for elem in population_without_screening])[:, :num_years]

        population_with_screening, _ = markov_model.get_population_level_statistics(screening_strategy)
        population_with_screening = np.mean(population_with_screening, axis=0) * 100
        celiac_disease_population_with_screening = np.array([elem[1:6] for elem in population_with_screening])[:, :num_years]

        # Updated color-blind friendly palette
        colors = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#0072B2', '#D55E00', '#CC79A7']

        for i, state in enumerate(settings.states):
            ax1.plot(population_without_screening[:, i], label=state, color=colors[i % len(colors)], linewidth=2)
        for i, state in enumerate(settings.states[1:6]):
            ax2.plot(celiac_disease_population_without_screening[:, i], label=state, color=colors[(i+1) % len(colors)], linewidth=2)
            ax3.plot(celiac_disease_population_with_screening[:, i], label=state, color=colors[(i+1) % len(colors)], linewidth=2)

        for ax, label in zip([ax1, ax2, ax3], ['A)', 'B)', 'C)']):
            ax.set_ylabel('% of population', fontsize=20, weight='bold')
            ax.set_xlim(left=0, right=num_years)
            ax.set_ylim(bottom=0)
            ax.set_xticks(np.arange(0, num_years + 1, step=5))
            ax.text(0, 1.1, label, transform=ax.transAxes, fontsize=24, weight='bold', va='top', ha='right')

        ax1.legend()
        ax3.set_xlabel('Age (years)', fontsize=20, weight='bold')

        figure_w, figure_h = medium_figure
        figure_h = figure_h * 2.5
        figure_size = (figure_w, figure_h)

        fig.set_size_inches(*figure_size)
        fig.tight_layout(pad=2.0)

    def alternative_acceptability_curves(self):
        title = f'Alternative cost-effectiveness acceptability curves'
        fig, ax = plt.subplots(num = title)
        screening_strategies = self.simulation_results.alternative_acceptability_curves.keys()
        for screening_strategy in screening_strategies:
            ax.plot(self.simulation_results.willingness_to_pay_vector, self.simulation_results.alternative_acceptability_curves[screening_strategy], utils.get_color(screening_strategy))
        ax.set(xlabel='Willingness to pay (€)', ylabel='Probability of cost-effectiveness (%)')
        ax.set_xlim(left=0, right=105000)
        ax.set_ylim(bottom=0)
        ax.axhline(color = 'k', linewidth=1)
        ax.axvline(color = 'k', linewidth=1)
        ax.grid(visible=True)
        fig.set_size_inches(*big_figure)
        fig.tight_layout()

    def acceptability_curves(self, screening_strategies):
        number_of_strategies = len(screening_strategies)
        title = f'Cost-effectiveness acceptability'
        if number_of_strategies == 1:
            hla_strategy = "no HLA genotyping" if not screening_strategies[0][1] else "with HLA genotyping"
            screening_strategy = f' curve, Screening ages(s) {screening_strategies[0][0]}, ' + hla_strategy
            title += screening_strategy
        else:
            title += ' curves'
        fig, ax = plt.subplots(num = title)
        for screening_strategy in screening_strategies:
            ax.plot(self.simulation_results.willingness_to_pay_vector, self.simulation_results.acceptability_curves[screening_strategy], utils.get_color(screening_strategy))
        colors = utils.get_colors(screening_strategies)
        colors_for_legends = set(colors)
        color_to_label_mapping = {value: key for key, value in settings.plot_colors.items()}
        preferred_legend_order = [
            'Single-time antibody screening',
            'First-line genetic + single-time antibody screening',
            'Repeated antibody screening',
            'First-line genetic + repeated antibody screening',
        ]
        legend_colors = {color_to_label_mapping[color]:color for color in colors_for_legends}
        #labels = list(legend_colors.keys())
        ordered_labels = [label for label in preferred_legend_order if label in legend_colors]
        ordered_handles = [Rectangle((0,0),1,1, color=legend_colors[label]) for label in ordered_labels]
        ax.legend(ordered_handles, ordered_labels, loc = 'lower right')
        ax.set_xlabel(xlabel='Willingness to pay (€)', fontsize=20, weight='bold')
        ax.set_ylabel(ylabel='Probability of cost-effectiveness (%)', fontsize=20, weight='bold')
        ax.set_xlim(left=0, right=105000)
        ax.set_ylim(bottom=0)
        ax.axhline(color = 'k', linewidth=1)
        ax.axvline(color = 'k', linewidth=1)
        ax.grid(visible=True)
        fig.set_size_inches(*big_figure)
        fig.tight_layout()

    def best_acceptability_curves(self, number_of_top_strategies_to_plot):
        number_of_top_strategies_to_plot = min(number_of_top_strategies_to_plot, len(self.simulation_results.delta_qalys.keys()))
        title = f'Cost-effectiveness acceptability curves of best {number_of_top_strategies_to_plot} strategies'
        fig, ax = plt.subplots(num = title)

        screening_strategies = self.simulation_results.get_best_strategies(statistic = 'probability at willingness threshold')

        for screening_strategy in screening_strategies:
            genotyping = 'with HLA genotyping' if screening_strategy[1] else 'without HLA genotyping'
            screening_strategy_label = f'Screening age(s): {screening_strategy[0]}, {genotyping},\n'
            probability = f'{self.simulation_results.probabilities_at_willingness_threshold[screening_strategy]:.2f}%'
            cost_effectiveness_probability_label = f'probability of cost-effectiveness at €{self.willingness_to_pay_threshold}: {probability}'
            label = screening_strategy_label + cost_effectiveness_probability_label
            ax.plot(self.simulation_results.willingness_to_pay_vector, self.simulation_results.acceptability_curves[screening_strategy], label=label)

        ax.legend()
        ax.set(xlabel='Willingness to pay (€)', ylabel='Probability of cost-effectiveness (%)', title=title)
        ax.axhline(color = 'k', linewidth=1)
        ax.axvline(color = 'k', linewidth=1)
        fig.set_size_inches(*big_figure)

    def percentage_of_celiac_disease_diagnosed(self):
        title = 'Percentage of diagnosed celiac disease cases'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)
        number_of_years_to_plot = 17
        percentage_of_celiac_disease_diagnosed = self.simulation_results.population_level_statistics_without_screening.get(title)
        percentage_of_celiac_disease_diagnosed = percentage_of_celiac_disease_diagnosed[:, :number_of_years_to_plot] * 100

        self._plot_mean_with_confidence_intervals(ax, percentage_of_celiac_disease_diagnosed, title + ', target: 30.8% @ 12', 'Age (years)', 'Percentage of diagnosed cases (%)')

        age = 12
        mean_at_12 = np.mean(percentage_of_celiac_disease_diagnosed[:, age])
        ax.plot(age, mean_at_12, 'kx', markersize=10, markeredgewidth=2)
        ax.annotate(f'{mean_at_12:.2f}%', (age, mean_at_12), xycoords='data', textcoords="offset points", xytext=(-45, 0), ha='center')

        fig.set_size_inches(*medium_figure)

    def clinical_prevalence_per_1000_as_a_function_of_age(self):
        title = 'Clinical prevalence per 1000'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)
        number_of_years_to_plot = 12
        clinical_prevalence_per_1000 = self.simulation_results.population_level_statistics_without_screening.get(title) 
        clinical_prevalence_per_1000 = clinical_prevalence_per_1000[:, :number_of_years_to_plot]

        self._plot_mean_with_confidence_intervals(ax, clinical_prevalence_per_1000, title + ', target: 2.9 @ 6', 'Age (years)', 'Cases per 1000')

        age = 6
        mean_clinical_prevalence_per_1000_at_6 = np.mean(clinical_prevalence_per_1000[:, age])
        ax.plot(age, mean_clinical_prevalence_per_1000_at_6, 'kx', markersize=10, markeredgewidth=2)
        ax.annotate(f'{mean_clinical_prevalence_per_1000_at_6:.2f}', (age, mean_clinical_prevalence_per_1000_at_6), xycoords='data', textcoords="offset points", xytext=(-30, 0), ha='center')

        fig.set_size_inches(*medium_figure)

    # Function that plots the cumulative risk of developing celiac disease
    def prevalence_and_cumulative_risk(self, transition_probabilities):
        title1 = f'Prevalence and cumulative risk of celiac disease, ages 0-{settings.number_of_years_to_simulate}'
        fig1, (ax1, ax2) = plt.subplots(1, 2, num = title1, constrained_layout=True)

        risk_of_celiac_disease = RiskOfCeliacDisease(transition_probabilities)
        prevalence_over_time_without_screening = self.simulation_results.population_with_celiac_disease_over_time_without_screening * 100 / self.simulation_results.alive_population_over_time_without_screening

        self._plot_mean_with_confidence_intervals(ax1, prevalence_over_time_without_screening, '', 'Age (years)', 'Prevalence of celiac disease (%)')
        cumulative_risk = [risk_of_celiac_disease.get_scaled_cumulative_risk(age) for age in range(settings.number_of_years_to_simulate)]
        ax2.plot(cumulative_risk)
        ax2.set(xlabel='Age (years)', ylabel='Cumulative risk of celiac disease (%)', title='')
        ax2.grid()

        fig1.suptitle(title1)
        fig1.set_size_inches(*medium_figure)

        number_of_years_to_plot = 25
        title2 = f'Prevalence and cumulative risk of celiac disease, ages 0-{number_of_years_to_plot}'
        fig2, (ax3, ax4) = plt.subplots(1, 2, num = title2, constrained_layout=True)
        
        self._plot_mean_with_confidence_intervals(ax3, prevalence_over_time_without_screening[:number_of_years_to_plot], '', 'Age (years)', 'Prevalence of celiac disease (%)')
        ax4.plot(cumulative_risk[:number_of_years_to_plot])
        ax4.set(xlabel='Age (years)', ylabel='Cumulative risk of celiac disease (%)', title='')
        ax4.grid()
        fig2.suptitle(title2)
        fig2.set_size_inches(*medium_figure)

    def celiac_disease_population_over_time(self):
        title = 'Undiagnosed and diagnosed population over time'
        fig, (ax1, ax2) = plt.subplots(1, 2, num = title, constrained_layout=True)
        undiagnosed_population = np.copy(self.simulation_results.undiagnosed_population_over_time_without_screening)
        undiagnosed_population[undiagnosed_population == 0] = np.nan
        celiac_disease_population = np.copy(self.simulation_results.population_with_celiac_disease_over_time_without_screening)
        celiac_disease_population[celiac_disease_population == 0] = np.nan

        percentage_of_undiagnosed_celiac_disease = 100 * undiagnosed_population / celiac_disease_population

        self._plot_mean_with_confidence_intervals(ax1, percentage_of_undiagnosed_celiac_disease, '', 'Age (years)', 'Percentage of undiagnosed celiac disease (%)', 'Mean percentage of undiagnosed (%)')
        self._plot_mean_with_confidence_intervals(ax2, 100 * undiagnosed_population, '', 'Age (years)', 'Percentage of population (%)', 'Mean undiagnosed population (%)')
        self._plot_mean_with_confidence_intervals(ax2, 100 * self.simulation_results.diagnosed_population_over_time_without_screening, '', 'Age (years)', 'Percentage of population (%)', 'Mean diagnosed population (%)')
        fig.suptitle(title)
        fig.set_size_inches(*medium_figure)

    def percentage_of_population_with_cd_diagnosis(self):
        title = 'Percentage of population that has received a celiac disease diagnosis'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)
        proportion_of_diagosed_population_over_time = 100 * self.simulation_results.diagnosed_population_over_time_without_screening / self.simulation_results.alive_population_over_time_without_screening
        
        self._plot_mean_with_confidence_intervals(ax, proportion_of_diagosed_population_over_time, title, 'Age (years)', 'Percentage of population (%)')
        fig.set_size_inches(*medium_figure)

    def percentage_of_asymptomatic_cd_without_diagnosis(self):
        title = 'Percentage of asymptomatic CeD without diagnosis (%)'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)

        proportion_of_asymptomatic_undiagnosed_population_over_time = np.array([
            [elem[1] for elem in population_distribution_over_time]
            for population_distribution_over_time in self.simulation_results.population_distribution_over_time_without_screening
        ])
        undiagnosed_population = np.copy(self.simulation_results.undiagnosed_population_over_time_without_screening)
        undiagnosed_population[undiagnosed_population == 0] = np.nan
        proportion_of_asymptomatic_cd_without_diagnosis_over_time = 100 * proportion_of_asymptomatic_undiagnosed_population_over_time / undiagnosed_population
        
        self._plot_mean_with_confidence_intervals(ax, proportion_of_asymptomatic_cd_without_diagnosis_over_time, title, 'Age (years)', 'Percentage of asymptomatic population (%)')
        fig.set_size_inches(*medium_figure)

    def annual_percentage_of_undiagnosed_cd_diagnosed(self):
        title = 'Percentage of undiagnosed celiac disease diagnosed annually'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)
        # Take the difference between subsequent years to get the number of new diagnoses
        number_of_new_diagnoses = np.diff(self.simulation_results.diagnosed_population_over_time_without_screening, axis=1)
        # Divide by the number of undiagnosed celiac disease to get the percentage diagnosed annually
        undiagnosed_population = np.copy(self.simulation_results.undiagnosed_population_over_time_without_screening)
        undiagnosed_population[undiagnosed_population == 0] = np.nan
        percentage_diagnosed_annually = 100 * number_of_new_diagnoses / undiagnosed_population[:, :-1]
        number_of_years_to_plot = 60
        percentage_diagnosed_annually = percentage_diagnosed_annually[:, :number_of_years_to_plot]

        self._plot_mean_with_confidence_intervals(ax, percentage_diagnosed_annually, title, 'Age (years)', 'Percentage of undiagnosed celiac disease diagnosed annually (%)')
        fig.set_size_inches(*medium_figure)

    def prevalence(self, number_of_years_to_plot = 16):
        title = f'Prevalence of celiac disease'
        fig, ax = plt.subplots(1, 1, num = title, constrained_layout=True)

        prevalence_over_time_without_screening = self.simulation_results.population_with_celiac_disease_over_time_without_screening * 100 / self.simulation_results.alive_population_over_time_without_screening
        prevalence_over_time_without_screening = prevalence_over_time_without_screening[:, :number_of_years_to_plot]

        self._plot_mean_with_confidence_intervals(ax, prevalence_over_time_without_screening, title + ', target: 2.2% @ 12', 'Age (years)', 'Prevalence (%)')

        age = 12
        mean_at_12 = np.mean(prevalence_over_time_without_screening[:, 12])
        ax.plot(age, mean_at_12, 'kx', markersize=10, markeredgewidth=2)
        ax.annotate(f'{mean_at_12:.2f}%', (age, mean_at_12), xycoords='data', textcoords="offset points", xytext=(-45, 0), ha='center')

        fig.set_size_inches(*medium_figure)