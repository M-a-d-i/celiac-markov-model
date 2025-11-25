import re

import matplotlib.pylab as plt
import numpy as np
from matplotlib.patches import Rectangle

from plotting import big_figure, huge_figure, medium_figure, wide_figure
from settings import settings
from utilities import utils


class BoxPlots:
    def __init__(self, simulation_results, willingness_to_pay_threshold):
        self.simulation_results = simulation_results
        self.willingness_to_pay_threshold = willingness_to_pay_threshold

    def _get_repeated_screening_labels(self, screening_strategies):
        labels = [re.sub('[(),]', '', f'{elem[0]}') for elem in screening_strategies]
        labels = [label.replace(' ', ',') for label in labels]
        return labels

    def box_plot_best_cost_effectiveness_probabilities_at_threshold(self, number_of_top_strategies_to_plot):
        number_of_top_strategies_to_plot = min(number_of_top_strategies_to_plot, len(self.simulation_results.delta_qalys.keys()))
        title = f'Probability of cost-effectiveness at €{self.willingness_to_pay_threshold} willingness to pay'
        fig, (ax1, ax2) = plt.subplots(2, 1, num=title, constrained_layout=True)

        screening_strategies = self.simulation_results.get_best_strategies(
            statistic = 'probability at willingness threshold'
        )

        cost_effectiveness_probability_boostrap_samples = np.array([self.simulation_results.probability_of_cost_effectiveness_at_threshold_bootstrap_samples[screening_strategy] for screening_strategy in screening_strategies]).T

        bplot1 = ax1.boxplot(cost_effectiveness_probability_boostrap_samples, vert=True, patch_artist=True)
        
        ax1.set_xticklabels([])
        ax1.set_ylabel('Probability of cost-effectiveness at €' + str(self.willingness_to_pay_threshold) + ' (%)')
        labels = list(settings.plot_colors.keys())
        handles = [Rectangle((0,0),1,1, color=settings.plot_colors[label]) for label in labels]
        ax1.legend(handles, labels)

        bplot2 = ax2.boxplot(cost_effectiveness_probability_boostrap_samples[:, :number_of_top_strategies_to_plot], vert=True, patch_artist=True)

        ax2.legend(handles, labels)
        ax2.grid(axis='y')
        labels = self._get_repeated_screening_labels(screening_strategies[:number_of_top_strategies_to_plot])
        ax2.set_xticks(ticks=np.arange(1, number_of_top_strategies_to_plot+1), labels=labels)
        ax2.set_ylabel('Probability of cost-effectiveness at €' + str(self.willingness_to_pay_threshold) + ' (%)')
        ax2.set_xlabel('Screening age(s) (years)')

        colors = utils.get_colors(screening_strategies)
        for bplot in (bplot1, bplot2):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)

        fig.set_size_inches(*wide_figure)

    def box_plot_best_icers(self, number_of_top_strategies_to_plot):
        number_of_top_strategies_to_plot = min(number_of_top_strategies_to_plot, len(self.simulation_results.delta_qalys.keys()))
        title = f'Incremental cost-effectiveness of best {number_of_top_strategies_to_plot} strategies'
        fig, (ax1, ax2) = plt.subplots(2, 1, num=title, constrained_layout=True)
        all_strategies_sorted_by_icer = self.simulation_results.get_best_strategies(statistic = 'icer')

        icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in all_strategies_sorted_by_icer]).T


        bplot1 = ax1.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)
        ax1.set_xticklabels([])
        ax1.set_ylabel('Incremental cost-effectiveness ratio (€/QALY)')
        #ax1.set_title(f'All screening strategies sorted from best to worst, range of screening ages: {screening_age_range}')
        labels = list(settings.plot_colors.keys())
        handles = [Rectangle((0,0),1,1, color=settings.plot_colors[label]) for label in labels]
        ax1.legend(handles, labels)

        bplot2 = ax2.boxplot(icer_bootrstrap_samples[:, :number_of_top_strategies_to_plot], vert=True, patch_artist=True)

        best_n_strategies = all_strategies_sorted_by_icer[:number_of_top_strategies_to_plot]
        for strategy, median_2d_line_object in zip(best_n_strategies, bplot2['medians']):
            x, y = median_2d_line_object.get_xdata()[0], max(median_2d_line_object.get_ydata())
            ax2.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

        ax2.legend(handles, labels)
        ax2.grid(axis='y')
        labels = self._get_repeated_screening_labels(best_n_strategies[:number_of_top_strategies_to_plot])
        ax2.set_xticks(ticks=np.arange(1, number_of_top_strategies_to_plot+1), labels=labels)
        ax2.set_ylabel('Incremental cost-effectiveness ratio (€/QALY)')
        ax2.set_xlabel('Screening age(s) (years)')

        colors = utils.get_colors(all_strategies_sorted_by_icer)
        for bplot in (bplot1, bplot2):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)

        fig.set_size_inches(*wide_figure)

    def box_plot_screening_age_against_icers(self):
        title = f'Incremental cost-effectiveness of single-time screening without HLA genotyping'
        fig, ax = plt.subplots(1, 1, num=title, constrained_layout=True)
        all_strategies = self.simulation_results.icer_expected_value.keys()
        # Select only strategies that have a single screening age and don't use HLA genotyping
        strategies = [strategy for strategy in all_strategies if len(strategy[0]) == 1 and not strategy[1]]
        # Sort strategies by the age of screening
        strategies = sorted(strategies, key=lambda x: x[0][0], reverse=False)
        icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in strategies]).T

        colors = utils.get_colors(strategies)
        labels = ['Single-time antibody screening']
        handles = [Rectangle((0,0),1,1, color=settings.plot_colors[label]) for label in labels]
        ax.legend(handles, labels)

        bplot = ax.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)

        for strategy, median_2d_line_object in zip(strategies, bplot['medians']):
            x, y = median_2d_line_object.get_xdata()[0], max(median_2d_line_object.get_ydata())
            ax.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

        ax.legend(handles, labels)
        ax.grid(axis='y')
        labels = [f'{elem[0][0]}' for elem in strategies]
        ax.set_xticks(ticks=np.arange(1, len(strategies)+1), labels=labels)
        ax.set_ylabel('Incremental cost-effectiveness ratio (€/QALY)')
        ax.set_xlabel('Screening age (years)')

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_color(color)

        fig.set_size_inches(*big_figure)

    def box_plot_screening_age_against_icer_with_best_repeated_screening_strategies(self):
        title = f'Incremental cost-effectiveness of single-time screening without HLA genotyping and best repeated screening strategies'
        fig, ax = plt.subplots(1, 1, num=title, constrained_layout=True)
        all_strategies = self.simulation_results.icer_expected_value.keys()
        # Select only strategies that have a single screening age and don't use HLA genotyping
        strategies = [strategy for strategy in all_strategies if len(strategy[0]) == 1 and not strategy[1]]
        # Sort strategies by the age of screening
        strategies = sorted(strategies, key=lambda x: x[0][0], reverse=False)
        icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in strategies]).T

        colors = utils.get_colors(strategies)
        labels = ['Single-time antibody screening', 'Repeated antibody screening']
        handles = [Rectangle((0,0),1,1, color=settings.plot_colors[label]) for label in labels]
        ax.legend(handles, labels)

        bplot = ax.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)

        for strategy, median_2d_line_object in zip(strategies, bplot['medians']):
            x, y = median_2d_line_object.get_xdata()[0], max(median_2d_line_object.get_ydata())
            ax.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

        ax.legend(handles, labels)
        ax.grid(axis='y')
        labels = [f'{elem[0][0]}' for elem in strategies]
        ax.set_xticks(ticks=np.arange(1, len(strategies)+1), labels=labels)
        ax.set_ylabel('Incremental cost-effectiveness ratio (€/QALY)')
        ax.set_xlabel('Single-time screening age (years)')

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_color(color)

        number_of_top_strategies_to_plot = len(strategies)
        strategies_from_best_to_worst = self.simulation_results.get_best_strategies(statistic = 'icer')
        # Get the best repeated screening strategies
        best_repeated_screening_strategies = [strategy for strategy in strategies_from_best_to_worst if len(strategy[0]) == 2][:number_of_top_strategies_to_plot]
        icer_bootstrap_samples_of_repeated_strategies = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in best_repeated_screening_strategies]).T

        bplot = ax.boxplot(icer_bootstrap_samples_of_repeated_strategies, vert=True, patch_artist=True)

        ax.set_xticks(ticks=np.arange(1, len(strategies)+1), labels=labels)

        for strategy, median_2d_line_object in zip(best_repeated_screening_strategies, bplot['medians']):
            x, y = median_2d_line_object.get_xdata()[0], max(median_2d_line_object.get_ydata())
            ax.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

        secax = ax.secondary_xaxis('top')
        labels = self._get_repeated_screening_labels(best_repeated_screening_strategies[:number_of_top_strategies_to_plot])
        secax.set_xticks(ticks=np.arange(1, number_of_top_strategies_to_plot+1), labels=labels)
        secax.set_xlabel('Repeated screening ages (years)')

        colors = utils.get_colors(best_repeated_screening_strategies)
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_color(color)

        fig.set_size_inches(*big_figure)

    def box_plot_all_repeated_screening_icers(self):
        all_strategies = self.simulation_results.icer_expected_value.keys()
        repeated_strategies_without_genotyping = [strategy for strategy in all_strategies if len(strategy[0]) == 2 and strategy[1] == False]
        # Sort repeated strategies by the age of first screening
        repeated_strategies_without_genotyping = sorted(
            repeated_strategies_without_genotyping,
            key=lambda x: (x[0][0], x[0][1]),
            reverse=False
        )
        first_screening_age = repeated_strategies_without_genotyping[0][0][0]
        last_screening_age = repeated_strategies_without_genotyping[-1][0][1]
        screening_age_range = last_screening_age - first_screening_age
        # Make nested lists of repeated strategies with the same first screening age
        repeated_strategies_without_genotyping = [[strategy for strategy in repeated_strategies_without_genotyping if strategy[0][0] == age] for age in range(first_screening_age, last_screening_age)]
        title = f'All repeated screening strategies incremental cost-effectiveness'
        fig, axs = plt.subplots(nrows=screening_age_range, ncols=1, num=title, constrained_layout=True)

        # If axs is not an array, throw error
        if not isinstance(axs, plt.ndarray):
            raise ValueError("axs should be a numpy array of axes, but got: " + str(type(axs))) 

        for i in range(screening_age_range):
            ax = axs[i]
            strategies = repeated_strategies_without_genotyping[i]
            icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in strategies]).T

            bplot = ax.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)

            for strategy, median_2d_line_object in zip(strategies, bplot['medians']):
                x, y = median_2d_line_object.get_xdata()[0], median_2d_line_object.get_ydata().max()
                ax.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

            ax.grid(axis='y')
            labels = [f'{elem[0][1]}' for elem in strategies]
            ax.set_xticks(ticks=np.arange(1, len(strategies)+1), labels=labels)

            colors = utils.get_colors(strategies)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)

        fig.set_size_inches((16, 6*screening_age_range))

    def box_plot_all_repeated_screening_icers_ordered(self):
        title = f'Incremental cost-effectiveness of all repeated screening strategies'

        repeated_strategies_sorted_by_icer = self.simulation_results.get_best_strategies(
            statistic = 'icer'
        )
        # select only repeated strategies without genotyping
        repeated_strategies_sorted_by_icer = [strategy for strategy in repeated_strategies_sorted_by_icer if len(strategy[0]) == 2 and strategy[1] == False]

        nrows = 5
        fig, axs = plt.subplots(nrows=nrows, ncols=1, num=title, constrained_layout=True)

        for i in range(nrows):
            ax = axs[i]
            number_of_boxplots_per_row = len(repeated_strategies_sorted_by_icer) // nrows
            screening_strategies = repeated_strategies_sorted_by_icer[i*number_of_boxplots_per_row:(i+1)*number_of_boxplots_per_row]
            icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in screening_strategies]).T

            bplot = ax.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)

            labels = self._get_repeated_screening_labels(screening_strategies)
            ax.set_xticks(ticks=np.arange(1, len(screening_strategies)+1), labels=labels)
            ax.grid(axis='y')

            colors = utils.get_colors(screening_strategies)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)

        fig.supylabel('Incremental cost-effectiveness ratio (€/QALY)')
        fig.supxlabel('Screening ages (years)')
        fig.set_size_inches( *(wide_figure * [1, 1.5]) )

    def box_plot_all_repeated_screening_icers_by_time_between_screenings(self):
        all_strategies = self.simulation_results.icer_expected_value.keys()
        repeated_strategies_without_genotyping = [strategy for strategy in all_strategies if len(strategy[0]) == 2 and strategy[1] == False]
        # Sort repeated strategies by the age of first screening
        repeated_strategies_without_genotyping = sorted(
            repeated_strategies_without_genotyping,
            key=lambda x: (x[0][1] - x[0][0], x[0][0]),
            reverse=False
        )
        first_screening_age = repeated_strategies_without_genotyping[0][0][0]
        last_screening_age = repeated_strategies_without_genotyping[-1][0][1]
        screening_age_range = last_screening_age - first_screening_age
        # Make nested lists of repeated strategies with the same first screening age
        repeated_strategies_without_genotyping = [[strategy for strategy in repeated_strategies_without_genotyping if strategy[0][1] - strategy[0][0] == diff] for diff in range(1, screening_age_range + 1)]
        title = f'Repeated screening strategies incremental cost-effectiveness, ordered by time between screenings'
        fig, axs = plt.subplots(nrows=screening_age_range, ncols=1, num=title, constrained_layout=True)

        # If axs is not an array, throw error
        if not isinstance(axs, plt.ndarray):
            raise ValueError("axs should be a numpy array of axes, but got: " + str(type(axs))) 

        for i in range(screening_age_range):
            ax = axs[i]
            strategies = repeated_strategies_without_genotyping[i]
            icer_bootrstrap_samples = np.array([self.simulation_results.icer_bootstrap_samples[screening_strategy] for screening_strategy in strategies]).T

            bplot = ax.boxplot(icer_bootrstrap_samples, vert=True, patch_artist=True)

            for strategy, median_2d_line_object in zip(strategies, bplot['medians']):
                x, y = median_2d_line_object.get_xdata()[0], median_2d_line_object.get_ydata().max()
                ax.text(x, y, f'{self.simulation_results.icer_quadrant[strategy]}', verticalalignment='bottom', horizontalalignment='left', fontsize=10, color='white')

            ax.grid(axis='y')
            labels = self._get_repeated_screening_labels(strategies)
            ax.set_xticks(ticks=np.arange(1, len(strategies)+1), labels=labels)

            colors = utils.get_colors(strategies)
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)

        fig.set_size_inches((16, 6*screening_age_range))