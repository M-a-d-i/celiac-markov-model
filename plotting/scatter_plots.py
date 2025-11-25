import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from plotting import big_figure, huge_figure, medium_figure, wide_figure
from settings import settings
from utilities import utils


class ScatterPlots:
    def __init__(self, simulation_results):
        self.simulation_results = simulation_results

    def _draw_axis_lines(self, ax):
        ax.axhline(color='k', linewidth=1)
        ax.axvline(color='k', linewidth=1)

    def scatter_all_screening_strategies_costs_and_qalys(self):
        title = 'Expected differences on the cost-effectiveness plane'
        fig, ax = plt.subplots(num=title)

        screening_strategies = self.simulation_results.delta_qalys_expected_value.keys()

        # Collect data points
        data_points = {}
        for screening_strategy in screening_strategies:
            delta_qaly = self.simulation_results.delta_qalys_expected_value[screening_strategy]
            delta_cost = self.simulation_results.delta_costs_expected_value[screening_strategy]
            color = utils.get_color(screening_strategy)
            ax.scatter(delta_qaly, delta_cost, s=80, color=color)
            data_points[screening_strategy] = (delta_qaly, delta_cost)

        self._draw_axis_lines(ax)

        if all(cost > 0 for _, cost in data_points.values()) and any(qaly > 0 for qaly, _ in data_points.values()):
            # Sort data points by QALYs
            sorted_data_points = sorted(data_points.items(), key=lambda x: x[1][0], reverse=True)
            # Identify the cost-effectiveness frontier
            frontier_points = [(0, 0)]  # Start from the origin
            frontier_colors = ['black']  # List to store colors of frontier points
            while True:
                last_point = frontier_points[-1]
                # Filter out all data points where QALYs are less than the last point on the frontier
                filtered_points = [(strategy, (qaly, cost)) for strategy, (qaly, cost) in sorted_data_points if qaly > last_point[0]]
                # Calculate slopes from last_point to all remaining points
                slopes = [(cost - last_point[1]) / (qaly - last_point[0]) for _, (qaly, cost) in filtered_points]
                # Filter out negative slopes
                filtered_slopes = [slope for slope in slopes if slope >= 0]
                if len(filtered_slopes) == 0:
                    break
                min_slope_index = filtered_slopes.index(min(filtered_slopes))
                frontier_points.append(filtered_points[min_slope_index][1])
                # Add the corresponding color to the frontier colors list
                frontier_colors.append(utils.get_color(filtered_points[min_slope_index][0]))
                # Annotate the screening strategy
                screening_ages = filtered_points[min_slope_index][0][0]
                if len(screening_ages) == 1:
                    screening_ages = f'({screening_ages[0]})'
                xy = filtered_points[min_slope_index][1]
                ax.annotate(text=screening_ages, xy=xy, textcoords="offset points", xytext=(0,10), ha='center')
            # Plot the cost-effectiveness frontier lines
            frontier_qalys, frontier_costs = zip(*frontier_points)
            ax.plot(frontier_qalys, frontier_costs, linestyle='--', color='black', linewidth=2)
            # Plot the frontier points with a different marker and original color, exclude origin
            for (qaly, cost), color in zip(frontier_points[1:], frontier_colors[1:]):
                ax.scatter(qaly, cost, s=100, color=color, marker='D')
            # Annotate the slopes as ICER
            for i in range(1, len(frontier_points)):
                x1, y1 = frontier_points[i-1]
                x2, y2 = frontier_points[i]
                icer = (y2 - y1) / (x2 - x1)
                # Calculate the rotation angle
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                ax.text((x1 + x2) / 2, (y1 + y2) / 2 - 1, f'{icer:.2f} €/QALY', ha='center', va='top', rotation=angle, transform_rotates_text=True, rotation_mode='anchor')

        colors = utils.get_colors(screening_strategies)
        colors_for_legends = set(colors)
        color_to_label_mapping = {value: key for key, value in settings.plot_colors.items()}
        preferred_legend_order = [
            'Single-time antibody screening',
            'First-line genetic + single-time antibody screening',
            'Repeated antibody screening',
            'First-line genetic + repeated antibody screening',
        ]
        legend_colors = {color_to_label_mapping[color]: color for color in colors_for_legends}
        ordered_labels = [label for label in preferred_legend_order if label in legend_colors]
        ordered_handles = [Rectangle((0, 0), 1, 1, color=legend_colors[label]) for label in ordered_labels]
        ax.legend(ordered_handles, ordered_labels, loc='lower left')

        ax.set_xlabel('$\\Delta$ QALYs', fontsize=20, weight='bold')
        ax.set_ylabel('$\\Delta$ Costs (€)', fontsize=20, weight='bold')
        ax.set_ylim(bottom=-40)
        ax.set_xlim(left=-0.00025)
        ax.grid(True)
        fig.set_size_inches(*big_figure)
        fig.tight_layout()

    def scatter_single_screening_strategy_costs_and_qalys(self, screening_strategy):
        title = 'Scatter plot of costs and QALYs of a single screening strategy'
        fig, ax = plt.subplots(num=title)
        ax.set(xlabel='Differences in lifetime QALY', ylabel='Differences in lifetime costs (€)', title=title)
        ax.scatter(self.simulation_results.delta_qalys[screening_strategy], self.simulation_results.delta_costs[screening_strategy], s=10)
        self._draw_axis_lines(ax)
        fig.set_size_inches(*medium_figure)
        fig.tight_layout()

    def scatter_best_cost_effectiveness_probability_and_expected_value_costs_and_qalys(self):
        title = 'Scatter plot of best cost-effectiveness probability and expected value of costs and QALYs'
        fig, ax = plt.subplots(num=title)
        ax.set(xlabel='Differences in lifetime QALY', ylabel='Differences in lifetime costs (€)', title=title)

        best_expected_value = self.simulation_results.get_best_strategies(statistic='icer')
        best_expected_value_costs = self.simulation_results.delta_costs[best_expected_value[0]]
        best_expected_value_qalys = self.simulation_results.delta_qalys[best_expected_value[0]]
        best_expected_value_strategy = f'Screening age(s): {best_expected_value[0][0]}, ' + ('with HLA genotyping' if best_expected_value[0][1] else 'no HLA genotyping') + '\n'
        expected_icer = f'Expected ICER: {self.simulation_results.icer_expected_value[best_expected_value[0]]:.2f} €/QALY\n'
        probability_of_cost_effectiveness = f'Probability of cost-effectiveness: {self.simulation_results.probabilities_at_willingness_threshold[best_expected_value[0]]:.2f}%'
        label = f'Strategy with best expected ICER: ' + best_expected_value_strategy + expected_icer + probability_of_cost_effectiveness
        ax.scatter(best_expected_value_qalys, best_expected_value_costs, s=10, label=label)

        best_cost_effectiveness_probability = self.simulation_results.get_best_strategies(statistic='probability at willingness threshold')
        best_cost_effectiveness_probability_costs = self.simulation_results.delta_costs[best_cost_effectiveness_probability[0]]
        best_cost_effectiveness_probability_qalys = self.simulation_results.delta_qalys[best_cost_effectiveness_probability[0]]
        best_cost_effectiveness_probability_strategy = f'Screening age(s): {best_cost_effectiveness_probability[0][0]}, ' + ('with HLA genotyping' if best_cost_effectiveness_probability[0][1] else 'no HLA genotyping') + '\n'
        expected_icer = f'Expected ICER: {self.simulation_results.icer_expected_value[best_cost_effectiveness_probability[0]]:.2f} €/QALY\n'
        probability_of_cost_effectiveness = f'Probability of cost-effectiveness: {self.simulation_results.probabilities_at_willingness_threshold[best_cost_effectiveness_probability[0]]:.2f}%'
        label = f'Strategy with best probability of cost-effectiveness: ' + best_cost_effectiveness_probability_strategy + expected_icer + probability_of_cost_effectiveness
        ax.scatter(best_cost_effectiveness_probability_qalys, best_cost_effectiveness_probability_costs, s=10, label=label)

        ax.legend()
        self._draw_axis_lines(ax)
        fig.set_size_inches(*big_figure)
        fig.tight_layout()

        # Make separate pandas dataframes out of best_expected_value and best_cost_effectiveness_probability
        df1 = pd.DataFrame({'Strategy': [f'{best_expected_value_strategy}'] * len(best_expected_value_costs),
                            'Differences in lifetime QALY': best_expected_value_qalys,
                            'Differences in lifetime costs (€)': best_expected_value_costs})
        df2 = pd.DataFrame({'Strategy': [f'{best_cost_effectiveness_probability_strategy}'] * len(best_cost_effectiveness_probability_costs),
                            'Differences in lifetime QALY': best_cost_effectiveness_probability_qalys,
                            'Differences in lifetime costs (€)': best_cost_effectiveness_probability_costs})
        df = pd.concat([df1, df2])

        # Use sns.jointplot to plot the same data
        joint_grid = sns.jointplot(data=df, x='Differences in lifetime QALY', y='Differences in lifetime costs (€)', hue='Strategy')
        # Draw x and y axis lines
        self._draw_axis_lines(joint_grid.ax_joint)
        # Increase figure size
        plt.gcf().set_size_inches(*big_figure)