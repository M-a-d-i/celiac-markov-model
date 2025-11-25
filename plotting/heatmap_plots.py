import matplotlib.cm
import matplotlib.pylab as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle


class HeatmapPlots:
    def __init__(self, simulation_results):
        self.simulation_results = simulation_results
        self.all_screening_strategies = list(self.simulation_results.inmb_expected_value.keys())
        # Separate strategies with and without genotyping
        self.single_screening_without_genotyping = [x for x in self.all_screening_strategies if len(x[0]) == 1 and x[1] == False]
        self.repeated_screening_without_genotyping = [x for x in self.all_screening_strategies if len(x[0]) == 2 and x[1] == False]
        self.single_screening_with_genotyping = [x for x in self.all_screening_strategies if len(x[0]) == 1 and x[1] == True]
        self.repeated_screening_with_genotyping = [x for x in self.all_screening_strategies if len(x[0]) == 2 and x[1] == True]
        # Sort repeated strategies in order of screening ages
        self.repeated_screening_without_genotyping = sorted(self.repeated_screening_without_genotyping, key=lambda x: (x[0][0], x[0][1]))
        self.repeated_screening_with_genotyping = sorted(self.repeated_screening_with_genotyping, key=lambda x: (x[0][0], x[0][1]))
        self.first_screening_age = np.min([x[0][0] for x in self.repeated_screening_without_genotyping])
        self.last_screening_age = np.max([x[0][1] for x in self.repeated_screening_without_genotyping])
        self.matrix_size = self.last_screening_age - self.first_screening_age

        # Figure properties
        self.figsize = (20, 25)
        self.title_fontsize = 28
        self.label_fontsize = 24
        self.tick_fontsize = 22
        self.icer_fontsize = 20
        self.confidence_interval_fontsize = 13
        self.antibody_screening_color = '#dae8fc'
        self.genetic_plus_antibody_screening_color = '#b8c4d6'
        self.shared_color_map = 'RdYlBu'
        self.confidence_interval_alpha = 0.85
        self.outlier_alpha = 0.30

    def plot_cost_effectiveness_heatmap(self, shared_colorbar: bool, use_transparency: bool, metric):
        if metric == 'icer':
            self.all_means = {k: v / 1000 for k, v in self.simulation_results.icer_expected_value.items()}
            self.all_lower_bounds = {k: v / 1000 for k, v in self.simulation_results.icer_lower_bound.items()}
            self.all_upper_bounds = {k: v / 1000 for k, v in self.simulation_results.icer_upper_bound.items()}
            self.color_map = 'RdYlBu_r'
            self.color_bar_label = 'Incremental cost-effectiveness ratio (€1,000/QALY)'
        elif metric == 'inmb':
            self.all_means = self.simulation_results.inmb_expected_value
            self.all_lower_bounds = self.simulation_results.inmb_lower_bound
            self.all_upper_bounds = self.simulation_results.inmb_upper_bound
            self.color_bar_label = 'Incremental net monetary benefit (€)'
            self.color_map = 'RdYlBu'
        else:
            raise ValueError("Invalid metric. Choose 'icer' or 'inmb'.")

        # Calculate normalization for heatmaps, strategies without genotyping
        self.icer_values_without_genotyping = [self.all_means[x] for x in self.single_screening_without_genotyping + self.repeated_screening_without_genotyping]
        self.vmin_without_genotyping = float(np.percentile(self.icer_values_without_genotyping, 2.5))
        self.vmax_without_genotyping = float(np.percentile(self.icer_values_without_genotyping, 95))
        self.norm_without_genotyping = Normalize(vmin=self.vmin_without_genotyping, vmax=self.vmax_without_genotyping)
        # Calculate normalization for heatmaps, strategies with genotyping
        self.icer_values_with_genotyping = [self.all_means[x] for x in self.single_screening_with_genotyping + self.repeated_screening_with_genotyping]
        self.vmin_with_genotyping = float(np.percentile(self.icer_values_with_genotyping, 2.5))
        self.vmax_with_genotyping = float(np.percentile(self.icer_values_with_genotyping, 95))
        self.norm_with_genotyping = Normalize(vmin=self.vmin_with_genotyping, vmax=self.vmax_with_genotyping)

        if shared_colorbar:
            self._cost_effectiveness_heatmap_with_shared_colorbar(use_transparency, metric)
        else:
            self._cost_effectiveness_heatmap_without_shared_colorbar(use_transparency, metric)
    
    def _cost_effectiveness_heatmap_without_shared_colorbar(self, use_transparency, metric):
        title = metric + ' heatmap, individual colorbars'
        if use_transparency:
            title += ', alpha adjusted'
        fig = plt.figure(num=title, constrained_layout=True, figsize=self.figsize)
        subfigs = fig.subfigures(2, 1, height_ratios=[1, 1])

        # Set different background colors for each subfigure
        subfigs[0].set_facecolor(self.antibody_screening_color)
        subfigs[1].set_facecolor(self.genetic_plus_antibody_screening_color)

        # Create subplots within each subfigure
        axs1 = subfigs[0].subplots(2, 1, gridspec_kw={'height_ratios': [1, 12], 'hspace': 0.025})
        axs2 = subfigs[1].subplots(2, 1, gridspec_kw={'height_ratios': [1, 12], 'hspace': 0.025})
        ax1, ax2 = axs1[0], axs1[1]
        ax3, ax4 = axs2[0], axs2[1]

        self._add_horizontal_lines(ax1, ax3)
        self._plot_heatmaps_and_single_screening(ax1, ax2, ax3, ax4, self.norm_without_genotyping, self.norm_with_genotyping, use_transparency)
        self._add_colorbars(subfigs[0], [ax1, ax2], self.norm_without_genotyping)
        self._add_colorbars(subfigs[1], [ax3, ax4], self.norm_with_genotyping)
        self._add_titles(ax1, ax3)

    def _cost_effectiveness_heatmap_with_shared_colorbar(self, use_transparency, metric):
        title = metric + ' heatmap, shared colorbar'
        if use_transparency:
            title += ', alpha adjusted'
        fig = plt.figure(num=title, figsize=self.figsize)

        gs1 = gridspec.GridSpec(2, 1, height_ratios=[1, 12], top=0.975, bottom=0.535, left=0.05, right=0.94, hspace=0.2)
        ax1 = fig.add_subplot(gs1[0])
        ax2 = fig.add_subplot(gs1[1])

        gs2 = gridspec.GridSpec(2, 1, height_ratios=[1, 12], top=0.465, bottom=0.035, left=0.05, right=0.94, hspace=0.2)
        ax3 = fig.add_subplot(gs2[0])
        ax4 = fig.add_subplot(gs2[1])

        # Unified normalization for shared colorbar
        vmin = min(float(self.vmin_without_genotyping), float(self.vmin_with_genotyping))
        vmax = max(float(self.vmax_without_genotyping), float(self.vmax_with_genotyping))
        norm = Normalize(vmin=vmin, vmax=vmax)

        self._draw_rectangles(fig)
        self._add_horizontal_lines(ax1, ax3)
        self._plot_heatmaps_and_single_screening(ax1, ax2, ax3, ax4, norm, norm, use_transparency)
        self._add_shared_colorbar(fig, norm)
        self._add_titles(ax1, ax3)

    def _add_horizontal_lines(self, ax1, ax3):
        ax1.axhline(y=1.60, color='black', linewidth=2, linestyle='--', clip_on=False)
        ax3.axhline(y=1.60, color='black', linewidth=2, linestyle='--', clip_on=False)

    def _plot_heatmaps_and_single_screening(self, ax1, ax2, ax3, ax4, norm_without_genotyping, norm_with_genotyping, use_transparency):
        self._plot_single_screening(ax1, self.single_screening_without_genotyping, 'Single-time screening age (years)', norm_without_genotyping, use_transparency)
        self._plot_heatmap(ax2, self.repeated_screening_without_genotyping, 'Repeated screening, first age (years)', 'Repeated screening, second age (years)', norm_without_genotyping, use_transparency)
        self._plot_single_screening(ax3, self.single_screening_with_genotyping, 'Single-time screening age (years)', norm_with_genotyping, use_transparency)
        self._plot_heatmap(ax4, self.repeated_screening_with_genotyping, 'Repeated screening, first age (years)', 'Repeated screening, second age (years)', norm_with_genotyping, use_transparency)

        # Ensure both heatmaps use the same size for the boxes
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_aspect(aspect='auto', adjustable='box')

    def _add_colorbars(self, subfig, axes, norm):
        color_bar = subfig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.shared_color_map), ax=axes, orientation='vertical', pad=0.01)
        color_bar.set_label(self.color_bar_label, weight='bold', fontsize=self.label_fontsize, labelpad=20)
        color_bar.ax.yaxis.set_label_position('left')
        color_bar.set_ticks([])
        color_bar.ax.text(-0.85, 0.95, 'Better', rotation=90, va='center', ha='center', fontsize=self.icer_fontsize, color='blue', transform=color_bar.ax.transAxes)
        color_bar.ax.text(-0.85, 0.05, 'Worse', rotation=90, va='center', ha='center', fontsize=self.icer_fontsize, color='red', transform=color_bar.ax.transAxes)

    def _add_shared_colorbar(self, fig, norm):
        cbar_ax = fig.add_axes([0.975, 0.035, 0.02, 0.94])  # Adjust the position and size of the colorbar
        color_bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=self.shared_color_map), cax=cbar_ax)
        color_bar.set_label(self.color_bar_label, weight='bold', fontsize=self.label_fontsize, labelpad=10)
        color_bar.ax.yaxis.set_label_position('left')
        color_bar.set_ticks([])
        color_bar.ax.text(-0.85, 0.98, 'Better', rotation=90, va='center', ha='center', fontsize=self.icer_fontsize, color='blue', transform=color_bar.ax.transAxes)
        color_bar.ax.text(-0.85, 0.02, 'Worse', rotation=90, va='center', ha='center', fontsize=self.icer_fontsize, color='red', transform=color_bar.ax.transAxes)

    def _add_titles(self, ax1, ax3):
        ax1.set_title('A) Antibody screening', fontsize=self.title_fontsize, pad=20, fontweight='bold')
        ax3.set_title('B) First-line genetic screening + antibody screening', fontsize=self.title_fontsize, pad=20, fontweight='bold')

    def _draw_rectangles(self, fig):
        fig.add_artist(Rectangle((0.0, 0.505), 0.94, 0.495, color=self.antibody_screening_color, zorder=-1))
        fig.add_artist(Rectangle((0.0, 0.000), 0.94, 0.495, color=self.genetic_plus_antibody_screening_color, zorder=-1))

    def _calculate_luminance(self, color):
        r, g, b = color[:3]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _add_text_to_heatmap(self, ax, j, i, icer_rounded_mean, icer_rounded_lower_bound, icer_rounded_upper_bound, background_color):
        luminance = self._calculate_luminance(background_color)
        text_color = 'white' if luminance < 0.5 else 'black'
        separation = 0.190
        ax.text(j, i-separation, f'{icer_rounded_mean}', ha='center', va='center', color=text_color, fontsize = self.icer_fontsize)
        confidence_interval_text = f'({icer_rounded_lower_bound},{icer_rounded_upper_bound})'
        ci_fontsize = self.confidence_interval_fontsize - max(0, len(confidence_interval_text) - 11)
        ax.text(j, i+separation, confidence_interval_text, ha='center', va='center', color=text_color, fontsize = ci_fontsize)

    def _plot_heatmap(self, ax, strategies, xlabel, ylabel, norm, use_transparency):
        icer_matrix = np.zeros((self.matrix_size, self.matrix_size))
        icer_confidence_interval_margin_matrix = np.zeros((self.matrix_size, self.matrix_size))
        icer_lower_bound_matrix = np.zeros((self.matrix_size, self.matrix_size))
        icer_upper_bound_matrix = np.zeros((self.matrix_size, self.matrix_size))
        for strategy in strategies:
            ((age1, age2), _) = strategy
            array_index = (age1 - self.first_screening_age, age2 - self.first_screening_age - 1)
            icer_matrix[array_index] = self.all_means[strategy]
            icer_lower_bound_matrix[array_index] = self.all_lower_bounds[strategy]
            icer_upper_bound_matrix[array_index] = self.all_upper_bounds[strategy]

        upper_mask = np.triu(np.ones_like(icer_matrix, dtype=bool), k=1).T

        masked_icer_matrix = np.ma.masked_where(upper_mask, icer_matrix)
        masked_icer_matrix = np.rot90(masked_icer_matrix, k=1)

        masked_icer_confidence_interval_margin_matrix = np.ma.masked_where(upper_mask, icer_confidence_interval_margin_matrix)
        masked_icer_confidence_interval_margin_matrix = np.rot90(masked_icer_confidence_interval_margin_matrix, k=1)

        masked_icer_lower_bound_matrix = np.ma.masked_where(upper_mask, icer_lower_bound_matrix)
        masked_icer_lower_bound_matrix = np.rot90(masked_icer_lower_bound_matrix, k=1)

        masked_icer_upper_bound_matrix = np.ma.masked_where(upper_mask, icer_upper_bound_matrix)
        masked_icer_upper_bound_matrix = np.rot90(masked_icer_upper_bound_matrix, k=1)

        # This currently only works with ICER, need to use a different approach for INMB
        alpha_matrix = np.full((self.matrix_size, self.matrix_size), 1.0)
        if use_transparency:
            alpha_matrix = np.full((self.matrix_size, self.matrix_size), self.outlier_alpha)
            best_strategy = np.unravel_index(np.argmin(masked_icer_matrix), masked_icer_matrix.shape)
            best_strategy_upper_bound = masked_icer_upper_bound_matrix[best_strategy]
            for i in range(self.matrix_size):
                for j in range(self.matrix_size):
                    if np.ma.getmaskarray(masked_icer_matrix)[i, j] == False:
                        if masked_icer_matrix[i, j] <= best_strategy_upper_bound:
                            alpha_matrix[i, j] = self.confidence_interval_alpha
            alpha_matrix[best_strategy] = 1.0

        im = ax.imshow(masked_icer_matrix, cmap=self.color_map, norm=norm, alpha=alpha_matrix)

        ax.set_xticks(ticks=np.arange(self.matrix_size), labels=[f'{x}' for x in np.arange(self.first_screening_age, self.last_screening_age)], fontsize=self.tick_fontsize)
        ax.set_yticks(ticks=np.arange(self.matrix_size), labels=[f'{x}' for x in np.arange(self.last_screening_age, self.first_screening_age, -1)], fontsize=self.tick_fontsize)

        ax.set_xlabel(xlabel, fontsize=self.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.label_fontsize)

        # Add text annotations for repeated screening strategies
        for i in range(self.matrix_size):
            for j in range(self.matrix_size):
                if np.ma.getmaskarray(masked_icer_matrix)[i, j] == False:  # Only add text where the mask is not applied
                    icer_rounded_mean = f'{masked_icer_matrix[i, j]:.1f}'
                    icer_rounded_lower_bound = f'{masked_icer_lower_bound_matrix[i, j]:.1f}'
                    icer_rounded_upper_bound = f'{masked_icer_upper_bound_matrix[i, j]:.1f}'
                    background_color = im.cmap(norm(masked_icer_matrix[i, j]))
                    self._add_text_to_heatmap(ax, j, i, icer_rounded_mean, icer_rounded_lower_bound, icer_rounded_upper_bound, background_color)

        # Add grid lines to separate the boxes
        ax.set_xticks(np.arange(-0.5, self.matrix_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.matrix_size, 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        return im

    def _plot_single_screening(self, ax, strategies, xlabel, norm, use_transparency):
        single_screening_ages = [x[0][0] for x in strategies]
        single_screening_ages_sorted = sorted(single_screening_ages)
        genotyping = strategies[0][1]
        single_icer_values = [self.all_means[(age,), genotyping] for age in single_screening_ages_sorted]
        icer_lower_bounds = [self.all_lower_bounds[(age,), genotyping] for age in single_screening_ages_sorted]
        icer_upper_bounds = [self.all_upper_bounds[(age,), genotyping] for age in single_screening_ages_sorted]

        alpha_values = np.full(len(single_icer_values), 1.0)
        if use_transparency:
            alpha_values = np.full(len(single_icer_values), self.outlier_alpha)
            best_strategy_index = np.argmin(single_icer_values)
            best_strategy_upper_bound = icer_upper_bounds[best_strategy_index]
            for i, value in enumerate(single_icer_values):
                if value <= best_strategy_upper_bound:
                    alpha_values[i] = self.confidence_interval_alpha
            alpha_values[best_strategy_index] = 1.0

        im = ax.imshow(np.array(single_icer_values).reshape(1, -1), cmap=self.color_map, aspect='auto', norm=norm, alpha=alpha_values.reshape(1, -1))

        ax.set_xticks(ticks=np.arange(len(single_screening_ages_sorted)), labels=[f'{x}' for x in single_screening_ages_sorted], fontsize=self.tick_fontsize)
        ax.set_yticks([])

        ax.set_xlabel(xlabel, fontsize=self.label_fontsize)

        # Add text annotations for single screening strategies
        for i, value in enumerate(single_icer_values):
            icer_rounded_mean= f'{value:.1f}'
            icer_rounded_lower_bound = f'{icer_lower_bounds[i]:.1f}'
            icer_rounded_upper_bound = f'{icer_upper_bounds[i]:.1f}'
            background_color = im.cmap(norm(value))
            self._add_text_to_heatmap(ax, i, 0, icer_rounded_mean, icer_rounded_lower_bound, icer_rounded_upper_bound, background_color)

        # Add grid lines to separate the boxes
        ax.set_xticks(np.arange(-0.5, len(single_screening_ages_sorted), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', size=0)

        return im