import argparse
import glob
import os

from markov_model.markov_simulation_controller import MarkovSimulationController
from plotting.plot_markov_model import PlotMarkovModel


def main(args):
    # Simulation settings
    num_of_simulations: int = args.num_of_simulations
    screening_age_range: tuple[int, int] = args.screening_age_range
    # Number of best performing strategies to plot in some supplementary figures
    number_of_top_strategies_to_plot: int = args.number_of_best_performing_strategies_to_plot
    plot_all_figures: bool = args.plot_all_figures
    run_sensitivity_analysis: bool = args.run_sensitivity_analysis
    run_model_validation: bool = args.run_model_validation
    willingness_to_pay_threshold: int = args.willingness_to_pay_threshold

    filenames = ['Default Scenario.yaml'] if not args.debug else ['Debug Scenario.yaml']
    paths = ['data/parameters/']

    scenarios = {
        'sensitivity analysis': run_sensitivity_analysis,
        'model validation': run_model_validation
    }

    for scenario, should_run in scenarios.items():
        if should_run:
            scenario_path = f'data/parameters/{scenario}/*.yaml'
            yaml_files = glob.glob(scenario_path)
            filenames.extend([os.path.basename(yaml_file) for yaml_file in yaml_files])
            paths.extend([f'data/parameters/{scenario}/'] * len(yaml_files))

    for i, (filename, path) in enumerate(zip(filenames, paths)):
        print(f"Simulating {filename} ({i+1}/{len(filenames)})")
        markov_model = MarkovSimulationController(num_of_simulations, willingness_to_pay_threshold, filename, path)
        simulated_screening_strategies = markov_model.simulate_screening_strategies(screening_age_range)
        simulation_results = markov_model.simulation_results

        plot = PlotMarkovModel(simulation_results, willingness_to_pay_threshold)
        plot.main_results(simulated_screening_strategies)
        plot.model_calibration_statistics()

        if plot_all_figures:
            plot.supplementary_figures(markov_model, number_of_top_strategies_to_plot)

        number_of_scenarios = len(filenames)
        display_figures = False if number_of_scenarios > 1 else True

        plot.save_figures_and_export_results(screening_age_range, num_of_simulations, filename, display_figures)
        plot.clear_figures()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulation model')
    parser.add_argument('-r', '--screening_age_range', nargs=2, type=int, default=(3, 18), help='Screening age range')
    parser.add_argument('-w', '--willingness_to_pay_threshold', type=int, default=20000, help='Willingness to pay threshold for cost-effectiveness analysis in Euros')
    parser.add_argument('-n', '--num_of_simulations', type=int, default=25000, help='Number of Monte Carlo simulations per screening strategy')
    parser.add_argument('-p', '--number_of_best_performing_strategies_to_plot', type=int, default=25, help='Number of best strategies to plot')
    parser.add_argument('-f', '--plot_all_figures', action='store_true', help='Plot all figures')
    parser.add_argument('-s', '--run_sensitivity_analysis', action='store_true', help='Run all sensitivity analysis scenarios')
    # There are currently no model validation scenarios, but this argument is kept for future use
    parser.add_argument('-m', '--run_model_validation', action='store_true', help='Run all model validation scenarios')
    parser.add_argument('-d', '--debug', action='store_true', help='Use Debug Scenario.yaml')
    main(parser.parse_args())