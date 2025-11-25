# HEDIMED Probabilistic Markov Model For Determining The Cost-Effectiveness of Mass Screening for Celiac Disease

## Running the code

Requirements for running the code: Python version 3.9.1, see requirements.txt file for required python packages, or run the following command in the terminal:

```bash
pip install -r requirements.txt
```

The code is run by the following command in the terminal:

```bash
python main.py
```

The main function doesn't have any mandatory arguments, but there are a few optional arguments that can be passed.  These are listed below:

```bash	
-r or --screening_age_range: Age range of the population to be screened. Default: (3, 18)
-w or --willingness_to_pay_threshold: Willingness to pay threshold for cost-effectiveness analysis in Euros. Default: 20000
-n or --num_of_simulations: Number of Monte Carlo simulations per screening strategy. Default: 25000
-p or --number_of_best_performing_strategies_to_plot: Number of best strategies to plot in some optional graphs. Default: 25
-f or --plot_all_figures: Plot all figures, including supplementary results and figures
-s or --run_sensitivity_analysis: Run all sensitivity analysis scenarios
-m or --run_model_validation: Run all model validation scenarios (currently no scenarios available)
-d or --debug: Use Debug Scenario.yaml (low variance scenario) instead of Default Scenario.yaml
```

Examples of how to run the code with optional arguments:

```bash
python main.py --screening_age_range 3 12 --num_of_simulations 5000
```

```bash
python main.py --screening_age_range 10 15 --plot_all_figures
```

```bash
python main.py --screening_age_range 19 35 --num_of_simulations 2500 --number_of_best_performing_strategies_to_plot 10
```

```bash
python main.py --run_sensitivity_analysis
```

```bash
python main.py --debug -n 500 --plot_all_figures
```

```bash
python main.py --willingness_to_pay_threshold 25000 --screening_age_range 5 15
```

## Additional utilities

### HLA Price Point Solver

The repository also includes `solve_hla_pricepoint.py`, a utility script that finds the optimal HLA genotyping cost where screening with and without genotyping have equivalent cost-effectiveness. This uses binary search to find the price breakpoint.

```bash
python solve_hla_pricepoint.py
```

This script will:
- Use binary search to find the HLA genotyping price breakpoint
- Compare single-time and repeated screening strategies
- Validate results using both debug and full simulation scenarios
