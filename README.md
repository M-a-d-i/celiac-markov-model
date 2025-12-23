# Probabilistic Markov Model For Determining The Cost-Effectiveness of Mass Screening for Celiac Disease

## Citation

If you use this code or the associated research, please cite the following publication:

**Mäkinen J, Heikkilä P, Pajula J, El Mekkaoui K, Størdal K, Lindfors K, et al. Screening for celiac disease in childhood: cost-effectiveness of multiple genetic and serological testing approaches. Clin Gastroenterol Hepatol. 2025. https://doi.org/10.1016/j.cgh.2025.11.030**

### BibTeX

```bibtex
@article{makinen_screening_2025,
	title = {Screening for celiac disease in childhood: cost-effectiveness of multiple genetic and serological testing approaches},
	issn = {1542-7714},
	shorttitle = {Screening for celiac disease in childhood},
	doi = {10.1016/j.cgh.2025.11.030},
	language = {eng},
	journal = {Clinical Gastroenterology and Hepatology: The Official Clinical Practice Journal of the American Gastroenterological Association},
	author = {Mäkinen, Jani and Heikkilä, Paula and Pajula, Juha and El Mekkaoui, Khaoula and Størdal, Ketil and Lindfors, Katri and Laiho, Jutta E. and Ranta, Jukka and Rökman, Jyri and Hyöty, Heikki and Emmert-Streib, Frank and Kurppa, Kalle and Kivelä, Laura and {HEDIMED investigator group}},
	month = dec,
	year = {2025},
	pmid = {41407089},
	keywords = {Health Economics, Cost-utility analysis, Pediatric screening, Preventive health care},
	pages = {S1542--3565(25)01032--8},
}
```

## Data Availability

The simulated results and datasets are available on Zenodo: 10.5281/zenodo.16877205

## Funding

This project received funding from the European Union’s Horizon 2020 research and innovation program under grant agreement No. 874864 HEDIMED, the Sigrid Jusélius Foundation, the Foundation for Pediatric Research, the Paulo Foundation, the Research  Council of Finland (361421, 347474), the Finnish Medical Foundation (7850), the Emil  Aaltonen Foundation, and the State funding for university-level health research, Tampere  University Hospital, Wellbeing services county of Pirkanmaa

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
