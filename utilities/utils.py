import numpy as np

from settings import settings


def generate_all_screening_strategies(screening_age_range):
    hla_genotyping = [False, True]
    youngest, oldest = screening_age_range
    oldest += 1
    screening_ages = [(i,) for i in range(youngest, oldest)]
    screening_ages += [(i, j) for i in range(youngest, oldest) for j in range(i+1, oldest)]
    screening_strategies = [(ages, hla) for ages in screening_ages for hla in hla_genotyping]
    return screening_strategies

def calculate_lifetime_costs_or_qalys(lifetime_population_distribution: np.ndarray, cost_or_qaly_matrix: np.ndarray, discount_rate: float, first_discount_year: int) -> float:
    modified_distribution = np.transpose(lifetime_population_distribution.copy())[:-1]
    # Apply half-cycle correction to state membership, transitions in the model happen at the end of each cycle
    modified_distribution[:,0] = modified_distribution[:,0]/2
    modified_distribution[:,-1] = modified_distribution[:,-1]/2
    years_without_discounting = np.ones(first_discount_year)
    years_with_discounting = np.array([(1+discount_rate)**(-i) for i in range(1, settings.number_of_years_to_simulate - first_discount_year + 1)])
    discount_vector = np.append(years_without_discounting, years_with_discounting)
    annual_discounted_costs_or_qalys = np.sum(cost_or_qaly_matrix*modified_distribution, axis=0) * discount_vector
    return np.sum(annual_discounted_costs_or_qalys)

def solve_beta_distribution_parameters(expected_value: float, standard_deviation: float) -> tuple[float, float]:
    alpha = ((1-expected_value)/standard_deviation**2 - 1/expected_value)*expected_value**2
    beta = alpha*(1/expected_value - 1)
    return alpha, beta

def solve_gamma_distribution_parameters(expected_value, standard_deviation):
    alpha = expected_value**2/standard_deviation**2
    beta = expected_value/standard_deviation**2
    return alpha, beta

def age_weighted_average(statistic_of_interest: np.ndarray) -> np.floating:
    # Use half-cycle correction to calculate age-weighted averages
    first_age_group = 0.5
    num_of_age_groups = len(statistic_of_interest)
    age_groups = np.arange(first_age_group, num_of_age_groups)
    weighted_average = np.average(age_groups, weights=statistic_of_interest)
    return weighted_average

def get_color(screening_strategy: tuple[tuple[int], bool]) -> str:
    color_mapping = {
        (1, False): 'Single-time antibody screening',
        (1, True): 'First-line genetic + single-time antibody screening',
        (2, False): 'Repeated antibody screening',
        (2, True): 'First-line genetic + repeated antibody screening',
    }
    key_for_color_mapping = (len(screening_strategy[0]), screening_strategy[1])
    key_for_color = color_mapping.get(key_for_color_mapping)
    
    if key_for_color is None:
        raise Exception(f"Color not found for screening strategy {screening_strategy}")
    
    color = settings.plot_colors.get(key_for_color)
    
    if color is None:
        raise Exception(f"Color not found for screening strategy {screening_strategy}")
    
    return color

# Return a different color for each screening strategy 
def get_colors(screening_strategies: list[tuple[tuple[int], bool]]) -> list[str]:
    return [get_color(i) for i in screening_strategies]
    
# Calculate 95% confidence interval for a timeseries based on the assumption that observations are identically and independently distributed, and come from the normal distribution
def iid_95_confidence_intervals_and_mean(statistic_of_interest: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(statistic_of_interest, axis=0)
    confidence_intervals = 1.96 * np.std(statistic_of_interest, axis=0)
    lower_bounds = means - confidence_intervals
    upper_bounds = means + confidence_intervals
    return means, lower_bounds, upper_bounds