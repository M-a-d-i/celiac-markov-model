from markov_model.markov_simulation_controller import MarkovSimulationController


def binary_search_hla_price(debug_filename, default_filename, num_simulations_debug, num_simulations_validation, willingness_to_pay_threshold, screening_age_range, hla_price_solution_tolerance, cost_effectiveness_diff_tolerance, comparison_type):
    low, high = 0, 100  # Initial range for HLA genotyping cost

    best_price = None
    previous_mid = None
    while True:
        print(f"Binary search range: {low:.2f}, {high:.2f}")
        mid = (low + high) / 2
        if previous_mid is not None and abs(mid - previous_mid) < hla_price_solution_tolerance:
            best_price = mid
            break
        previous_mid = mid
        cost_effectiveness_diff = simulate_and_get_cost_effectiveness_diff(mid, debug_filename, num_simulations_debug, willingness_to_pay_threshold, screening_age_range, comparison_type)

        if abs(cost_effectiveness_diff) < cost_effectiveness_diff_tolerance:
            best_price = mid
            break
        if cost_effectiveness_diff < 0:
            high = mid
        else:
            low = mid

    if abs(cost_effectiveness_diff) > cost_effectiveness_diff_tolerance:
        return None

    print(f"Validating HLA genotyping cost: {mid:.2f}")
    cost_effectiveness_diff = simulate_and_get_cost_effectiveness_diff(mid, default_filename, num_simulations_validation, willingness_to_pay_threshold, screening_age_range, comparison_type)
    return best_price

def simulate_and_get_cost_effectiveness_diff(new_price, data_filename, num_simulations, willingness_to_pay_threshold, screening_age_range, comparison_type):
    print(f"Testing HLA genotyping cost: {new_price:.2f}")
    simulation_results = update_hla_price_simulate_model_and_get_results(new_price, data_filename, num_simulations, willingness_to_pay_threshold, screening_age_range)
    best_strategies = get_best_strategies(simulation_results.inmb_expected_value)
    cost_effectiveness_diff = get_cost_effectiveness_diff(best_strategies, comparison_type)
    return cost_effectiveness_diff

def update_hla_price_simulate_model_and_get_results(new_price, data_filename, num_simulations, willingness_to_pay_threshold, screening_age_range):
    model = MarkovSimulationController(num_simulations, willingness_to_pay_threshold, data_filename)
    model.imported_data.screening_parameters["screening_tests"]["hla_genotyping"]["cost"] = new_price
    model.reinitialize_markov_model()
    simulated_strategies = model.simulate_screening_strategies(screening_age_range)
    return model.simulation_results

def get_cost_effectiveness_diff(best_strategies, comparison_type):
    single_time_with_genotyping, single_time_without_genotyping, repeated_with_genotyping, repeated_without_genotyping = best_strategies
    if comparison_type == "single":
        cost_effectiveness_with = single_time_with_genotyping[1]
        cost_effectiveness_without = single_time_without_genotyping[1]
    else:
        cost_effectiveness_with = repeated_with_genotyping[1]
        cost_effectiveness_without = repeated_without_genotyping[1]
    
    cost_effectiveness_diff = cost_effectiveness_with - cost_effectiveness_without
    
    print(f"INMB with genotyping: {cost_effectiveness_with:.2f}, without genotyping: {cost_effectiveness_without:.2f}, difference: {cost_effectiveness_diff:.2f}")
    
    return cost_effectiveness_diff

def get_best_strategies(input_dict):
    categories = {
        "single_time_with_genotyping": {},
        "single_time_without_genotyping": {},
        "repeated_with_genotyping": {},
        "repeated_without_genotyping": {}
    }

    for key, value in input_dict.items():
        (ages, genotyping) = key
        if len(ages) == 1:
            if genotyping:
                categories["single_time_with_genotyping"][key] = value
            else:
                categories["single_time_without_genotyping"][key] = value
        elif len(ages) == 2:
            if genotyping:
                categories["repeated_with_genotyping"][key] = value
            else:
                categories["repeated_without_genotyping"][key] = value

    for category in categories:
        categories[category] = dict(sorted(categories[category].items(), key=lambda item: item[1], reverse=True))

    best_single_time_with_genotyping = list(categories["single_time_with_genotyping"].items())[0]
    best_single_time_without_genotyping = list(categories["single_time_without_genotyping"].items())[0]
    best_repeated_with_genotyping = list(categories["repeated_with_genotyping"].items())[0]
    best_repeated_without_genotyping = list(categories["repeated_without_genotyping"].items())[0]

    return best_single_time_with_genotyping, best_single_time_without_genotyping, best_repeated_with_genotyping, best_repeated_without_genotyping

if __name__ == "__main__":
    debug_filename = "Debug Scenario.yaml"
    default_filename = "Default Scenario.yaml"
    num_simulations_debug = 25
    num_simulations_validation = 25000
    willingness_to_pay_threshold = 20000
    screening_age_range = (3, 18)
    hla_price_solution_tolerance = 0.01
    cost_effectiveness_diff_tolerance = 0.01
    comparison_types = ["single", "repeated"]

    for comparison_type in comparison_types:
        print(f'Running binary search for HLA genotyping price breakpoint for {comparison_type} comparison')
        best_price = binary_search_hla_price(debug_filename, default_filename, num_simulations_debug, num_simulations_validation, willingness_to_pay_threshold, screening_age_range, hla_price_solution_tolerance, cost_effectiveness_diff_tolerance, comparison_type)
        if best_price is not None:
            print(f"HLA genotyping price breakpoint: {best_price:.2f}")
        else:
            print("Failed to find an optimal HLA genotyping cost.")
