import numpy as np

from random_parameter_generator.generate_risk_of_celiac_disease import RiskOfCeliacDisease
from settings import settings


class TransitionProbabilities:
    def __init__(self, number_of_simulations: int, imported_data):
        self.number_of_simulations = number_of_simulations
        self.imported_data = imported_data
        self.risk_of_celiac_disease = RiskOfCeliacDisease(self.imported_data.prevalence_data)
        self.transition_probabilities_healthy_to_celiac_disease = self.risk_of_celiac_disease.transition_probabilities_healthy_to_celiac_disease
        self.asymptomatic_transitions = self.imported_data.asymptomatic_transitions
        self.diagnostic_delays = self.imported_data.diagnostic_delays
        self.gluten_free_diet_compliance = self.imported_data.gluten_free_diet_compliance
        self.compliance_after_diagnosis = self.imported_data.compliance_after_diagnosis
        self.hazard_ratio_data = self.imported_data.mortality_hazard_ratio
        self.initial_celiac_disease_transitions = self.imported_data.initial_celiac_disease_transitions

        self.transition_probabilities = self._generate_transition_probabilities()

    def _generate_transition_probabilities(self):
        from_healthy_state = self._from_healthy_state()
        from_asymptomatic_state = self._from_asymptomatic_state()
        from_symptomatic_state = self._from_symptomatic_state()
        from_clinical_evaluation_state = self._from_clinical_evaluation_state()
        from_compliant_state = self._from_compliant_state()
        mortality_hazard_ratios = self._mortality_hazard_ratios()

        # Store results in a dictionary
        randomized_transition_probabilities = {
            "from healthy": from_healthy_state,
            "from asymptomatic": from_asymptomatic_state,
            "from symptomatic": from_symptomatic_state,
            "from clinical evaluation": from_clinical_evaluation_state,
            "gluten free compliance rates": from_compliant_state,
            "mortality hazard ratios": mortality_hazard_ratios
        }

        # Return the dictionary
        return randomized_transition_probabilities

    def _from_healthy_state(self):
        def generate_total_transition_probabilities_for_developing_celiac_disease(age: int):
            if self.transition_probabilities_healthy_to_celiac_disease[age] == 0:
                return np.zeros(self.number_of_simulations)
            level_of_evidence = self.risk_of_celiac_disease.level_of_evidence
            alpha = level_of_evidence*self.transition_probabilities_healthy_to_celiac_disease[age]
            beta  = level_of_evidence*(1-self.transition_probabilities_healthy_to_celiac_disease[age])
            return np.random.beta(alpha, beta, size=self.number_of_simulations)

        ages = self.initial_celiac_disease_transitions['age']
        level_of_evidence = self.initial_celiac_disease_transitions['level_of_evidence']
        manifestation_weights = np.array([
            self.initial_celiac_disease_transitions['probability_weights']['to_asymptomatic'],
            self.initial_celiac_disease_transitions['probability_weights']['to_symptomatic'],
            self.initial_celiac_disease_transitions['probability_weights']['to_clinical_evaluation']
        ]).T * level_of_evidence
        celiac_disease_manifestation_distribution= np.zeros((settings.number_of_years_to_simulate, self.number_of_simulations, 3))

        for (age_from, age_to), weights in self._parse_age_ranges(ages, manifestation_weights):
            size = (age_to - age_from, self.number_of_simulations)
            celiac_disease_manifestation_distribution[age_from:age_to] = np.random.dirichlet(weights, size=size)

        celiac_disease_manifestation_distribution = celiac_disease_manifestation_distribution.transpose((1, 0, 2))

        # This works as intended
        total_transition_probability_of_developing_celiac_disease = np.array([
            generate_total_transition_probabilities_for_developing_celiac_disease(age) 
            for age in range(0, settings.number_of_years_to_simulate)
        ], dtype=np.float32).T

        # These operations work as intended
        transition_probabilities_to_different_manifestations_of_celiac_disease = total_transition_probability_of_developing_celiac_disease[:, :, np.newaxis] * celiac_disease_manifestation_distribution
        healthy = 1 - np.sum(transition_probabilities_to_different_manifestations_of_celiac_disease, axis=2)
        healthy = np.expand_dims(healthy, axis=-1)
        transition_probabilities = np.concatenate((healthy, transition_probabilities_to_different_manifestations_of_celiac_disease), axis=-1)

        return transition_probabilities

    def _from_asymptomatic_state(self) -> np.ndarray:
        probability_weights = self.asymptomatic_transitions['probability_weights']
        to_asymptomatic = probability_weights['to_asymptomatic']
        to_symptomatic = probability_weights['to_symptomatic']
        to_clinical_evaluation = probability_weights['to_clinical_evaluation']
        transitions_probabilities_from_asymptomatic = np.array([to_asymptomatic, to_symptomatic, to_clinical_evaluation]).T

        ages = self.asymptomatic_transitions['age']
        level_of_evidence = self.asymptomatic_transitions['level_of_evidence']

        diriclet_distribution_parameters = transitions_probabilities_from_asymptomatic * level_of_evidence

        probabilities_shape = (settings.number_of_years_to_simulate, self.number_of_simulations, diriclet_distribution_parameters.shape[1])
        age_based_probabilities = np.zeros(probabilities_shape)

        for (age_from, age_to), distribution_params in self._parse_age_ranges(ages, diriclet_distribution_parameters):
            size = (age_to - age_from, self.number_of_simulations)
            age_based_probabilities[age_from:age_to] = np.random.dirichlet(distribution_params, size=size)

        age_based_probabilities = age_based_probabilities.transpose((1, 0, 2))

        return age_based_probabilities

    def _from_symptomatic_state(self) -> np.ndarray:
        def solve_transition_probabilities(num_years: int, delay_from_symptomatic: float, delay_from_clinical_evaluation: float, level_of_evidence: int) -> np.ndarray:
            # See supplementary material for the derivation of these equations
            symptomatic_to_clinical_evaluation = (delay_from_clinical_evaluation-1)/(delay_from_clinical_evaluation*(-delay_from_clinical_evaluation+delay_from_symptomatic+1))
            symptomatic_to_diagnosis = 1/(delay_from_clinical_evaluation*(-delay_from_clinical_evaluation+delay_from_symptomatic+1))
            symptomatic_to_symptomatic = 1 - symptomatic_to_clinical_evaluation - symptomatic_to_diagnosis
            symptomatic_to_compliant = symptomatic_to_diagnosis * self.compliance_after_diagnosis
            symptomatic_to_non_compliant = symptomatic_to_diagnosis - symptomatic_to_compliant

            clinical_evaluation_to_clinical_evaluation = (delay_from_clinical_evaluation - 1) / delay_from_clinical_evaluation
            expected_passage_time_from_clinical_evalution = 1 / (1 - clinical_evaluation_to_clinical_evaluation)
            assert np.isclose(expected_passage_time_from_clinical_evalution, delay_from_clinical_evaluation, rtol=1e-5), "Expected passage time from clinical evaluation is not equal to the delay from clinical evaluation"

            expected_passage_time_from_symptomatic = (1+symptomatic_to_clinical_evaluation*expected_passage_time_from_clinical_evalution)/(1-(1-symptomatic_to_clinical_evaluation-symptomatic_to_diagnosis))
            assert np.isclose(expected_passage_time_from_symptomatic, delay_from_symptomatic, rtol=1e-5), "Expected passage time from symptomatic is not equal to the delay from symptomatic"

            parameters = np.array([symptomatic_to_symptomatic, symptomatic_to_clinical_evaluation, symptomatic_to_compliant, symptomatic_to_non_compliant]) * level_of_evidence
            size = (num_years, self.number_of_simulations)

            return np.random.dirichlet(parameters, size=size)

        ages = self.diagnostic_delays['age']
        delay_from_first_doctors_visit = np.array(self.diagnostic_delays['from_first_doctors_visit'])
        delay_from_first_symptoms = np.array(self.diagnostic_delays['from_first_symptoms'])
        level_of_evidence = np.array(self.diagnostic_delays['level_of_evidence'])

        generated_transition_probabilities = np.zeros((settings.number_of_years_to_simulate, self.number_of_simulations, 4))

        for (age_from, age_to), delay_from_symptoms, delay_from_doctors_visit, evidence in self._parse_age_ranges(ages, delay_from_first_symptoms, delay_from_first_doctors_visit, level_of_evidence):
            num_years = age_to - age_from
            generated_transition_probabilities[age_from:age_to] = solve_transition_probabilities(num_years, delay_from_symptoms, delay_from_doctors_visit, evidence)

        generated_transition_probabilities = generated_transition_probabilities.transpose((1, 0, 2))

        return generated_transition_probabilities

    def _from_clinical_evaluation_state(self) -> np.ndarray:
        def solve_transition_probabilities(num_years: int, delay_from_clinical_evaluation: float, level_of_evidence: int) -> np.ndarray:
            clinical_evaluation_to_clinical_evalution = (delay_from_clinical_evaluation - 1) / delay_from_clinical_evaluation
            clinical_evaluation_to_diagnosis = 1 - clinical_evaluation_to_clinical_evalution
            clinical_evaluation_to_compliant = clinical_evaluation_to_diagnosis * self.compliance_after_diagnosis
            clinical_evaluation_to_non_compliant = clinical_evaluation_to_diagnosis - clinical_evaluation_to_compliant

            expected_passage_time_from_clinical_evalution = 1 / (1 - clinical_evaluation_to_clinical_evalution)
            assert np.isclose(expected_passage_time_from_clinical_evalution, delay_from_clinical_evaluation, rtol=1e-5), "Expected passage time from clinical evaluation is not equal to the delay from clinical evaluation"

            parameters = np.array([clinical_evaluation_to_clinical_evalution, clinical_evaluation_to_compliant, clinical_evaluation_to_non_compliant]) * level_of_evidence
            size = (num_years, self.number_of_simulations)

            return np.random.dirichlet(parameters, size=size)

        ages = self.diagnostic_delays['age']
        delay_from_first_doctors_visit = np.array(self.diagnostic_delays['from_first_doctors_visit'])
        level_of_evidence = np.array(self.diagnostic_delays['level_of_evidence'])

        generated_transition_probabilities = np.zeros((settings.number_of_years_to_simulate, self.number_of_simulations, 3))

        for (age_from, age_to), delay, evidence in self._parse_age_ranges(ages, delay_from_first_doctors_visit, level_of_evidence):
            num_years = age_to - age_from
            generated_transition_probabilities[age_from:age_to] = solve_transition_probabilities(num_years, delay, evidence)

        generated_transition_probabilities = generated_transition_probabilities.transpose((1, 0, 2))

        return generated_transition_probabilities

    def _from_compliant_state(self) -> np.ndarray:
        ages = self.gluten_free_diet_compliance['age']
        compliance_rates = self.gluten_free_diet_compliance['compliance_rate']
        level_of_evidence = self.gluten_free_diet_compliance['level_of_evidence']
        
        alphas = np.array(compliance_rates) * np.array(level_of_evidence)
        betas = np.array(level_of_evidence) - alphas

        generated_compliance_rates = np.zeros((settings.number_of_years_to_simulate, self.number_of_simulations))

        for (age_from, age_to), alpha, beta in self._parse_age_ranges(ages, alphas, betas):
            # Handle indexing problem unique to GFD compliance rates
            # Transition probabilities tell the next age compliance instead of the current, hence the -1 and if statements
            age_to = age_to - 1 if age_to != settings.number_of_years_to_simulate else age_to
            age_from = age_from - 1 if age_from != 0 else age_from
            size = (age_to - age_from, self.number_of_simulations)
            # Handle GDF compliance 100% case separately
            if beta == 0:
                generated_compliance_rates[age_from:age_to] = np.ones(size)
            else:
                generated_compliance_rates[age_from:age_to] = np.random.beta(alpha, beta, size=size)

        return generated_compliance_rates.T

    def _mortality_hazard_ratios(self) -> np.ndarray:
        mean = self.hazard_ratio_data['mean']
        ci_95_low, ci_95_high = self.hazard_ratio_data['95_ci']
        log_mean = np.log(mean)
        sigma = (np.log(ci_95_high) - np.log(ci_95_low)) / (2*1.96)
        return np.random.lognormal(mean=log_mean, sigma=sigma, size=(self.number_of_simulations, settings.number_of_years_to_simulate))

    def _parse_age_ranges(self, ages, *arrays):
        parsed_ranges = []
        for age_range, *values in zip(ages, *arrays):
            age_from, age_to = age_range.split('-')
            age_from = int(age_from)
            age_to = int(age_to) + 1 if age_to else settings.number_of_years_to_simulate
            parsed_ranges.append(((age_from, age_to), *values))
        return parsed_ranges