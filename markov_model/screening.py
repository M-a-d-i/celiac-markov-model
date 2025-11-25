class Screening():
    def __init__(self, screening_parameters):
        self.screening_tests = screening_parameters["screening_tests"]
        self.cost_of_testing = screening_parameters["cost_of_testing"]
        self.hla_genotyping_cost = self.screening_tests["hla_genotyping"]["cost"] + self.cost_of_testing
        self.ttg_iga_cost = self.screening_tests["ttg_iga"]["cost"] + self.cost_of_testing
        self.ema_iga_cost = self.screening_tests["ema_iga"]["cost"] + self.cost_of_testing

        self.screening_costs = screening_parameters["costs"]
        self.biopsy_costs = self.screening_costs["biopsy"]
        self.diagnosis_followup_costs = self.screening_costs["diagnosis_followup"]

        self.biopsy_disutility = screening_parameters["biopsy_disutility"]
        self.genetic_risk_prevalence = screening_parameters["genetic_risk_prevalence"]
        self.celiac_disease_patients_with_10xULN_ttg_iga = screening_parameters["celiac_disease_patients_with_10xuln_ttg_iga"]
        self.false_positive_healthy_patients_with_10xULN_ttg_iga = screening_parameters["false_positive_healthy_patients_with_10xuln_ttg_iga"]

        self.antibody_screening_sensitivity = self.screening_tests["ttg_iga"]["sensitivity"]
        self.antibody_screening_specificity = self._get_antibody_screening_specificity()
        self.genetic_plus_antibody_screening_sensitivity = self.screening_tests["hla_genotyping"]["sensitivity"] * self.antibody_screening_sensitivity

    def _get_antibody_screening_specificity(self):
        false_positive_rate = (1-self.screening_tests["ttg_iga"]["specificity"]) * self.false_positive_healthy_patients_with_10xULN_ttg_iga * (1-self.screening_tests["ema_iga"]["specificity"])
        specificity = 1 - false_positive_rate
        return specificity

    def _get_genetic_plus_antibody_screening_specificity(self, prevalence_of_cd):
        false_positives_from_genotyping = (1 - self.screening_tests["hla_genotyping"]["specificity"]) * (1 - self.genetic_risk_prevalence)
        true_positives_from_genotyping_without_cd = self.screening_tests["hla_genotyping"]["sensitivity"] * (self.genetic_risk_prevalence - prevalence_of_cd)
        positives_from_genetic_screening_without_cd_proceeding_to_antibody_screening = false_positives_from_genotyping + true_positives_from_genotyping_without_cd
        false_positives_from_genetic_plus_antibody_screening = positives_from_genetic_screening_without_cd_proceeding_to_antibody_screening * (1 - self.antibody_screening_specificity)
        negatives = 1 - prevalence_of_cd
        specificity = (negatives - false_positives_from_genetic_plus_antibody_screening) / negatives 
        return specificity

    def get_screening_sensitivity(self, test_for_hla):
        assert isinstance(test_for_hla, bool), "test_for_hla must be a boolean"
        if test_for_hla:
            return self.genetic_plus_antibody_screening_sensitivity
        else:
            return self.antibody_screening_sensitivity

    def get_screening_specificity(self, test_for_hla, prevalence_in_participants):
        assert isinstance(test_for_hla, bool), "test_for_hla must be a boolean"
        if test_for_hla:
            return self._get_genetic_plus_antibody_screening_specificity(prevalence_in_participants)
        else:
            return self.antibody_screening_specificity

    def _calculate_true_positives(self, percentage_of_population, test_name):
        if percentage_of_population == 0:
            return 0
        if not 0 <= percentage_of_population <= 1:
            raise ValueError("Percentage of population must be positive.")
        return percentage_of_population * self.screening_tests[test_name]["sensitivity"]

    def _calculate_false_positives(self, percentage_of_population, test_name):
        if percentage_of_population == 0:
            return 0
        if not 0 <= percentage_of_population <= 1:
            raise ValueError("Percentage of population must be positive.")
        return percentage_of_population * (1 - self.screening_tests[test_name]["specificity"])

    def _calculate_screening_costs_from_ttg_iga_to_followup(self, percentage_of_population_participating_in_ttg_iga, percentage_of_population_participating_in_ttg_iga_with_cd):
        percentage_of_population_without_cd = percentage_of_population_participating_in_ttg_iga - percentage_of_population_participating_in_ttg_iga_with_cd

        ttg_iga_cost = percentage_of_population_participating_in_ttg_iga * self.ttg_iga_cost
        ttg_iga_true_positives = self._calculate_true_positives(percentage_of_population_participating_in_ttg_iga_with_cd, "ttg_iga")
        ttg_iga_true_positives_with_10xULN = ttg_iga_true_positives * self.celiac_disease_patients_with_10xULN_ttg_iga
        ttg_iga_true_positives_under_10xULN = ttg_iga_true_positives - ttg_iga_true_positives_with_10xULN
        ttg_iga_false_positives = self._calculate_false_positives(percentage_of_population_without_cd, "ttg_iga")
        ttg_iga_false_positives_with_10xULN = ttg_iga_false_positives * self.false_positive_healthy_patients_with_10xULN_ttg_iga
        ttg_iga_false_positives_under_10xULN = ttg_iga_false_positives - ttg_iga_false_positives_with_10xULN

        percentage_participating_in_ema_iga = ttg_iga_true_positives_with_10xULN + ttg_iga_false_positives_with_10xULN
        ema_iga_cost = percentage_participating_in_ema_iga * self.ema_iga_cost
        ema_iga_true_positives = self._calculate_true_positives(ttg_iga_true_positives_with_10xULN, "ema_iga")
        ema_iga_false_positives = self._calculate_false_positives(ttg_iga_false_positives_with_10xULN, "ema_iga")
        ema_iga_false_negatives = ttg_iga_true_positives_with_10xULN - ema_iga_true_positives
        ema_iga_true_negatives = ttg_iga_false_positives_with_10xULN - ema_iga_false_positives

        biopsy_participants = ttg_iga_true_positives_under_10xULN + ttg_iga_false_positives_under_10xULN + ema_iga_false_negatives + ema_iga_true_negatives
        biopsy_disutility = biopsy_participants * self.biopsy_disutility
        biopsy_cost = biopsy_participants * self.biopsy_costs

        screening_procedure_costs = ttg_iga_cost + ema_iga_cost + biopsy_cost

        diagnosis_followup_costs = (ema_iga_true_positives + ema_iga_false_positives + biopsy_participants) * self.diagnosis_followup_costs

        screening_costs_from_ttg_iga_to_followup = screening_procedure_costs + diagnosis_followup_costs

        return screening_costs_from_ttg_iga_to_followup, biopsy_disutility

    def _calculate_population_distribution_after_genotyping(self, percentage_of_population_participating_in_screening, undiagnosed_cd_prevalence):
        percentage_resistant_to_cd = percentage_of_population_participating_in_screening * (1 - self.genetic_risk_prevalence)
        percentage_susceptible_to_cd_without_cd = percentage_of_population_participating_in_screening * self.genetic_risk_prevalence - undiagnosed_cd_prevalence

        genotyping_true_positives_with_cd = self._calculate_true_positives(undiagnosed_cd_prevalence, "hla_genotyping")
        genotyping_true_positives_without_cd = self._calculate_true_positives(percentage_susceptible_to_cd_without_cd, "hla_genotyping")
        genotyping_false_positives = self._calculate_false_positives(percentage_resistant_to_cd, "hla_genotyping")
        percentage_with_positive_genotyping = genotyping_true_positives_with_cd + genotyping_true_positives_without_cd + genotyping_false_positives 
        return percentage_with_positive_genotyping, genotyping_true_positives_with_cd

    def calculate_screening_cost_and_disutility(self, percentage_of_population_participating_in_screening, undiagnosed_cd_prevalence, screening_round_number, test_for_hla):
        total_screening_costs = 0
        biospy_disutility = 0
        if not test_for_hla:
            screening_costs_from_ttg_iga_to_followup, disutility = self._calculate_screening_costs_from_ttg_iga_to_followup(percentage_of_population_participating_in_screening, undiagnosed_cd_prevalence)
            total_screening_costs += screening_costs_from_ttg_iga_to_followup
            biospy_disutility += disutility
        elif test_for_hla:
            percentage_participating_in_ttg_iga, genotyping_true_positives_with_cd = self._calculate_population_distribution_after_genotyping(percentage_of_population_participating_in_screening, undiagnosed_cd_prevalence)
            screening_costs_from_ttg_iga_to_followup, disutility = self._calculate_screening_costs_from_ttg_iga_to_followup(percentage_participating_in_ttg_iga, genotyping_true_positives_with_cd)
            total_screening_costs += screening_costs_from_ttg_iga_to_followup
            biospy_disutility += disutility
            if screening_round_number == 1:
                genotyping_cost = percentage_of_population_participating_in_screening * self.hla_genotyping_cost
                total_screening_costs += genotyping_cost

        return total_screening_costs, biospy_disutility