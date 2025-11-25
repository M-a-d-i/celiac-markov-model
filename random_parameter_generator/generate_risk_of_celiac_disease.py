import numpy as np

from settings import settings


class RiskOfCeliacDisease:
    def __init__(self, prevalence_data: dict):
        # Load data from yaml file
        self.prevalence_data = prevalence_data
        self.cumulative_risk = self.prevalence_data['cumulative_risk']
        self.scaling = self.prevalence_data['scaling']
        # Cumulative risk data
        self.ages = np.array(self.cumulative_risk['age'])
        self.cumulative_risks = np.array(self.cumulative_risk['risk'])
        self.level_of_evidence = self.cumulative_risk['level_of_evidence']
        # Extrapolate risk curve by fitting a linear function to the last two data points
        self.extrapolated_risk_curve = self._get_extrapolated_risk_curve()

        self.risk_scaling_factor = self.scaling['risk'] / self._get_cumulative_risk(self.scaling['age'])

        self.transition_probabilities_healthy_to_celiac_disease = np.array([
            self._get_transition_probability_healthy_to_celiac_disease(age) 
            for age in range(0, settings.number_of_years_to_simulate)
        ])

    def get_scaled_cumulative_risk(self, age: int) -> np.float64:
        return self._get_cumulative_risk(age) * self.risk_scaling_factor

    def _get_transition_probability_healthy_to_celiac_disease(self, age: int) -> np.float64:
        # Calculations based on pages 18-19 of: https://aaltodoc.aalto.fi/bitstream/handle/123456789/119634/master_M%c3%a4kinen_Jani_2023.pdf?sequence=1&isAllowed=y
        cumulated_risk = self.get_scaled_cumulative_risk(age + 1) - self.get_scaled_cumulative_risk(age)
        cumulated_risk = cumulated_risk / 100
        transition_probability_healthy_to_celiac_disease = cumulated_risk / (1 - self.get_scaled_cumulative_risk(age) / 100)
        return transition_probability_healthy_to_celiac_disease

    def _get_cumulative_risk(self, age: int) -> np.float64:
        # Select between interpolation and extrapolation
        extrapolation_start_age = self.ages[-2]
        if age > extrapolation_start_age:
            extrapolated_risk = np.float64(np.polyval(self.extrapolated_risk_curve, age))
            return extrapolated_risk
        else:
            interpolated_risk = np.float64(np.interp(age, self.ages, self.cumulative_risks))
            return interpolated_risk

    def _get_extrapolated_risk_curve(self) -> np.poly1d:
        # Select data for extrapolation
        extrapolation_start_age = self.ages[-2]
        extrapolation_end_age = self.ages[-1]
        extrapolation_start_risk = self.cumulative_risks[-2]
        extrapolation_end_risk = self.cumulative_risks[-1]
        age_data = np.array([extrapolation_start_age, extrapolation_end_age])
        risk_data = np.array([extrapolation_start_risk, extrapolation_end_risk])
        extrapolation_data = np.polyfit(age_data, risk_data, 1)
        # Fit linear function as extrapolation
        return np.poly1d(extrapolation_data)