from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class Settings:
    # Colors used for plotting the four different screening approaches (single, repeated, with and without HLA genotyping)
    # Plotting colors that take colorblindness into account
    plot_colors: Dict[str, str] = field(default_factory=lambda: {
        'Single-time antibody screening': '#377eb8',
        'First-line genetic + single-time antibody screening': '#ff7f00',
        'Repeated antibody screening': '#4daf4a',
        'First-line genetic + repeated antibody screening': '#984ea3',
    })
    # Number of steps in the simulation, 1 step = 1 year
    number_of_years_to_simulate: int = 85
    # States of celiac disease in the model
    states: tuple = ('HEALTHY', 'ASYMPTOMATIC', 'SYMPTOMATIC', 'CLINICALEVALUATION', 'COMPLIANT', 'NONCOMPLIANT', 'DEAD')
    # List of screening ages, use -1 for no screening
    # First parameter tells the screening age(s), second parameter indicates whether HLA genotyping is used
    no_screening: tuple = ((-1,), False)
    # Discounting rate for screening, for now, screening costs are not discounted
    screening_cost_discount_rate: float = 0.00