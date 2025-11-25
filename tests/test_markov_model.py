import os
import sys
import unittest

import numpy as np

sys.path.append('..')
from markov_model.markov_simulation_controller import MarkovSimulationController


class TestTestMarkovModel(unittest.TestCase):

    def assert_transition_probability_matrices(self, filename, path):
        transition_probability_matrices = []
        num_of_simulations = 100
        willingness_to_pay_threshold = 20000
        simu_controller = MarkovSimulationController(num_of_simulations, willingness_to_pay_threshold, filename, path)
        transition_probability_matrices = simu_controller.transition_probability_matrices
        # Check that all elements are positive and that all rows sum to one
        self.assertTrue(np.all(transition_probability_matrices >= 0))
        row_sums = np.sum(transition_probability_matrices, axis=3)
        np.testing.assert_almost_equal(row_sums, 1)

    def test_all_yaml_files(self):
        yaml_files = []
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/parameters/'))
        for root, _, files in os.walk(base_path):
            for file in files:
                if file.endswith('.yaml'):
                    yaml_files.append((file, root))
        
        for yaml_file, path in yaml_files:
            path = os.path.join(path, '')  # Ensure path ends with a slash
            with self.subTest(yaml_file=yaml_file):
                self.assert_transition_probability_matrices(yaml_file, path)

if __name__ == '__main__':
    unittest.main()