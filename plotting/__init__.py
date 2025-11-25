import matplotlib
import numpy as np

# Set default font size for all figures
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rcParams['figure.max_open_warning'] = 0

# Default figure sizes
small_figure = np.array([8, 4.5])
medium_figure = np.array([12, 6.75])
wide_figure = np.array([24, 12])
big_figure   = np.array([16,  9])
huge_figure  = big_figure * 1.3

from plotting import plot_markov_model
