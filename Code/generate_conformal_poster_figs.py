# generate_conformal_poster_figs.py
#
# Script to generate all of the figures for the NeurIPS poster

import compare_calibrators
import seaborn as sns
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np

import conformal_multiclass_fitting as Cfit
import conformal_multiclass_experiments as CME
import cifar_processing as CiPro
import conformal_multiclass as CM
import pandas as pd
import synthetic_cqr_data

# ---------------------------------------------------------------------------
# Generate figure with conditional calibration (see method
# plot_single_calibration_experiment in the file compare_calibrators).
# ---------------------------------------------------------------------------

# First, show the failure of "single threshold" conformal
dim_disc = 5
n_val = 200
compare_calibrators.plot_naive_calibration(dim_disc, n_val,
                                           seed = 2,
                                           filename = "naive_calibration.pdf")

# Show a 'failure' of the marginal guarantee (like, why it is bad)

# Show this in the "real" world

compare_calibrators.plot_single_calibration_comparison(
    dim_disc, n_val, alpha = .1, seed = 2,
    filename = "split_versus_full_conditional.pdf")

compare_calibrators.plot_discretized_calibration(
    dim_disc, n_val, alpha = .12,
    seed=2, filename = "w-conditional-example.pdf")
