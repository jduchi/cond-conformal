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

# --------------------------------------------------------------------------- #
#      Generates the comparison between the calibration methods (Fig. 2)      #
# --------------------------------------------------------------------------- #

df_one_dim = compare_calibrators.PerformConditionalCoverageExperiment(
    200, sampling_ratios = [10, 20, 40, 80, 160],
    dim_disc = 5)

df_one_dim.to_pickle("simulate_one_d_200.pkl")

df_one_dim = pd.read_pickle("Data/simulate_one_d_200.pkl")
df_one_dim["type"] = df_one_dim["type"].replace("offline", "split")
df_one_dim["type"] = df_one_dim["type"].replace("online", "full")

compare_calibrators.PlotConditionalExperiment(
    df_one_dim, desired_miscoverage = .1, aspect_ratio = 2)

# --------------------------------------------------------------------------- #
#    This is the actual random directions experiment on the CIFAR-100 data    #
# --------------------------------------------------------------------------- #

random_direction_tuple = \
    CME.random_directions_test(10, .001, do_full_conformal = True)

(clc, chc, cmc, slc, shc, smc, flc, fhc, fmc) = random_direction_tuple[:]

random_df = CME.make_frame_of_random_directions_test(
    clc, chc, cmc, slc, shc, smc, flc, fhc, fmc)

random_df.to_pickle("random_directions_cifar_10_10.pkl")

# Now plot it

sns.catplot(data = random_df, x = "Direction", y = "Coverage", kind = "box",
            hue = "Type", palette = "Blues",
            height = 4, aspect = 1.6)
plt.axhline(y = .9, color = "r", linestyle = "-", linewidth = 1)
plt.ylabel("Coverage", fontsize=14)
plt.xlabel("Data slice", fontsize = 14)
plt.savefig("random_directions_experiment.pdf", bbox_inches = "tight")
plt.show()
