# compare_calibrators.py
#
# Functions for comparing the Gibbs et al. and Jung et al. (with Bai
# et al.'s scaling correction) methods for conditional conformal
# prediction.

import numpy as np
import pandas as pd
import pdb
import seaborn as sns
import matplotlib.pyplot as plt

from conditionalconformal.synthetic_data import generate_cqr_data
from conditionalconformal.synthetic_data import indicator_matrix
from conditionalconformal import CondConf
from collections import namedtuple

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import basic_conditional_coverage as BCE
from tqdm import tqdm
import cvxpy as cvx
import warnings
from importlib import reload

import synthetic_cqr_data
reload(synthetic_cqr_data)

def predict_full_conformal(X_val, y_val, X_test,
                             poly : PolynomialFeatures,
                             reg : LinearRegression,
                             Phi_function,
                             alpha = .1):
    """Uses the full conformal method to perform predictions

    Returns a tuple (lbs, ubs) of vectors of (respectively) lower and
    upper bounds for predicted y values on the test data in X_test
    using a CondConf predictor.  Assumes that poly is a set of
    polynomial features and reg is a fit regression
    model. Phi_function should be a featurization method.

    """
    score_fn = lambda x, y : y - reg.predict(poly.fit_transform(x))
    score_inv_fn_ub = lambda s, x : \
        [-np.inf, reg.predict(poly.fit_transform(x)) + s]
    score_inv_fn_lb = lambda s, x : \
        [reg.predict(poly.fit_transform(x)) + s, np.inf]
    
    cond_conf = CondConf(score_fn, Phi_function, infinite_params = {})
    cond_conf.setup_problem(X_val, y_val)
    n_test = len(X_test)

    lbs = np.zeros((n_test,))
    ubs = np.zeros((n_test,))

    lbs_r = np.zeros((n_test,))
    ubs_r = np.zeros((n_test,))

    ii = 0
    for x_t in X_test:
        res = cond_conf.predict(alpha / 2, x_t, score_inv_fn_lb,
                                exact=True, randomize=True)
        lbs[ii] = res[0]
        res = cond_conf.predict(1 - alpha / 2, x_t, score_inv_fn_ub,
                                exact=True, randomize=True)
        ubs[ii] = res[1]
        ii += 1
    return (lbs, ubs)

def predict_split_conformal(X_val, y_val, X_test, poly, reg, Phi_function,
                              alpha = .1):
    """Uses the split conformal method to perform predictions

    Returns a tuple (lbs, ubs) of vectors of (respectively) lower and
    upper bounds for predicted y values on the test data in X_test
    using a CondConf predictor.  Assumes that poly is a set of
    polynomial features and reg is a fit regression
    model. Phi_function should be a featurization method.

    """
    scores_val = y_val - reg.predict(poly.fit_transform(X_val))
    n_val = X_val.shape[0]
    Phi_val = np.hstack((np.ones((n_val, 1)), Phi_function(X_val)))

    dim = Phi_val.shape[1]
    theta_low = cvx.Variable(dim)
    alpha_low = (1 - alpha/2)
    t = cvx.Variable(n_val)
    s = cvx.Variable(n_val)
    objective = cvx.Minimize(alpha_low * sum(t) + (1 - alpha_low) * sum(s))
    constraints = [t >= Phi_val @ theta_low - scores_val,
                   t >= 0,
                   s >= scores_val - Phi_val @ theta_low,
                   s >= 0]
    problem = cvx.Problem(objective, constraints)

    if ("MOSEK" in cvx.installed_solvers()):
        problem.solve(solver = cvx.MOSEK)
    else:
        problem.solve()

    theta_up = cvx.Variable(dim)
    alpha_up = alpha/2
    loss = cvx.sum(alpha_up * cvx.pos(Phi_val @ theta_up - scores_val) +
                   (1 - alpha_up) * cvx.pos(scores_val - Phi_val @ theta_up))
    problem = cvx.Problem(cvx.Minimize(loss))
    if ("MOSEK" in cvx.installed_solvers()):
        problem.solve(solver = cvx.MOSEK)
    else:
        problem.solve()

    n_test = X_test.shape[0]
    Phi_test = np.hstack((np.ones((n_test, 1)), Phi_function(X_test)))
    lbs = reg.predict(poly.fit_transform(X_test)) + Phi_test @ theta_low.value
    ubs = reg.predict(poly.fit_transform(X_test)) + Phi_test @ theta_up.value

    return (lbs, ubs)

def plot_single_calibration_comparison(dim_disc : int, n_val : int,
                                       alpha : float = .1,
                                       alpha_correction = True, seed = 0):
    (X_train, y_train, X_val, y_val, X_test, y_test) = \
        synthetic_cqr_data.generate_sinusoid_data(seed = seed, n_val = n_val,
                                                  n_disc = dim_disc,
                                                  n_train = 200)
    (lbs, ubs, lbs_off, ubs_off, disc) = full_split_predictions(
        X_train, y_train, X_val, y_val, X_test,
        alpha = alpha, alpha_correction = alpha_correction,
        dim_disc = dim_disc)

    # This is dirty rotten cheating... but that's fine
    rng = np.random.default_rng(seed)
    (theta_0, theta_1) = (2 * rng.random(2) - 1)
    (phi_0, phi_1) = (3.75 * np.pi + np.pi / 4) * rng.random(2)
    f_val = theta_0 * np.cos(phi_0 * X_val) + theta_1 * np.sin(phi_1 * X_val)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(X_val.squeeze(), f_val, "k-")
    axs[1].plot(X_val.squeeze(), f_val, "k-")
    axs[0].plot(X_test.squeeze(), y_test, "k.")
    axs[1].plot(X_test.squeeze(), y_test, "k.")
    # Plot the full method
    axs[0].fill_between(X_test.squeeze(), y1 = lbs, y2 = ubs)
    # Plot the split method
    axs[1].fill_between(X_test.squeeze(), y1 = lbs_off, y2 = ubs_off)
    plt.tight_layout()
    plt.show()

def full_split_predictions(X_train, y_train,
                               X_val, y_val, X_test,
                               dim_disc : int,
                               alpha : float = .1,
                               alpha_correction = True,
                               maximum_x : float = 1.0,
                               num_poly_features : int = 5):
    """Returns vectors of the full and split predictions on the given data

    Constructs a polynomial regression with degree num_poly_features,
    which it fits to the given training data (in X_train, y_train).
    Then performs both the split and full calibration experiment,
    where the conditional methods divide the data into features at
    intervals of size maximum_x / dim_disc.  The desired coverage
    level is 1 - alpha.

    Returns a 5-tuple of

      (lbs_full, ubs_full, lbs_split, ubs_split, disc)

    of the estimated lower (lbs) and upper (ubs) confidence bounds for
    both the full and split methods. The disc vector is the discretizations,
    so that x in [disc[i], disc[i + 1]) means x is in feature i.

    """
    poly = PolynomialFeatures(num_poly_features)
    reg = LinearRegression().fit(poly.fit_transform(X_train), y_train)
    discretization_level = 1 / dim_disc
    disc = np.arange(0, maximum_x + discretization_level, discretization_level)
    def Phi_function(x):
        return indicator_matrix(x, disc)
    (lbs, ubs) = predict_full_conformal(X_val, y_val, X_test, poly,
                                        reg, Phi_function, alpha = alpha)

    alpha_corrected = alpha
    dim_ratio = dim_disc / X_val.shape[0]
    if (alpha_correction):
        alpha_corrected = ((alpha - dim_ratio / 2)) / (1 - dim_ratio)
    alpha_corrected = max(alpha_corrected, 0)

    (lbs_off, ubs_off) = predict_split_conformal(
        X_val, y_val, X_test, poly, reg, Phi_function,
        alpha = alpha_corrected)
    
    return (lbs, ubs, lbs_off, ubs_off, disc)
    
def CompareConditionalCalibratorsOnce(dim_disc : int,
                                      n_val : int,
                                      alpha : float = .1,
                                      alpha_correction = False,
                                      seed = 0):
    """Generates a synthetic (one-dimensional) dataset and compares split
    and full calibration schemes

    Returns two tuples (on_coverage, off_coverage), each a tuple t containing

    t.marginal_coverage : the marginal coverage of the method
    t.group_coverage : the coverage of the method for each group in the data
    t.mean_length : the mean length of the confidence intervals
    t.group_length : the confidence interval length for each group in the data

    (here t is either on_coverage or off_coverage, whether we use the
    full method or split method).

    """
    # (X_train, y_train, X_val, y_val, X_test, y_test) = \
    #     generate_cqr_data(seed = seed, n_calib = n_val)
    (X_train, y_train, X_val, y_val, X_test, y_test) = \
        synthetic_cqr_data.generate_sinusoid_data(seed = seed, n_val = n_val,
                                                  n_disc = dim_disc)
    (lbs, ubs, lbs_off, ubs_off, disc) = \
        full_split_predictions(
            X_train, y_train, X_val, y_val, X_test,
            alpha = alpha, alpha_correction = alpha_correction,
            dim_disc = dim_disc, num_poly_features = 6)
    
    # Now, check whether we've gotten the good coverage
    covering_full = ((lbs <= y_test) & (ubs >= y_test))
    covering_split = ((lbs_off <= y_test) & (ubs_off >= y_test))
    n_test = len(y_test)

    # Now, find the coverage (both marginal and by set of indices)
    indicators_groups = indicator_matrix(X_test, disc)
    num_groups = indicators_groups.shape[1]
    full_group_coverage = np.zeros(num_groups)
    split_group_coverage = np.zeros(num_groups)
    full_group_length = np.zeros(num_groups)
    split_group_length = np.zeros(num_groups)
    for ii in range(num_groups):
        inds = (indicators_groups[:, ii] == 1)
        num_to_test = max(sum(inds), 1)
        full_group_coverage[ii] = sum(covering_full[inds]) / num_to_test
        split_group_coverage[ii] = sum(covering_split[inds]) / num_to_test
        full_group_length[ii] = sum(ubs[inds] - lbs[inds]) / num_to_test
        split_group_length[ii] = \
            sum(ubs_off[inds] - lbs_off[inds]) / num_to_test
        
    Coverage = namedtuple("Coverage",
                          ["marginal_coverage", "group_coverage",
                           "mean_length", "group_length"])
    full_coverage = Coverage(sum(covering_full) / n_test,
                               full_group_coverage,
                               np.mean(ubs - lbs),
                               full_group_length)
    split_coverage = Coverage(sum(covering_full) / n_test,
                                split_group_coverage,
                                np.mean(ubs_off - lbs_off),
                                split_group_length)
    return (full_coverage, split_coverage)

def _fill_df_tuple(df, start_ind, coverage_tuple, method_type, sample_ratio):
    """Fills some rows in the given data frame with the tuple's data
    """
    for jj in range(len(coverage_tuple.group_coverage)):
        df.loc[start_ind + jj] = \
            [1 - coverage_tuple.group_coverage[jj], jj,
             sample_ratio, method_type,
             coverage_tuple.group_length[jj]]

# NOTE (jduchi): Saved the results of a run of 200 of these
# to the data frame 'Data/simulate_one_d_200.pkl'

def PerformConditionalCoverageExperiment(
        num_experiments : int,
        sampling_ratios : list = [10., 20., 40., 80.],
        dim_disc : int = 10):
    """Performs multiple experiments comparing full and split
    calibration methods.

    """
    ## FOR THIS EXPERIMENT WE DISABLE THE ANNOYING DATA PASSING WARNINGS
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    df = pd.DataFrame(columns = ["Miscoverage", "Group Index",
                                 "Ratio",
                                 "type", "Length"])
    num_rows_in_data_frame = \
        2 * (num_experiments * len(sampling_ratios) * dim_disc + 1)
    df.reindex(range(num_rows_in_data_frame))

    curr_ind = 0
    for iter in tqdm(range(num_experiments)):
        for ratio_ind in range(len(sampling_ratios)):
            sample_ratio = sampling_ratios[ratio_ind]
            n_val = round(sample_ratio * (dim_disc + 1))
            (full_cov, split_cov) = \
                CompareConditionalCalibratorsOnce(
                    dim_disc, n_val, alpha = .1,
                    alpha_correction = True, seed = curr_ind)

            _fill_df_tuple(df, curr_ind, full_cov, "full", sample_ratio)
            curr_ind += len(full_cov.group_coverage)
            _fill_df_tuple(df, curr_ind, split_cov, "split", sample_ratio)
            curr_ind += len(split_cov.group_coverage)
            df.loc[curr_ind] = \
                [1 - full_cov.marginal_coverage, "marginal",
                 sample_ratio, "full", full_cov.mean_length]
            df.loc[curr_ind + 1] = \
                [1 - split_cov.marginal_coverage, "marginal",
                 sample_ratio, "split", split_cov.mean_length]
            curr_ind += 2
    return df

def PlotConditionalExperiment(df : pd.DataFrame, desired_miscoverage = .1,
                              aspect_ratio : float = 2.):
    """Plots results of the method PerformConditionalConverageExperiment

    For the dataframe given, which is assumed to be the result of
    PerformConditionalConverageExperiment, plots 2x the number of
    distinct "Ratio" values (in the df.Ratio column) plots. The first
    set displays miscoverages, the second the length of the confidence
    sets.

    """
    unique_ratios = np.unique(df.Ratio)
    for rat in unique_ratios:
        sub_df = df[df["Ratio"] == rat]
        sns.catplot(data = sub_df, x = "Group Index",
                    y = "Miscoverage", kind = "bar", hue = "type",
                    edgecolor = "k",
                    height = 4, aspect = aspect_ratio)
        plt.axhline(y = desired_miscoverage, color = "r", linestyle = "-",
                    linewidth = 1)
        plt.xlabel("Group Index", fontsize=16)
        plt.ylabel("Miscoverage", fontsize=16)
        plt.savefig(f"group_miscoverage_ratio_{round(rat)}.pdf",
                    bbox_inches="tight")
        plt.show()
    for rat in unique_ratios:
        sub_df = df[df["Ratio"] == rat]
        sub_df = sub_df[sub_df["Group Index"] != "marginal"]
        sns.catplot(data = sub_df, x = "Group Index",
                    y = "Length", kind = "box", hue = "type",
                    height = 4, aspect = aspect_ratio)
        plt.yscale("log")
        plt.xlabel("Group Index", fontsize=16)
        plt.ylabel("Length", fontsize=16)
        plt.savefig(f"group_length_ratio_{round(rat)}.pdf",
                    bbox_inches="tight")
        plt.show()
        
