# python -m venv 315a
# source 315a/bin/activate

from typing import List
import numpy as np
import cvxpy as cvx
from numpy.random import randn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm
from collections import namedtuple
from typing import Callable

from conditionalconformal import CondConf



def GenerateData(n, dim, w = None,
                 heterogeneous_noise = None):
    X = randn(n, dim)
    if (w is None):
        w = randn(dim)
    elif (w is not None):
        if (w.shape[0] != dim):
            raise ValueError("Dimension of w is {w.shape[0]}, should be {dim}")
    # Generate w uniformly at random on sphere
    w = w / np.sqrt(np.dot(w, w))
    if (heterogeneous_noise is None):
        sigmas = np.ones(n)
    else:
        sigmas = heterogeneous_noise(X)
    y = X @ w + sigmas * randn(n)
    return (X, y, w)

def PredictQuantileLevels(
        X : np.ndarray,
        theta_hat,
        Phi_function : Callable[[np.ndarray], np.ndarray] = None):
    """Returns vector of predictions of error quantile level

    Returns vector with entries

    v = Phi * theta_hat,

    where Phi is the matrix constructed by Phi_function(X), though if
    Phi_function is unspecified, simply makes the first column all 1s
    and remainder an indicator matrix of X being positive.

    """
    if (Phi_function is None):
        nn = X.shape[0]
        Phi = np.hstack((np.ones((nn, 1), (X > 0))))
    else:
        Phi = Phi_function(X)
    return Phi @ theta_hat

def FitStaticQuantile(X, y, w_hat, alpha):
    S = np.abs(X @ w_hat - y)
    return np.quantile(S, 1 - alpha)

def FitConformalQuantile(
        X, y, w_hat, alpha,
        Phi_function : Callable[[np.ndarray], np.ndarray] = None):
    """Fits a quantile predictor on positive-parts of X covariates

    Finds the function h of the form h(x) = theta' * phi(x), where

      phi(x) = (1, 1{x_1 > 0}, ..., 1{x_d > 0})

    is an indicator function of positive parts including a bias,
    so that h minimizes

      sum_{i = 1}^n l_alpha(h(X_i) - S_i)

    where S_i = abs(y_i - x_i' * w_hat) is the absolute error of the
    putative prediction using w_hat.  Returns theta such that the
    minimizer is

    h(x) = theta' * phi(x)

    """
    nn = X.shape[0]
    dd = X.shape[1] + 1
    # First, re-featurize X
    if (Phi_function is None):
        Phi = np.hstack((np.ones((nn, 1)), (X > 0)))
    else:
        Phi = Phi_function(X)
    # Now get absolute errors values
    S = np.abs(X @ w_hat - y)

    theta = cvx.Variable(dd)
    loss = cvx.sum(alpha * cvx.pos(Phi @ theta - S) +
                   (1 - alpha) * cvx.pos(S - Phi @ theta))
    problem = cvx.Problem(cvx.Minimize(loss))
    if ("MOSEK" in cvx.installed_solvers()):
        problem.solve(solver = cvx.MOSEK)
    else:
        problem.solve()
    theta_hat = theta.value
    return theta_hat

def FullConditionalQuantilePredictions(X_calib : np.ndarray,
                                       y_calib : np.ndarray,
                                       X_test : np.ndarray,
                                       w_hat : np.ndarray,
                                       alpha : float):
    """Returns prediction intervals for each test data point

    Uses the Cherian et al. method to compute confidence sets for each
    data point in X_test. Returns an (n_test x 2) array of confidence
    intervals for y values on the test set.

    Uses score function s(x, y) = |y - x' * w_hat|, and predicts
    the 1 - alpha quantile of the resulting scores on the test data,
    using the featurization

    phi(x) = [1, 1{x_1 > 0}, ..., 1{x_d > 0}]

    that is, an intercept and indicators of positivity.

    """
    # define score functions based on absolute loss
    score_fn = lambda x, y : np.abs(y - np.dot(x, w_hat))
    score_inv_fn = lambda s, x : \
        [np.dot(x.flatten(), w_hat) - s, np.dot(x.flatten(), w_hat) + s]
    # Define featurization function
    def phi_fn(x):
        if (len(x.shape) > 1):
            return np.concatenate((np.ones((x.shape[0], 1)), x > 0), axis = 1)
        return np.concatenate(([1], x > 0))
    cond_conf = CondConf(score_fn, phi_fn, infinite_params = {})
    cond_conf.setup_problem(X_calib, y_calib)
    n_test = X_test.shape[0]
    y_interval_test_preds = np.zeros((n_test, 2))
    for ii in tqdm(range(len(X_test))):
        x_i = X_test[ii, :]
        # Returns an np array of the form [[low], [high]]
        residual = cond_conf.predict(1 - alpha, x_i,
                                     score_inv_fn, exact = True,
                                     randomize = True)
        y_interval_test_preds[ii, :] = [residual[0][0], residual[1][0]]
    return y_interval_test_preds

def RunConditionalCoverageExperiment(ntrain: int, nval: int, d: int,
                                     alpha: float,
                                     coord_groups: List[List[int]],
                                     random_sigma = False):
    """Tests group conditional coverage based on coordinate groups

    Performs a single experiment using ntrain training data points and
    nval validation datapoints, in dimension d, for a linear
    regression problem. Specifically, generates 3 data sets: a
    training set, test set, and validation set, consisting of data in
    dimension d and where

      y = x' * theta + sigma(x) * N(0, 1)

    where sigma(x) is determined as follows: if random_sigma is true,
    then uses

      sigma^2(x) = dot(multipliers, (x > 0))

    where multipliers is a random vector with Uni[0, 2] entries.
    Otherwise, sets multipliers to be logarithmically spaced between
    e^4 and e^{-4}.

    The coord_groups list contains groups of coordinates to condition
    on.
    
    Returns a tuple with named entries (not necessarily in this order)
    as follows:

     offline_miscoverage
     offline_size
     offline_marginal
     offline_marginal_size

    Performance of the method building an offline confidence
    interval. Respectively, the elements contain the miscoverage by
    coordinate group, the size of the confidence set by coordinate group,
    and scalars of the marginal coverage and the average set size

     online_miscoverage
     online_size
     online_marginal
     online_marginal_size

    Performance of the method building confidence intervals depending
    on the actual observed test data point (i.e., the Gibbs et al. method).
    Otherwise, same as above.
    
     static_marginal
     static_miscoverage
     static_size
     static_marginal_size

    Performance of vanilla conformal method, which fits a single
    quantile to construct confidence sets.

    """
    if (random_sigma):
        multipliers = 2 * np.random.rand(d)
    else:
        multipliers = np.exp(np.linspace(4, -4, d))
    def noise_func(X):
        sigmas = np.sqrt((X > 0) @ multipliers)
        return sigmas
    
    (X, y, w) = GenerateData(ntrain, d, heterogeneous_noise = noise_func)
    (Xval, yval, w) = GenerateData(nval, d, w,
                                   heterogeneous_noise = noise_func)
    ntest = 10 * ntrain
    (Xtest, ytest, w) = GenerateData(ntest, d, w,
                                     heterogeneous_noise = noise_func)
    # Find estimated predictor
    w_hat = np.linalg.solve(X.T @ X, X.T @ y)
    # Find calibration vector with corrected alpha
    sampling_ratio = nval / d
    alpha_corrected = ((alpha - 1 / (2 * sampling_ratio))
                       / (1 - 1 / sampling_ratio))
    theta_hat = FitConformalQuantile(Xval, yval, w_hat, alpha_corrected)
    # Do the vanilla coverage (static quantile)
    s_pred_static = FitStaticQuantile(Xval, yval, w_hat, alpha * (1 - 1 / nval))
    # Now, evaluate coverage
    S_test = np.abs(Xtest @ w_hat - ytest)
    S_pred = PredictQuantileLevels(Xtest, theta_hat)
    y_intervals_online = FullConditionalQuantilePredictions(
        Xval, yval, Xtest, w_hat, alpha)

    # Store the mis-coverages across the coordinate groups
    cvg_by_group_offline = np.zeros(len(coord_groups))
    cvg_by_group_online = np.zeros(len(coord_groups))
    cvg_by_group_static = np.zeros(len(coord_groups))
    size_by_group_offline = np.zeros(len(coord_groups))
    size_by_group_online = np.zeros(len(coord_groups))
    size_by_group_static = np.zeros(len(coord_groups))
    ii = 0
    for coord_inds in coord_groups:
        inds = np.ones(Xtest.shape[0], dtype=np.int32)
        for c_ind in coord_inds:
            inds = inds & (Xtest[:, c_ind] > 0)
        test_indices = (inds > 0)
        n_conditional_test = sum(test_indices)  # Number in this group
        # Evaluate for offline construction of predictor
        num_failures = np.sum(S_test[test_indices] > S_pred[test_indices])
        prop_failures = num_failures / max(n_conditional_test, 1)
        cvg_by_group_offline[ii] = prop_failures
        size_by_group_offline[ii] = np.mean(S_pred[test_indices])

        # Evaluate for online construction of predictor
        num_cover = np.sum(
            (ytest[test_indices] >= y_intervals_online[test_indices, 0]) &
            (ytest[test_indices] <= y_intervals_online[test_indices, 1]))
        cvg_by_group_online[ii] = \
            (n_conditional_test - num_cover) / max(n_conditional_test, 1)
        size_by_group_online[ii] = \
            np.mean(y_intervals_online[test_indices, 1]
                    - y_intervals_online[test_indices, 0]) / 2

        # Evaluate for static quantile
        num_failures = np.sum(S_test[test_indices] > s_pred_static)
        prop_failures = num_failures / max(n_conditional_test, 1)
        cvg_by_group_static[ii] = prop_failures
        size_by_group_static[ii] = s_pred_static
        ii += 1

    Coverage = namedtuple("Coverage",
               ["offline_miscoverage", "offline_size",
                "online_miscoverage", "online_size",
                "static_miscoverage", "static_size",
                "offline_marginal",
                "online_marginal",
                "static_marginal",
                "offline_marginal_size",
                "online_marginal_size",
                "static_marginal_size"])
    mean_offline_size = np.mean(S_pred)
    mean_online_size = np.mean(y_intervals_online[:, 1]
                               - y_intervals_online[:, 0]) / 2
    cvg = Coverage(cvg_by_group_offline, size_by_group_offline,
                   cvg_by_group_online, size_by_group_online,
                   cvg_by_group_static, size_by_group_static,
                   sum(S_test > S_pred) / ntest,
                   1 - sum((ytest >= y_intervals_online[:, 0]) &
                           (ytest <= y_intervals_online[:, 1])) / ntest,
                   sum(S_test > s_pred_static) / ntest,
                   mean_offline_size,
                   mean_online_size,
                   s_pred_static)
    return cvg

def ConditionalMiscoverageExperiments(num_experiments, alpha, *,
                                      dim = 20, ntrain = 100,
                                      sampling_ratio = 10,
                                      coord_groups = None):
    """Performs experiments to evaluate types of conditional coverage
    """
    if (coord_groups is None):
        num_groups = min(round(dim / 2), 10)
        num_per_group = round(np.ceil(dim / num_groups))
        coord_groups = []
        num_left = dim
        curr_ind = 0
        while (curr_ind < dim):
            num_left = dim - curr_ind
            num_in_group = min(num_left, num_per_group)
            coord_groups.append(list(range(curr_ind, curr_ind + num_in_group)))
            curr_ind = curr_ind + num_in_group

    # Make data frame for storing results
    df = pd.DataFrame(columns = ["Miscoverage", "Group Index", "q_type",
                                 "Length"])
    num_rows_in_data_frame = 3 * num_experiments * (len(coord_groups) + 1)

    curr_ind = 0
    df = df.reindex(range(num_rows_in_data_frame))
    for iter in range(num_experiments):
        print(f"Conditional coverage experiment {iter + 1} "
              + f"of {num_experiments}")
        nval = dim * sampling_ratio
        cvg_tuple = \
            RunConditionalCoverageExperiment(ntrain, nval,
                                             dim, alpha,
                                             coord_groups)
        for jj in range(len(coord_groups)):
            df.loc[curr_ind + jj] = \
                [cvg_tuple.offline_miscoverage[jj], jj, "offline",
                 cvg_tuple.offline_size[jj]]
        curr_ind += len(coord_groups)
        for jj in range(len(coord_groups)):
            df.loc[curr_ind + jj] = \
                [cvg_tuple.online_miscoverage[jj], jj, "online",
                 cvg_tuple.online_size[jj]]
        curr_ind += len(coord_groups)
        for jj in range(len(coord_groups)):
            df.loc[curr_ind + jj] = \
                [cvg_tuple.static_miscoverage[jj], jj, "static",
                 cvg_tuple.static_size[jj]]
        curr_ind += len(coord_groups)
        df.loc[curr_ind] = \
            [cvg_tuple.offline_marginal, "marginal", "offline",
             cvg_tuple.offline_marginal_size]
        df.loc[curr_ind + 1] = \
            [cvg_tuple.online_marginal, "marginal", "online",
             cvg_tuple.online_marginal_size]
        df.loc[curr_ind + 2] = \
            [cvg_tuple.static_marginal, "marginal", "static",
             cvg_tuple.static_marginal_size]
        curr_ind += 3
    return df

def PlotConditionalCoverageExperiment(df, desired_miscoverage = .1,
                                      plot_kind = "box"):
    """Plots the results of ComputeConditionalMiscoverages

    Options for the parameter plot_kinds are those from seaborn's
    catplot, and good defaults include box (makes a box plot) and bar
    (makes a bar/histogram-like plot)

    """
    custom_colors = [(.3, .8, 1.), (1., .7, .1), (0, .8, 0)]
    sns.catplot(data = df, x = "Group Index",
                y = "Miscoverage", kind = plot_kind.lower(), hue="q_type",
                height = 5, aspect = 2, palette = custom_colors)
    plt.axhline(y = desired_miscoverage, color = "r", linestyle = "-.",
                linewidth=1)
    plt.show()

    sns.catplot(data = df, x = "Group Index",
                y = "Length", kind = "box", hue = "q_type",
                height = 5, aspect = 2,
                palette = custom_colors)
    plt.show()


    
    

def RunSingleCalibrationExperiment(ntrain, nval, d, alpha):
    """Performs a single calibration experiment

    Generates three samples for a regression problem, then uses them
    to evaluate a conformalized "conditional-like" quantile predictor:
    a training sample, validation sample, and a test sample.

    Each dataset is generated according to

    y = X * w + noise,

    where w is identical throughout and Uniform on the sphere, X is a
    random Gaussian matrix, and noise is N(0, 1) noise. For the
    training data, X is an ntrain-by-d matrix, for the validation, it
    is an nval-by-d matrix, and for the test, it is (10 *
    ntrain)-by-d.

    Fits a conformal quantile predictor (see method
    FitConformalQuantile) on the validation dataset using level alpha,
    and returns the empirical miscoverage rate on the test dataset
    as well as the average confidence interval length.

    """
    (X, y, w) = GenerateData(ntrain, d)
    (Xval, yval, w) = GenerateData(nval, d, w)
    ntest = 10 * ntrain
    (Xtest, ytest, w) = GenerateData(ntest, d, w)
    # Find estimated predictor
    w_hat = np.linalg.solve(X.T @ X, X.T @ y)
    # Find calibration vector
    theta_hat = FitConformalQuantile(Xval, yval, w_hat, alpha)
    # Now, evaluate coverage
    S_test = np.abs(Xtest @ w_hat - ytest)
    S_pred = PredictQuantileLevels(Xtest, theta_hat)
    num_failures = np.sum(S_test > S_pred)
    return (num_failures / ntest, np.mean(S_pred))


def ComputeMarginalMiscoverages(num_experiments, alpha, *,
                                dim = 20, ntrain = 100,
                                ratios = [5, 10, 20, 40],
                                correction_style = None,
                                print_each = 10):
    """Performs experiment to compute the miscoverages on random Gaussian data

    Performs num_experiments distinct experiments, where each experiment
    consists of the following:

    - Generate training data (X, y), where X has size ntrain-by-dim, and fit
      a linear regression on it.
    - Compute a "conditional" conformalized predictor using a
      validation dataset of size dim * ratios (varying ratios), using
      a level alpha_effective, where alpha_effective is chosen as
      follows depending on correction_style:

      alpha_effective = alpha if correction_style is None

      alpha_effective = alpha + (d / nval) * (alpha - 1)
         if correction_style is "Naive"

      alpha_effective = (alpha - d / (2 * nval)) / (1 - d / nval)
         if correction_style is "Scaling"

      The naive correction corresponds to using the (1 + 1/n)(1 -
      alpha) quantile familiar from vanilla conformal prediction, but
      we increase this with dimension. The "Scaling" correction
      corresponds to the asymptotics that Bai et al. identify in
      "Understanding the Under-Coverage Bias in Uncertainty
      Estimation"
    
    - Evaluate the (marginal) miscoverage fraction

    Returns two num_experiments-by-len(ratios) matrices: the first in
    the tuple is the miscoverages in each of these experiments, the
    second is the average length of the confidence set in the
    experiment.

    """
    # For each of the following ratios
    miscoverages_by_experiment = np.zeros((num_experiments, len(ratios)))
    conf_len_by_experiment = np.zeros((num_experiments, len(ratios)))
    for iter in range(num_experiments):
        if ((iter + 1) % print_each == 0):
            print(f"Experiment {iter+1} of {num_experiments}")
        for r_ind in range(len(ratios)):
            nval = dim * ratios[r_ind]
            # Compute correction
            alpha_corrected = alpha
            if (correction_style is None or correction_style.lower() == "none"):
                alpha_corrected = alpha
            elif (correction_style.lower() == "naive"):
                alpha_corrected = alpha + (dim / nval) * (alpha - 1)
            elif (correction_style.lower() == "bai"):
                alpha_corrected = (alpha - dim / (2 * nval)) / (1 - dim / nval)
            else:
                raise ValueError(f"Correction style {correction_style}" +
                                 "not supported")
            alpha_corrected = max(alpha_corrected, 0)
            (p_fail, conf_len) = RunSingleCalibrationExperiment(
                ntrain, nval, dim, alpha_corrected)
            miscoverages_by_experiment[iter, r_ind] = p_fail
            conf_len_by_experiment[iter, r_ind] = conf_len
    return (miscoverages_by_experiment, conf_len_by_experiment)

def MakeFrameOfResults(num_experiments = 200, *,                       
                       alpha = .1,
                       correction_styles = ["None", "Naive", "Scaling"],
                       ratios = [5, 10, 20, 40]):
    """Makes a data frame corresponding to the experiments

    Performs num_experiments experiments using
    ComputeMarginalMiscoverages for each correction style in
    correction_styles and each ratio of dimension to sample size in
    ratios, using a default coverage level alpha.

    The returned data frame has columns corresponding to
    "Miscoverage", "Ratio", and "Correction".

    """
    num_rows_in_frame = num_experiments * len(correction_styles) * len(ratios)
    df = pd.DataFrame(columns = ["Miscoverage", "Length", "Ratio",
                                 "Correction"])
    df = df.reindex(range(num_rows_in_frame))

    curr_ind = 0
    for c_style in correction_styles:
        (msc, c_lengths) = ComputeMarginalMiscoverages(
            num_experiments, alpha,
            correction_style = c_style,
            ratios = ratios)
        for jj in range(len(ratios)):
            curr_ratio = ratios[jj]
            for ii in range(num_experiments):
                df.loc[curr_ind + ii] = [msc[ii, jj], c_lengths[ii, jj],
                                         curr_ratio, c_style]
            curr_ind += num_experiments
    return df

def PlotExperiment(df, desired_miscoverage = .1):
    custom_colors = [(.4, .4, 1.), (1., .7, 0), (0, .8, 0)]
    sns.catplot(data = df, x = "Ratio", y = "Miscoverage",
                hue = "Correction", kind="box", palette = custom_colors)
    # palette="bright")
    plt.axhline(y = desired_miscoverage, color = "r", linestyle = "-.",
                linewidth=1)
    plt.savefig("miscoverage_guassian_linreg.pdf", bbox_inches = "tight")
    plt.show()

    sns.catplot(data = df, x = "Ratio", y = "Length",
                hue = "Correction", kind="box", palette = custom_colors)
    # palette="colorblind")
    plt.savefig("confsize_guassian_linreg.pdf", bbox_inches = "tight")
    plt.show()
    
