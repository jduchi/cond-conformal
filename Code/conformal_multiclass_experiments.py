# conformal_multiclass_experiments.py
#
# Code to do multiclass classification (on CIFAR-100) using a
# pretrained 50 layer ResNet. Note: this is *research* code, so do not
# expect it to be particularly robust.
#
# Author: John Duchi (jduchi@stanford.edu)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pdb
from tqdm import tqdm

from importlib import reload

from conditionalconformal.condconf import setup_cvx_problem_calib
from conditionalconformal.condconf import CondConf

# Include the processing file and class files
import cifar_processing as CiPro
import conformal_multiclass as CM
import conformal_multiclass_fitting as Cfit

# Make the naming easier, and re-import to make sure it's current
reload(Cfit)
reload(CM)

# TODO (jduchi): Should add some methodology to allow checking
# coverage by actually integrating against the weight functions rather
# than just doing actual coverage, as this is what we are supposed
# to be targeting.

from conformal_multiclass import LinearMulticlass
from conformal_multiclass import RandomLinear
from conformal_multiclass import SimpleScalarWithProjection
from conformal_multiclass import LogisticScoreDataset
from conformal_multiclass import QuantileLoss
from conformal_multiclass import ConformalPredictor

import cifar_processing

def get_training_data(prop_train = .8):
    """Returns a randomly split CIFAR100 training dataset

    Returns a tuple (train, val) containing the training data and
    validation data in the form of a NumpyDataset, where prop_train
    fraction of the data are in train and (1 - prop_train) are in val.

    """
    full_train_data = CiPro.NumpyDataset(
        np.load("cifar100_train_features.npy"),
        np.load("cifar100_train_labels.npy"))
    (train_data, val_data) = \
        CiPro.split_into_train_and_validation(full_train_data, prop_train)
    return (train_data, val_data)

def single_experiment(desired_dimension_ratio = .01,
                      dim_reduction_type = Cfit.DimReductionType.RANDOM,
                      *,
                      num_training_epochs = 7,
                      desired_alpha = .1,
                      do_full_conformal : bool = False):
    prop_train = .8
    (train_data, val_data) = get_training_data(prop_train)
    training_l2 = 1e-3

    (pred_model, conditional_quant_model, vanilla_quant_model) = \
        Cfit.train_model_and_conformalize(
            train_data, val_data,
            num_epochs = num_training_epochs,
            desired_dimension_ratio = desired_dimension_ratio,
            dim_reduction_type = dim_reduction_type,
            training_l2 = training_l2,
            alpha_desired = desired_alpha)

    proj_matrix = \
        conditional_quant_model.projection_layer.projection_matrix.detach()

    test_data = CiPro.NumpyDataset(
        np.load("cifar100_test_features.npy"),
        np.load("cifar100_test_labels.npy"))

    cond_eval_tuple = \
        evaluate_conformal_along_extremes(test_data, pred_model,
                                          quant_model = conditional_quant_model,
                                          direction_vectors = proj_matrix.T,
                                          do_full_conformal = False)

    v_eval_tuple = \
        evaluate_conformal_along_extremes(test_data, pred_model,
                                          quant_model = vanilla_quant_model,
                                          direction_vectors = proj_matrix.T,
                                          do_full_conformal = False)
    if (do_full_conformal):
        full_eval_tuple = \
            evaluate_conformal_along_extremes(
                test_data, pred_model, direction_vectors = proj_matrix.T,
                val_data = val_data,
                do_full_conformal = True)
        return (cond_eval_tuple, v_eval_tuple, full_eval_tuple)
    else:
        return (cond_eval_tuple, v_eval_tuple, None)

def random_directions_test(total_tests = 10, desired_dim_ratio = .002,
                           do_full_conformal : bool = False):
    """Does an experiment of coverage on random split directions

    Performs total_tests different (offline-computed) conditional
    conformal predictors, each running the method single_experiment,
    with features on an example x determined by W @ x. Here, W is a
    random matrix of i.i.d. N(0, 1) entries, whose leading dimension
    is desired_dim_ratio * n_val (where n_val is the validation sample
    size).

    Returns a tuple

    (conditional_low_coverages, conditional_high_coverages,
     conditional_marg_coverage,
     static_low_coverages, static_high_coverages,
     static_marg_coverage,
     full_low_coverages, full_high_coverages,
     full_marg_coverage)

    whose entries are  arrays or scalars, indexed by the leading dimension of
    W.  The arrays *_low_coverages are the coverages of the two
    conformal predictors (static---meaning a single scalar to
    determine the confidence set size---and conditional, which uses W
    @ x) on the lowest test data with the lowest 20% values for
    dot(W[i, :], x), one for each potential index i.  *_high_coverages
    is the same, but with coverage rates for the highest 20% values.

    """
    conditional_low_coverages = []
    conditional_high_coverages = []
    conditional_marg_coverage = []
    static_low_coverages = []
    static_high_coverages = []
    static_marg_coverage = []
    full_low_coverages = []
    full_high_coverages = []
    full_marg_coverage = []
    test_data = CiPro.NumpyDataset(
        np.load("cifar100_test_features.npy"),
        np.load("cifar100_test_labels.npy"))
    for test_ind in tqdm(range(total_tests)):
        # print(f"*** Test {test_ind + 1} of {total_tests} ***")
        # Get the conditional coverage dictionary and the "vanilla"
        # (static) coverage dictionary
        (c_eval, s_eval, f_eval) = single_experiment(
            desired_dimension_ratio = desired_dim_ratio,
            dim_reduction_type = Cfit.DimReductionType.RANDOM,
            do_full_conformal = do_full_conformal)
        conditional_marg_coverage.append(c_eval[0] / c_eval[1])
        static_marg_coverage.append(s_eval[0] / s_eval[1])
        for cov_low in c_eval[2]:
            conditional_low_coverages.append(cov_low)
        for cov_high in c_eval[3]:
            conditional_high_coverages.append(cov_high)
        for stat_low in s_eval[2]:
            static_low_coverages.append(stat_low)
        for stat_high in s_eval[3]:
            static_high_coverages.append(stat_high)
        if (do_full_conformal):
            for f_low in f_eval[2]:
                full_low_coverages.append(f_low)
            for f_high  in f_eval[3]:
                full_high_coverages.append(f_high)
            full_marg_coverage.append(f_eval[0] / f_eval[1])

    return (conditional_low_coverages, conditional_high_coverages,
            conditional_marg_coverage,
            static_low_coverages, static_high_coverages,
            static_marg_coverage,
            full_low_coverages, full_high_coverages,
            full_marg_coverage)
    
def superclass_conditional_test(total_tests = 10, pca_dim = 40):
    conditional_coverages = {}
    static_coverages = {}
    test_data = CiPro.NumpyDataset(
        np.load("cifar100_test_features.npy"),
        np.load("cifar100_test_labels.npy"))
    for test_ind in tqdm(range(total_tests)):
        # print(f"*** Test {test_ind + 1} of {total_tests} ***")
        # Get the conditional coverage dictionary and the "vanilla"
        # (static) coverage dictionary
        (c_cov, v_cov) = superclass_conditional_experiment(pca_dim, test_data)
        for key in c_cov.keys():
            if (key in conditional_coverages):
                conditional_coverages[key].append(c_cov[key])
                static_coverages[key].append(v_cov[key])
            else:
                conditional_coverages[key] = [c_cov[key]]
                static_coverages[key] = [v_cov[key]]

    return (conditional_coverages, static_coverages)

def make_frame_of_dictionaries(conditional_coverages,
                               static_coverages):
    """Makes a data frame from dictionaries of coverages

    The data frame is indexed by Coverage, Superclass, and quant_type,
    where the quant_type is whether it uses a "conditional" quantile
    or a static quantile to estimate.
    """
    keys = conditional_coverages.keys()
    num_keys = len(keys)
    total_class_coverages = 0;
    for key in conditional_coverages.keys():
        total_class_coverages += len(conditional_coverages[key])
    for key in static_coverages.keys():
        total_class_coverages += len(static_coverages[key])
    df = pd.DataFrame(columns = ["Coverage", "Superclass",
                                 "quant_type"])
    df.reindex(range(total_class_coverages))
    curr_ind = 0
    for key in conditional_coverages.keys():
        curr_coverages = conditional_coverages[key]
        for ii in range(len(curr_coverages)):
            df.loc[curr_ind] = [curr_coverages[ii], key, "conditional"]
            curr_ind += 1
    for key in static_coverages.keys():
        curr_coverages = static_coverages[key]
        for ii in range(len(curr_coverages)):
            df.loc[curr_ind] = [curr_coverages[ii], key, "static"]
            curr_ind += 1
    return df

def make_frame_of_random_directions_test(
        conditional_low_coverages, conditional_high_coverages,
        conditional_marg_coverages,
        static_low_coverages, static_high_coverages,
        static_marg_coverages,
        full_low_coverages, full_high_coverages,
        full_marg_coverages):
    """
    """
    num_tests = len(static_marg_coverages)
    
    df = pd.DataFrame(columns = ["Coverage", "Experiment",
                                 "Direction", "Type"])
    total_entries = (len(conditional_high_coverages) * 6 + 3 * num_tests)
    df.reindex(range(total_entries))
    curr_ind = 0
    all_coverages = [conditional_low_coverages, conditional_high_coverages,
                     static_low_coverages, static_high_coverages,
                     full_low_coverages, full_high_coverages]
    num_directions_per_test = \
        round(len(conditional_high_coverages) / num_tests)
    names = ["Split", "Split", "Static", "Static", "Full", "Full"]
    # Add in coverages across the splits in experiments
    for c_ind in range(len(all_coverages)):
        coverage = all_coverages[c_ind]
        for ii in range(len(coverage)):
            c_level = coverage[ii]
            df.loc[curr_ind] = [c_level,
                                round(np.floor(ii / num_directions_per_test)),
                                ii % num_directions_per_test,
                                names[c_ind]]
            curr_ind += 1
    # Add in marginal coverages across experiments
    names = ["Split", "Static", "Full"]
    marg_coverages = [conditional_marg_coverages,
                      static_marg_coverages, full_marg_coverages]
    for c_ind in range(len(marg_coverages)):
        method_coverage = marg_coverages[c_ind]
        for ii in range(len(method_coverage)):
            c_level = method_coverage[ii]
            df.loc[curr_ind] = [c_level, ii, "marginal", names[c_ind]]
            curr_ind += 1
    return df

def plot_superclass_frame(df, desired_coverage = .9):
    """Plots results of experiment
    """
    # First, get the mean coverage for conditional predictor
    conditional_df = df[df["quant_type"] == "static"]
    mean_values = conditional_df.groupby(["Superclass"])["Coverage"].mean().sort_values()
    order_to_show = list(mean_values.keys())
    sns.catplot(data = df, x = "Superclass", y = "Coverage",
                hue = "quant_type", kind = "box",
                order = list(mean_values.keys()))
    plt.axhline(y = desired_coverage, color = "r", linestyle = "-.",
                linewidth=1)
    plt.show()

    # Now we'll do the miscoverage. Copy the dataframe into a new one
    miscoverage_df = df.copy()
    miscoverage_df.rename(columns = {"Coverage": "Miscoverage"},
                          inplace = True)
    miscoverage_df["Miscoverage"] = 1 - miscoverage_df["Miscoverage"]
    s_mdf = miscoverage_df[miscoverage_df["quant_type"] == "static"]
    mean_vals = s_mdf.groupby(["Superclass"])["Miscoverage"].mean().sort_values()
    sns.catplot(data = miscoverage_df, x = "Superclass", y = "Miscoverage",
                hue = "quant_type", kind = "bar",
                order = list(mean_vals.keys()))
    plt.axhline(y = 1 - desired_coverage, color = "r", linestyle = "-.",
                linewidth = 1)
    plt.show()
    # Now, compute some statistics of the mis-coverages. Construct
    # dataframe with them in it.
    tmp_group = miscoverage_df.groupby(["Superclass", "quant_type"])
    mean_miscoverage = tmp_group["Miscoverage"].mean().reset_index()

def superclass_conditional_experiment(
        pca_dim = len(cifar_processing.superclass_mapping.keys()),
        test_data = None):
    prop_train = .8
    (train_data, val_data) = get_training_data(prop_train)
    training_l2 = 1e-3
    desired_alpha = .1
    num_epochs = 10
    desired_dimension_ratio = pca_dim / len(val_data)
    (pred_model, conditional_quant_model, vanilla_quant_model) = \
        Cfit.train_model_and_conformalize(
            train_data, val_data,
            num_epochs = num_epochs,
            desired_dimension_ratio = desired_dimension_ratio,
            dim_reduction_type = Cfit.DimReductionType.SUPER_CLASSES,
            training_l2 = training_l2,
            alpha_desired = desired_alpha, use_cvx = True)

    if (test_data is None):
        test_data = CiPro.NumpyDataset(
            np.load("cifar100_test_features.npy"),
            np.load("cifar100_test_labels.npy"))
        
    (c_test_correct, test_total, c_coverages) = \
        evaluate_superclass_conditional_coverages(
            test_data, pred_model, conditional_quant_model)
    c_coverages["marginal"] = c_test_correct / test_total
    
    (v_test_correct, test_total, v_coverages) = \
        evaluate_superclass_conditional_coverages(
            test_data, pred_model, vanilla_quant_model)
    v_coverages["marginal"] = v_test_correct / test_total
    
    return (c_coverages, v_coverages)

def evaluate_superclass_conditional_coverages(
        test_data, pred_model, quant_model,
        superclass_mapping = cifar_processing.superclass_mapping):
    """Iterates through the given superclass dictionary and evaluates
    conditional coverage for each superclass.

    """
    s_mapping = cifar_processing.superclass_mapping
    superclass_names = s_mapping.keys()
    coverages = {}
    # Make conformal predictor
    cpred = CM.ConformalPredictor(pred_model, quant_model,
                                  predict_large = True)
    # First, evaluate marginal coverage on the test data
    (test_correct, test_total) = \
        cpred.evaluate_coverage(test_data[:][0], test_data[:][1])

    for s_class in superclass_names:
        print(f"\tEvaluating on class " + s_class)
        filtered = cifar_processing.FilteredDataset(test_data,
                                                    s_mapping[s_class])
        (c_correct, subset_total) = \
            cpred.evaluate_coverage(filtered[:][0], filtered[:][1])
        coverages[s_class] = c_correct / subset_total
    return (test_correct, test_total, coverages)
        
def evaluate_conformal_predictor(test_data, pred_model, quant_model,
                                 *,
                                 proj_matrix = None,
                                 num_random_splits = 10):
    """Evaluates performance of a conformal prediction model

    Evaluates the performance of a conformal prediction model that
    uses quant_model to give scores at which to threshold labels from
    the prediction model.  Assumes that we should return labels with
    *large* scores (i.e., in the ConformalPredictor we use
    predict_large = True).

    Performs num_random_splits of the given test_data and evaluates
    coverage on each of those random splits. To choose the random
    splits, uses a random vector w, and takes the test split to be
    those data points x in test_data satisfying dot(w, x) > 0. To
    choose w, uses one of two methods:
    
      If proj_matrix is non-null, then takes a random v ~ N(0, I) of
      appropriate size, and sets w = proj_matrix @ v

      If proj_matrix is null, simply takes w ~ N(0, I) of appropriate size

    Returns a 4-tuple of the form (test_correct, test_total,
    avg_set_sizes, coverages), where test_correct and test_total are
    the marginal number of correctly covered examples (in test) and
    the total test set size, while avg_set_sizes and coverages are
    num_random_splits vectors of the average predicted confidence set
    size and the coverage fraction, one for each random data split.

    """
    # if (not isinstance(quant_model, CM.SimpleScalarWithProjection)):
    #     raise TypeError("Expected a SimpleScalarWithProjection quantile model"
    #                     + f", but got {type(quant_model)}")
    if (proj_matrix is not None and
        proj_matrix.shape[0] != quant_model.input_dim):
        raise ValueError("Need projection matrix shape to match input dimension"
                         + f" ({quant_model.input_dim}), but got "
                         + f"{proj_matrix.shape[0]}")
    # Make conformal predictor
    cpred = CM.ConformalPredictor(pred_model, quant_model,
                                  predict_large = True)
    # First, evaluate marginal coverage on the test data
    (test_correct, test_total) = \
        cpred.evaluate_coverage(test_data[:][0], test_data[:][1])

    # Returned vectors are the coverage rate and average set size
    avg_set_sizes = np.zeros(num_random_splits)
    coverages = np.zeros(num_random_splits)

    # Let's do a test with the random splits based on the actual
    # projection matrix
    for ii in range(num_random_splits):
        if (proj_matrix is not None):
            v = np.random.randn(proj_matrix.shape[1])
            w = proj_matrix @ v
        else:
            w = np.random.randn(quant_model.input_dim)
        sides = (test_data.data @ w > 0)
        pos_indices = np.flatnonzero(sides)
        subset_test_data = torch.utils.data.Subset(test_data, pos_indices)
        (c_correct, subset_total) = \
            cpred.evaluate_coverage(subset_test_data[:][0],
                                    subset_test_data[:][1])
        s_size_test = cpred.set_sizes(subset_test_data[:][0]).cpu().numpy()
        coverages[ii] = c_correct / subset_total
        avg_set_sizes[ii] = np.mean(s_size_test)
        
    return (test_correct, test_total,
            avg_set_sizes, coverages)

def cover_status_vector(Y_true : torch.tensor,
                        Y_pred_sets : list[torch.tensor]):
    """Computes number of times the true Y belongs to a prediction set
    """
    status_vec = np.zeros(len(Y_true), dtype = np.bool)
    for ii in range(len(Y_true)):
        status_vec[ii] = any(Y_true[ii] == Y_pred_sets[ii])
    return status_vec
    
def evaluate_conformal_along_extremes(
        test_data,
        pred_model : CM.LinearMulticlass,
        *,
        quant_model : CM.SimpleScalarWithProjection = None,
        do_full_conformal : bool = False,
        val_data : Dataset = None,
        direction_vectors = None,
        alpha : float = .1,
        quantile_level = .2):
    """Evaluates conformal prediction along some extreme directions

    Evaluates the coverage rates along the highest and lowest
    quantile_level quantiles of the data, as described by the
    (proj_dim x input_dim) matrix of direction vectors.  In
    particular, for each row w of direction_vectors (i.e., w =
    direction_vectors[i, :]), computes coverage on

      All datapoints in test_data whose x vector is in the upper
      (1 - quantile_level) quantile of dot(x, w)

      All datapoints in test_data whose x vector is in the lower
      quantile_level quantile of dot(x, w)

    Returns the tuple
    
    (test_correct, test_total, low_coverages, high_coverages)

    containing the total number of correctly covered examples and
    total examples (test_correct and test_total) in the test data,
    then vectors of proj_dim length for the coverage and set sizes of
    the low quantiles and high quantiles, respectively.

    """
    cpred = CM.ConformalPredictor(pred_model, quant_model,
                                  predict_large = True)

    # First, evaluate marginal coverage on the test data
    covering_examples = []
    if (do_full_conformal):
        X_val_full = val_data[:][0].cpu().numpy() # n_val x 2048 sized
        X_val = X_val_full @ direction_vectors.T.cpu().numpy()
        score_val_data = CM.LabelScoreDataset(val_data, pred_model)
        scores_val = score_val_data[:][1]
        X_test_full = test_data[:][0].cpu().numpy()  # num_test x 2048
        X_test = X_test_full @ direction_vectors.T.cpu().numpy()
        score_test_data = CM.LabelScoreDataset(test_data, pred_model)
        scores_test = score_test_data[:][1]
        covering_examples = compute_coverages_full_conformal(
            X_val, scores_val, X_test, scores_test, alpha)
    else:
        covering_examples = split_conformal_coverage_vector(
            test_data, pred_model, quant_model)
        
    num_test_correct = sum(covering_examples)
    num_total_test = len(covering_examples)

    number_of_splits = direction_vectors.shape[0]
    XW = test_data[:][0] @ direction_vectors.T

    low_coverages = np.zeros(number_of_splits)
    high_coverages = np.zeros(number_of_splits)
    
    # Filter the dataset by being a low quantile or a high quantile
    for ii in range(number_of_splits):
        q_low = torch.quantile(XW[:, ii], quantile_level)
        q_high = torch.quantile(XW[:, ii], 1 - quantile_level)
        w = direction_vectors[ii, :]

        examples_low = (XW[:, ii] <= q_low).numpy()
        num_low_examples = max(sum(examples_low).item(), 1)
        sum_low_cover = sum(covering_examples[examples_low]).item()
        low_coverages[ii] = sum_low_cover / num_low_examples

        examples_high = (XW[:, ii] >= q_high).numpy()
        num_high_examples = max(sum(examples_high).item(), 1)
        sum_high_cover = sum(covering_examples[examples_high]).item()
        high_coverages[ii] = sum_high_cover / num_high_examples
        
    # Return the marginal coverage data as well as the low and high
    # coverage
    return (num_test_correct, num_total_test,
            low_coverages, high_coverages)

def split_conformal_coverage_vector(test_data : Dataset,
                                    prediction_model : nn.Module,
                                    quant_model : nn.Module):
    cpred = CM.ConformalPredictor(prediction_model, quant_model,
                                  predict_large = True)
    test_pred_sets = cpred.predict(test_data[:][0])
    Y_true = test_data[:][1]
    coverage_vec = np.zeros(len(test_data), dtype = np.bool)
    
    for ii in range(len(Y_true)):
        coverage_vec[ii] = any(Y_true[ii] == test_pred_sets[ii])
    return coverage_vec
    
def full_conformal_coverage_vector(val_data : Dataset,
                                   test_data : Dataset,
                                   prediction_model : nn.Module,
                                   W : np.ndarray,
                                   alpha : float):
    # W should be 2048 x projection dimension
    X_val_full = val_data[:][0].cpu().numpy()  # n_val x 2048
    X_val = X_val_full @ W
    score_val_data = CM.LabelScoreDataset(val_data, prediction_model)
    scores_val = score_val_data[:][1]
    # Get the test data too
    X_test_full = test_data[:][0].cpu().numpy()  # num_test x 2048
    X_test = X_test_full @ W
    score_test_data = CM.LabelScoreDataset(test_data, pred_model)
    scores_test = score_test_data[:][1]
    return compute_coverages_full_conformal(X_val, scores_val,
                                            X_test, scores_test, alpha)

## TODO(jduchi): 1. Add a bias term to this!
## 2. Do the exact predictions perhaps
## 3. Also, probably want to store the marginal coverages in our
##    random directions experiment.

# def compute_coverages_full_conformal(X_cal : np.ndarray,
#                                      scores_cal,
#                                      X_test : np.ndarray,
#                                      scores_test,
#                                      alpha : float):
#     score_fn = lambda x, s : s
#     score_inv_fn = lambda s, x : [-np.inf, s]
#     def Phi_function(x):
#         if (x.ndim == 1):
#             return np.concatenate((x, [1]))
#         return np.concatenate((x, np.ones((x.shape[0], 1))), axis = 1)
    
#     cond_conf = CondConf(score_fn, Phi_function, infinite_params = {})
#     cond_conf.setup_problem(X_cal, scores_cal)
#     coverage_booleans = np.zeros(X_test.shape[0], np.bool)
#     for ii in tqdm(range(len(coverage_booleans))):
#         s_pred = cond_conf.predict(1 - alpha, X_test[ii], score_inv_fn,
#                                    exact = True, randomize = True)
#         coverage_booleans[ii] = (s_pred >= scores_test[ii])
#     return coverage_booleans
    
def compute_coverages_full_conformal(X_cal : np.ndarray,
                                     scores_cal,
                                     X_test : np.ndarray,
                                     scores_test,
                                     alpha : float):
    """Uses the Gibbs et al. full conformal method to compute coverage

    Returns a len(X_test) boolean array coverage_booleans with
    indicators of whether the full conditional conformal approach
    covers.  coverage_booleans[i] is True if the ith test example is
    covered, False otherwise.

    Assumes as input matrices X_cal and X_test, which are already
    featurized appropriately, and score vectors for both the calibration
    and test sets.

    """
    coverage_booleans = np.zeros(X_test.shape[0], np.bool)

    X_cal = np.concatenate((X_cal, np.ones((X_cal.shape[0], 1))), axis = 1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis = 1)
    for ii in tqdm(range(len(coverage_booleans))):
        prob = setup_cvx_problem_calib(
            1 - alpha, None,
            np.concatenate((scores_cal, np.array([scores_test[ii]]))),
            np.vstack((X_cal, X_test[ii,:])),
            {})
        prob.solve(solver = "MOSEK")
        coverage_booleans[ii] = \
            (scores_test[ii] <= X_test[ii, :] @ prob.constraints[2].dual_value)
    return coverage_booleans

