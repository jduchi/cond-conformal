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
import cvxpy as cvx
import warnings
from enum import Enum

import pdb

from importlib import reload

# Include the processing file and class files
import cifar_processing as CiPro
import conformal_multiclass as CM

class DimReductionType(Enum):
    """Type of dimension reduction to apply for conformalization

    The currently supported types are

      PCA: Perform principal components to find the best directions

      RANDOM: Choose a random dimension reduction matrix with
      i.i.d. N(0, 1) entries

      LABEL_CORRELATED: Take the dimension reduction matrix to
      correspond to directions most correlated with given sets of
      labels.  That is, if Y = [group_1, ..., group_k] consists of k
      groups of labels, the dimension reduction matrix W will be of
      size (k x input_dim), and row i of W will be

        W[i, :] = (mu_i - mu) / norm(mu_i - mu),

      where mu is the mean of the covariates across all the data and
      mu_i the mean of the covariates on examples whose label belongs
      to group_i.

      SUPER_CLASSES: Take the dimension matrix by first fitting a
      prediction model to a set of superclasses, then set W[i, :] to
      be the predictor associated with that class.

    """
    RANDOM = 1
    PCA = 2
    LABEL_CORRELATED = 3
    SUPER_CLASSES = 4

def train_model_and_conformalize(train_data, val_data,
                                 *,
                                 num_epochs = 10,
                                 desired_dimension_ratio = .01,
                                 dim_reduction_type = DimReductionType.RANDOM,
                                 training_l2 = .001,
                                 alpha_desired = .1,
                                 use_cvx = True):
    """Runs a single experiment of training a model and producing a
    conformal set predictor

    Fits a (linear) multiclass prediction model on the CIFAR-100
    dataset using the train_data, using the val_data as a validation
    dataset for constructing a conformal predictor.  The prediction
    model uses L2 regularization with multiplier training_l2, and the
    conformal predictor targets coverage level (1 - alpha).

    The conformal quantile predictor predicts quantiles with a
    projected linear model, which projects down to dimension
    desired_dimension_ratio * size of validation set
    (see the method train_quantile_model). If pca_quantile is True,
    uses principal vectors from the training data as the projection
    matrix, otherwise uses a random matrix.

    Returns the 3-tuple

      (pred_model, conditional_quant_model, vanilla_quant_model)

    where pred_model is the fit predictive model, conditional_quant_model
    predicts conditional quantiles, and vanilla_quant_model predicts
    the static quantile of scores from pred_model.

    """

    # Now, fit the model
    print("*** Training Prediction Model ***")
    pred_model = train_prediction_model(
        train_data,
        num_epochs = num_epochs,
        init_stepsize = .2,
        l2_regularization = training_l2,
        print_every = -1)
    (top_1, top_5) = evaluate(val_data, pred_model,
                                      torch.device("cpu"), k = 5)
    print(f"\tValidation accuracy: {top_1:.2f}, top 5: {top_5:.2f}")

    # Fit the "conditional" conformal predictor
    val_size = len(val_data)
    assert (desired_dimension_ratio < .2 and desired_dimension_ratio >= 0), \
        "Must have 0 <= dimension ratio < .2"
    proj_dim = round(val_size * desired_dimension_ratio)
    W = find_dim_reduction_matrix(train_data, input_dim = 2048,
                                  reduction_type = dim_reduction_type,
                                  class_map = CiPro.superclass_mapping,
                                  proj_dim = proj_dim)
                                  
    alpha = ((alpha_desired - proj_dim / (2 * val_size))
             / (1 - proj_dim / val_size))
    alpha = max(alpha, 0)
    score_data = CM.LabelScoreDataset(val_data, pred_model)
    big_num_epochs = round(max(4, desired_dimension_ratio * val_size / 10) *
                           num_epochs)

    print(f"*** Training Quantile Models ***")
    if (use_cvx):
        c_quant_model = quantile_from_scored_label_data(
            score_data, W = W, alpha = alpha, proj_dim = proj_dim)
    else:
        c_quant_model = train_quantile_model(
            score_data, num_epochs = big_num_epochs,
            proj_dim = proj_dim,
            init_stepsize = .2, print_every = 100,
            predict_large = True,
            alpha = alpha, W = W)

    v_quant_model = train_quantile_model(
        score_data, num_epochs = 1,
        proj_dim = 0,
        init_stepsize = .2, print_every = -1,
        predict_large = True, alpha = alpha_desired)
        
    return (pred_model, c_quant_model, v_quant_model)

def train_quantile_model(dataset, *,
                         proj_dim = 10,
                         num_epochs = 20, print_every = 100,
                         init_stepsize = .02, predict_large = False,
                         alpha = .1,
                         W = None,
                         in_dimension = 2048):
    """Fits linear model to predict particular quantiles of dataset

    Assuming dataset has targets (responses) that are scalars, fits a
    linear model (SimpleScalarWithProjection) to predict either the
    alpha-largest fraction (if predict_large is True) or
    alpha-smallest fraction of the responses in the data.
    The model fit is of the form

      f(x) = theta' * W * x

    where W is a particular projection-like matrix.  If W is None,
    then it is a random (proj_dim x input_dim) matrix, as in the
    SimpleScalarWithProjection module. Otherwise, the argument is
    passed in.

    Fits model using AdaGrad with given initial stepsize for
    num_epochs epochs.

    """
    model = CM.SimpleScalarWithProjection(in_dimension, proj_dim, W)
    # Let's just fit on the CPU because it doesn't matter much.
    device = torch.device("cpu")

    train_loader = DataLoader(dataset, batch_size = 64, shuffle = True)
    criterion = CM.QuantileLoss((1 - alpha) if predict_large else alpha)
    optimizer = optim.Adagrad(model.parameters(), lr = init_stepsize)
    for epoch in range(num_epochs):
        single_training_epoch(model, criterion, optimizer, train_loader,
                              device,
                              epoch = epoch, print_every = print_every)

    # Aggressively make sure the bias term is correct
    fix_final_bias(dataset, model, alpha = alpha, predict_large = predict_large)
    return model

def fix_final_bias(dataset, model, alpha = .1, predict_large = False):
    """Fixes the final bias level in the model to force it to have the
    desired quantile

    """
    s_pred = model(dataset[:][0]) - model.fc.bias
    s_actual = dataset[:][1]
    diff = s_pred.flatten().detach() - s_actual
    if (predict_large):
        with torch.no_grad():
            model.fc.bias.copy_(-torch.quantile(diff, 1 - alpha))
    else:
        with torch.no_grad():
            model.fc.bias.copy_(-torch.quantile(diff, alpha))

def single_training_epoch(model, criterion, optimizer, data_loader, device,
                          epoch = 0, print_every = 100):
    """Performs a single epoch (looping through all data) of training
    """
    running_loss = 0.0
    total_steps = 0
    model.train()
    for (x, y) in data_loader:
        (x, y) = (x.to(device), y.to(device))
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_steps += 1
        if (print_every > 0 and total_steps % print_every == 0):
            print(f"\tEpoch [{epoch + 1}], " +
                  f"Average Loss: {running_loss / total_steps:.4f}")
    
def train_prediction_model(train_data, *,
                           num_epochs = 10, print_every = 100,
                           init_stepsize = .01,
                           l2_regularization = .01,
                           in_dimension = 2048,
                           num_classes = 100):
    """Fits a LinearMulticlass model on the given training data

    Uses Adagrad to fit a LinearMulticlass model, which the method
    returns, on the provided training data (train_data). Iterates for
    num_epochs epochs, printing a status update every print_every
    steps of the individual training epochs. (Note that because we use
    batches of size 64 within the optimizer, this may mean few
    printouts)

    """
    model = CM.LinearMulticlass(in_dimension, num_classes)
    # Let's just fit on the CPU because it doesn't matter much.
    device = torch.device("cpu")
    l2_regularization = max(0, l2_regularization)
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters(), lr = init_stepsize,
                              weight_decay = l2_regularization)
    running_loss = 0.0
    total_steps = 0
    for epoch in range(num_epochs):
        single_training_epoch(model, criterion, optimizer, train_loader,
                              device, epoch, print_every = print_every)
        (train_1_acc, train_5_acc) = evaluate(train_data, model, device, k = 5)
        print(f"Train accuracy: {train_1_acc:.2f}, top 5: {train_5_acc:.2f}")
    return model

def evaluate(dataset, model, device, k = 5):
    """Returns the top-k accuracy of the given model on the given dataset.
    """
    model.eval()
    top_1_correct = 0
    top_k_correct = 0
    total = 0
    data_loader = DataLoader(dataset, batch_size = 64, shuffle = False)
    with torch.no_grad():
        for (xs, ys) in data_loader:
            xs = xs.to(device)
            ys = ys.to(device)
            outputs = model(xs) # shape is (batch_size x num_classes)
            _, predicted = torch.max(outputs.data, 1)
            top_1_correct += (predicted == ys).sum().item()
            _, predicted_inds = torch.topk(outputs, k = k, dim = 1)
            # Compare targets with the top-5 indices
            top_k_correct += \
                (predicted_inds == ys.view(-1, 1)).sum().item() 
            total += ys.size(0)

    top_1_accuracy = 100 * top_1_correct / total
    top_k_accuracy = 100 * top_k_correct / total
    return (top_1_accuracy, top_k_accuracy)

def find_dim_reduction_matrix(dataset, *,
                              reduction_type = DimReductionType.RANDOM,
                              input_dim = 2048,
                              **kwargs):
    """Finds dimension reduction matrix

    Returns a (k x input_dim) matrix W to be used for reducing dimensions
    in a (linear) prediction problem. Depending on reduction_type,
    the following W are returned:

    RANDOM: Requires argument k = proj_dim, and W has i.i.d. N(0, 1) entries

    PCA: Requires argument k = proj_dim, and takes W to be the first k
    principal components of the covariates in the data.

    LABEL_CORRELATED: Requires argument class_map, a dictionary whose
    entries consist of arrays of classes of interest. Returns W whose
    row i is chosen as indicated in DimReductionType.

    """
    W = None
    if (reduction_type == DimReductionType.RANDOM):
        proj_dim = kwargs["proj_dim"]
        W = np.random.randn(proj_dim, input_dim)
    elif (reduction_type == DimReductionType.SUPER_CLASSES):
        print(f"Finding covariate directions by fitting model to superclasses")
        if (kwargs.get("class_map") is None):
            raise ValueError("Expected argument class_map")
        class_mapping = kwargs["class_map"]
        class_filtered_data = CiPro.CollapsedLabelDataset(dataset,
                                                          class_mapping)
        filtered_pred_model = train_prediction_model(
            class_filtered_data, num_epochs = 5,
            init_stepsize = .2, l2_regularization = 1e-6, print_every = -1,
            num_classes = class_filtered_data.num_classes)
        # W should be size k-by-input dimension, where k is the number
        # of superclasses
        W = filtered_pred_model.fc.weight.detach()
    elif (reduction_type == DimReductionType.LABEL_CORRELATED):
        # Now, we ignore the projection dimension, but find the
        # directions correlated to the label groups
        print(f"Finding covariate directions correlated to classes")
        if (kwargs.get("class_map") is None):
            raise ValueError("Expected argument class_map")
        class_mapping = kwargs["class_map"]
        W = torch.zeros((len(class_mapping), input_dim))
        ind = 0
        mean_X = torch.mean(dataset[:][0], 0)
        # Could detach the data here or later
        # mean_X = np.mean(dataset[:][0].detach().cpu().numpy(), 0)
        for (key, class_list) in class_mapping.items():
            filtered_dataset = CiPro.FilteredDataset(dataset, class_list)
            w = torch.mean(filtered_dataset[:][0], 0) - mean_X
            w = w / torch.norm(w)
            W[ind, :] = w.detach() # .numpy()
            ind += 1
    elif (reduction_type == DimReductionType.PCA):
        proj_dim = kwargs["proj_dim"]
        print(f"Performing PCA to get {proj_dim} vectors to project")
        W = principal_vectors(dataset, num_pcs = proj_dim)
        W = W.T  # Make W have size (k x input_dim)
    else:
        raise ValueError(f"Unknown dimension reduction type {reduction_type}")
    return W

def principal_vectors(dataset, num_pcs = 5):
    """Returns the first num_pcs principal components of the data

    V = principal_vectors(dataset, num_pcs) returns the first num_pcs
    principal components of the data in dataset. Assumes that
    dataset[:][0] contains the X matrix (n x d) of the data, where d
    is the desired projection dimension.

    """
    X = dataset[:][0]
    # Copy it, but whatever
    centered_data = X - X.mean(0)
    cov_mat = torch.mm(centered_data.T, centered_data) / X.shape[0]
    (evals, evecs) = torch.linalg.eig(cov_mat)
    # Just take real parts
    evals = evals.real
    evecs = evecs.real
    sorted_indices = torch.argsort(evals, descending=True)
    sorted_eigenvalues = evals[sorted_indices]
    num_pcs = min(num_pcs, evecs.shape[1])
    return evecs[:, sorted_indices[0:num_pcs]]

# Would like to be able to pull features out of dataset and directly
# fit a quantile regression on it using CVX.

def quantile_from_scored_label_data(dataset, *,
                                    proj_dim = 10,
                                    alpha = .1,
                                    W = None,
                                    in_dimension = 2048):
    """Fits quantile regression model from LabelScoreDataset

    Uses cvxpy to fit a quantile regression model directly from a
    score dataset. If W is specified (non-null), uses it as a
    projection to transform the covariate data in the dataset first.

    """
    if (W is None):
        W = np.random.randn(proj_dim, in_dimension)
    if (isinstance(W, torch.Tensor)):
        W = W.detach().cpu().numpy()

    X = dataset.data[:][0].detach().cpu().numpy() @ W.T
    y = dataset.transformed_labels
    (theta, bias) = fit_quantile_regression(X, y, 1 - alpha, fit_bias = True)
    proj_dim = W.shape[0]
    in_dimension = W.shape[1]
    model = CM.SimpleScalarWithProjection(in_dimension, proj_dim, W = W)
    # Set weights and bias appropriately. We should have
    # model.fc.weight be a (1 x proj_dim) Tensor, while model.fc.bias
    # should be a length 1 Tensor.
    model.fc.weight.data[:] = torch.tensor(theta)[:]
    model.fc.bias.data[:] = torch.tensor(bias)
    return model
    
def fit_quantile_regression(X, y, alpha, fit_bias = True):
    """Fits a quantile regression model using CVX

    theta = fit_quantile_regression(X, y, alpha) fits theta to solve

      min. mean(L_alpha(X * theta - y)),

    where X is an (n x d) data matrix, y is a vector of scalars, and

      L_alpha(t) = alpha * max(t, 0) + (1 - alpha) * max(-t, 0)

    is the quantile loss, which attempts to fit a predictor of the form

      f(x) = theta' * x

    to predict the (1 - alpha) quantile of y given x. (So when alpha
    is small, predicts large numbers.)

    If fit_bias is true, includes a bias term in the above model and
    returns the pair (theta, intercept), where intercept is a scalar
    and predictions take the form

      f(x) = theta' * x + intercept.

    """
    nn = X.shape[0]
    dd = X.shape[1]
    theta = cvx.Variable(dd)
    t = cvx.Variable(nn)
    s = cvx.Variable(nn)
    bias = cvx.Variable()

    all_ones = np.ones(nn)
    
    objective = cvx.Minimize(
        alpha * sum(t) + (1 - alpha) * sum(s))
    if (fit_bias):
        constraints = [t >= X @ theta + bias * all_ones - y,
                       t >= 0,
                       s >= y - X @ theta - bias * all_ones,
                       s >= 0]
    else:
        constraints = [t >= X @ theta - y,
                       t >= 0,
                       s >= y - X @ theta,
                       s >= 0]        
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver = cvx.MOSEK)

    if (problem.status == cvx.OPTIMAL):
        if (fit_bias):
            return (theta.value, bias.value)
        else:
            return theta.value
    else:
        warnings.warn(f"Problem not solved: status{problem.status}" +
                      ". Returning all zeros vector")
        return np.zeros(nn)
