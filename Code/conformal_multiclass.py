# conformal_multiclass.py
#
# Code to do multiclass classification (typically on CIFAR-100) using
# a pretrained 50 layer ResNet. Note: this is *research* code, so do
# not expect it to be particularly robust.
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

# Include the processing file
import cifar_processing as CiPro
import pdb

class LinearMulticlass(nn.Module):
    """Constructs a single layer multiclass classifier

    model = LinearMulticlass(dim_in, num_classes) constructs a
    multiclass classifier with input size dim_in and output number of
    classes num_classes.

    """

    def __init__(self, dim_in = 2048, num_classes = 100):
        super().__init__()
        self.fc = nn.Linear(dim_in, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten inputs
        x = self.fc(x)
        return x

class RandomLinear(nn.Module):
    """A random linear layer
    """
    def __init__(self, dim_in, dim_out):
        super(RandomLinear, self).__init__()
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.projection_matrix = \
            nn.Parameter(torch.randn(dim_in, dim_out), requires_grad = False)
        
    def forward(self, x):
        # Simply applies the random projection
        return x @ self.projection_matrix

class SimpleScalarWithProjection(nn.Module):
    """A simple 1 layer NN with a (random) projection to reduce dimensionality

    model = SimpleScalarWithProjection(dim_in, dim_proj, W = None)
    constructs a neural network module with an input dimension dim_in,
    then implements a foward pass

      f(x) = theta' * W * x.

    If W is None, then sets W to be a random i.i.d. N(0, 1) (Gaussian)
    matrix of size (dim_proj x dim_in), so that it reduces
    dimensionality. Otherwise, so long as W is of the correct
    size, takes the argument W as a parameter.

    """

    def __init__(self, dim_in, dim_proj, W = None):
        super().__init__()
        self.input_dim = dim_in
        self.proj_dim = dim_proj
        self.projection_layer = RandomLinear(dim_in, dim_proj)
        if (W is not None):
            if (W.shape[0] != dim_proj or W.shape[1] != dim_in):
                raise ValueError(
                    f"Expected W to be shape ({dim_proj} x {dim_in})" +
                    f" but got ({W.shape[0]} x {W.shape[1]})")
            # Now we replace the random linear layer's projection
            # matrix with W.
            self.projection_layer.projection_matrix = nn.Parameter(
                torch.tensor(W.T).to(torch.float) if isinstance(W, np.ndarray)
                else W.T,
                requires_grad = False)
            
        self.fc = nn.Linear(self.proj_dim, 1, bias = True)

    def forward(self, x):
        x = self.projection_layer(x)
        return self.fc(x)

class LabelScoreDataset(Dataset):
    """Transforms a given dataset into a duplicate with new labels

    DS = LabelScoreDataset(dataset, model) returns a new Dataset
    object whose labels are all given by applying the model to the
    data in dataset, then taking the target value to be the models
    output value at the correct label. That is, if

    f(x)

    is the output of the model on input x, the new transformed label
    on example (x, y) is

    f(x)[y]

    the yth output.

    """
    
    def __init__(self, dataset, model):
        self.data = dataset
        self.transformed_labels = np.zeros(len(dataset))

        device = torch.device("cpu")
        model.eval()
        # Now compute transformed scores
        for idx in range(len(self.data)):
            (x, y) = self.data[idx]
            outputs = model(x.view(1, -1).to(device))
            self.transformed_labels[idx] = outputs.squeeze()[y].item()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, _ = self.data[idx]
        ty = self.transformed_labels[idx]
        return (x, ty)

class LogisticScoreDataset(Dataset):
    """Transforms a given dataset into a duplicate with new labels

    DS = LogisticScoreDataset(dataset, model) returns a new Dataset
    object whose labels are all given by applying the model to the
    data in dataset, then computing its logistic loss.

    """

    def __init__(self, dataset, model):
        self.data = dataset
        self.transformed_labels = np.zeros(len(dataset))

        device = torch.device("cpu")
        model.eval()
        criterion = nn.crossentropyloss()
        def label_transform(x, y):
            # it's possible these transformations are so brittle that
            # they only work with a linearmulticlass model (see
            # above), but i won't worry about that for now. here, we
            # flatten x to make sure it's the currect size.
            x = x.view(1, -1)
            y = y.view(1)
            (x, y) = (x.to(device), y.to(device))
            outputs = model(x)
            loss = criterion(outputs, y)
            return loss
        
        # Now compute transformed scores
        for idx in range(len(self.data)):
            (x, y) = self.data[idx]
            transform_y = label_transform(x, y)
            self.transformed_labels[idx] = transform_y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, _ = self.data[idx]
        ty = self.transformed_labels[idx]
        return (x, ty)

class QuantileLoss(nn.Module):
    """Implements the quantile loss

    The quantile loss in this case is defined to be

    loss(y_pred, y_true) = alpha * max(y_pred - y_true, 0) +
                           (1 - alpha) * max(y_true - y_pred, 0)

    Note that this means minimizers of the loss should be predicting
    the (1 - alpha) quantile: there is less penalty for predicting
    over than under.

    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha  # alpha value in (0, 1)

    def forward(self, y_pred, y_true):
        # Ensure y_pred and y_true are of the same shape
        assert y_pred.squeeze().shape == y_true.squeeze().shape, \
            ("Predictions and targets must have the same shape, got " +
             f"{y_pred.shape} and {y_true.shape}")

        # Calculate the loss
        errors = y_pred.squeeze() - y_true.squeeze()
        # Set loss to be alpha or (1 - alpha) as the error is positive
        # or negative.
        loss = torch.where(errors >= 0,
                           self.alpha * errors, 
                           -(1 - self.alpha) * errors)
        
        return loss.mean()  # Return mean loss

class ConformalPredictor():
    """A prediction method that returns plausible labels

    cp = ConformalPredictor(pred_model, scoring_model, predict_large = False)

    constructs a ConformalPredictor object, which provides a predict
    method that takes as input a PyTorch tensor, then returns a list
    of plausible labels for each example in the input tensor.

    On a single input example x, the two input models (pred_model and
    scoring_model) are expected to output, respectively, num_classes
    scores for pred_model and a single score for the scoring_model.

    """
    def __init__(self, pred_model, scoring_model, predict_large = False):
        self.pred_model = pred_model
        self.scoring_model = scoring_model
        self.predict_large_examples = predict_large
        
    def predict(self, X) -> torch.tensor:
        """Returns set of plausible labels

        predict(X) returns a list of plausible labels according to the
        internally stored scoring model. Assuming X is of size
        (batch_size x input_dim), for each individual example x in X,
        returns the set of labels y satisfying

          s(x, y) <= scoring_model(x)

        if predict_large_examples is False (the default), while it returns
        the labels y satisfying

          s(x, y) >= scoring_model(x)

        if predict_large_examples is True.

        The returned format is as a list of lists, i.e.,

        Y = [[y_{11}, y_{12}, ..., y_{1k}], ...]

        with one row for each example in X and different lengths of rows.

        """
        Y_inds = self._get_Y_inds(X)
        batch_size = Y_inds.shape[0]
        plausible_Y = [[]] * batch_size
        for ii in range(batch_size):
            plausible_Y[ii] = torch.nonzero(Y_inds[ii]).flatten()
        return plausible_Y

    def _get_Y_inds(self, X):
        outputs = self.pred_model(X)  # Size (batch_size x output_dim)
        batch_size = outputs.shape[0]
        # Construct matrix of elements that are larger/smaller than
        # the score per example
        score_per_example = self.scoring_model(X)
        Y_inds = ((score_per_example <= outputs)
                  if self.predict_large_examples else
                  (score_per_example >= outputs))
        return Y_inds

    def set_sizes(self, X):
        """Returns prediction set sizes on input batch X
        """
        Y_inds = self._get_Y_inds(X)
        return Y_inds.sum(1)
    
    def evaluate_coverage(self, X, Y):
        """Evaluates the coverage of the model on a set of examples

        (correct, batch_size) = evaluate_coverage(X, Y)

        returns the number of examples for which the true label Y[i]
        is in the set of predicted plausible Y (correct) and the total
        number of examples in the batch (batch_size).

        """
        plausible_Y = self.predict(X)
        # Unfortunately, because the set of plausible Y has different sizes,
        # we have to loop here
        correct = 0
        for ii in range(len(plausible_Y)):
            correct += (plausible_Y[ii] == Y[ii]).sum().item()
        return (correct, len(plausible_Y))
        
