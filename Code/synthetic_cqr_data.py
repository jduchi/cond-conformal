# synthetic_cqr_data
#
# Helper methods for generating synthetic conditional (quantile)
# regression data for conformal prediction

import numpy as np

def generate_sinusoid_function(seed : int = 0):
    """Returns parameters for sinusoidal function

    Returns the parameters (theta_0, theta_1, phi_0, phi_1) for a function
    of the form

    f(x) = theta_0 * cos(phi_0 * x) + theta_1 * sin(phi_1 * x)
    """
    rng = np.random.default_rng(seed)
    (theta_0, theta_1) = (2 * rng.random(2) - 1)
    (phi_0, phi_1) = (3.75 * np.pi + np.pi / 4) * rng.random(2)
    return (theta_0, theta_1, phi_0, phi_1)

def generate_sinusoid_data(seed : int = 0, n_val : int = 100,
                           n_test : int = 500, n_train : int = 200,
                           n_disc : int = 5):
    """Generate random sinusoidal data.

    Implements the regression function

      E[Y | x] = f(x) = theta_0 * cos(phi_0 * x) + theta_1 * sin(phi_1 * x),

    where phi_0 and phi_1 are random and uniform in [pi / 4, 4 * pi], and
    theta_0 and theta_1 are Uniform[-1, 1] random variables.

    The distribution of Y - f(x) depends on the position of x, where
    x is evenly distributed in [0, 1]. At a point x, we take

     Y(x) = f(x) + (1 / lambda_0(x)) * Exp(1)   w.p. p(x)
     Y(x) = f(x) - (1 / lambda_1(x)) * Exp(1)   w.p. 1 - p(x)

    so that Y has the correct mean but different upper and lower tails.
    We discretize the values of x into n_disc equally spaced bins
    (i.e., [0, 1 / n_disc), [1 / n_disc, 2 / n_distc), etc.).
    The values of lambda_0(x) and lambda_1(x) are chosen based on
    the bin index into which x falls, where if x falls in bin i, then

      lambda_0(x) = 1 / np.abs(cos(i))

      lambda_1(x) = 1 / np.abs(sin(i))

    and, accordingly, p = lambda_0 / (lambda_1 + lambda_0).
    
    """
    (theta_0, theta_1, phi_0, phi_1) = generate_sinusoid_function(seed)
    rng = np.random.default_rng(seed)
    x_train = np.linspace(1 / (2 * n_train), 1 - 1 / (2 * n_train), n_train)
    x_test = np.linspace(1 / (2 * n_test), 1 - 1 / (2 * n_test), n_test)
    x_val = np.linspace(1 / (2 * n_val), 1 - 1 / (2 * n_val), n_val)

    discretizations = np.arange(0, 1, 1 / n_disc)
    bins_train = _bin_vector(x_train, discretizations)
    bins_test = _bin_vector(x_test, discretizations)
    bins_val = _bin_vector(x_val, discretizations)
    lambda_0 = np.exp(np.linspace(-3, 0, n_disc))
    lambda_1 = np.exp(np.linspace(-4, -1, n_disc))
    # lambda_0 = 1 / (4 * np.abs(np.cos(np.linspace(1, 2 * n_disc, n_disc))))
    # lambda_1 = 1 / (4 * np.abs(np.sin(np.linspace(1, 2 * n_disc, n_disc))))
    
    y_train = _generate_sinusoid_data(
        theta_0, theta_1, phi_0, phi_1, x_train, bins_train,
        lambda_0, lambda_1, rng)
    y_test = _generate_sinusoid_data(
        theta_0, theta_1, phi_0, phi_1, x_test, bins_test,
        lambda_0, lambda_1, rng)
    y_val = _generate_sinusoid_data(
        theta_0, theta_1, phi_0, phi_1, x_val, bins_val,
        lambda_0, lambda_1, rng)
    return (x_train.reshape(n_train, 1), y_train,
            x_val.reshape(n_val, 1), y_val,
            x_test.reshape(n_test, 1), y_test)

def _generate_sinusoid_data(theta_0, theta_1, phi_0, phi_1,
                            x : np.ndarray,
                            bins : np.ndarray,
                            lambda_0 : np.ndarray, lambda_1 : np.ndarray,
                            rng : np.random.Generator):
    """Generates sinusoidal data

    Helper method for actually generating sinusoidal data as in the
    method generate_sinusoid_data.

    """
    nn = len(x)
    f_vals = (theta_0 * np.cos(phi_0 * x)
               + theta_1 * np.sin(phi_1 * x))
    exp_train = rng.exponential(size = nn)
    lambda_by_bin_0 = lambda_0[bins]
    lambda_by_bin_1 = lambda_1[bins]
    p_sides = lambda_by_bin_0 / (lambda_by_bin_0 + lambda_by_bin_1)
    flips = rng.random(nn) < p_sides[bins]  # When true, take lambda_1
    y = np.zeros(nn)
    y[~flips] = f_vals[~flips] + lambda_by_bin_0[~flips] * exp_train[~flips]
    y[flips] = f_vals[flips] - lambda_by_bin_1[flips] * exp_train[flips]
    return y
    
def _bin_vector(x : np.ndarray, discretizations : np.ndarray):
    """Computes vector of which bins elements of x belong to

    Assuming x is an n-vector, returns an n-vector b whose ith element
    is

    b[i] = j

    if x[i] belongs to bin j, as defined by discretizations. Here,
    bin j is the half-open interval

      [discretizations[j], discretizations[j + 1])

    where the last interval has endpoint +infty.

    """
    comparisons = x[:, np.newaxis] >= discretizations
    bins = np.sum(comparisons, 1) - 1
    return bins
                               
