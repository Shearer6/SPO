import numpy as np
from typing import Tuple, Callable, Optional, Union
from scipy.stats import norm, uniform, bernoulli
import gurobipy as gp
from gurobipy import GRB

# Utility function to replace Julia's eye (identity matrix)
def eye(n: int) -> np.ndarray:
    return np.eye(n)


# Least squares solver
def cost_least_squares(X: np.ndarray, c: np.ndarray, intercept: bool = False) -> np.ndarray:
    """
    Solve a least-squares problem to fit a d x p matrix B such that c approx == B*X
    """
    p, n = X.shape
    d, n2 = c.shape

    if n != n2:
        raise ValueError("dimensions are mismatched")

    if intercept:
        X = np.vstack([X, np.ones((1, n))])

    Xt = X.T
    ct = c.T

    # Solve Xt @ B.T = ct using NumPy's least squares solver
    Bt, _, _, _ = np.linalg.lstsq(Xt, ct, rcond=None)

    return Bt.T


# Ridge regression solver
def ridge(X: np.ndarray, c: np.ndarray, reg_param: float) -> np.ndarray:
    """
    Solve a ridge regression problem min_B (1/n)0.5*||c - B*X||^2 + 0.5*reg_param*||B||^2
    """
    p, n = X.shape
    d, n2 = c.shape

    if n != n2:
        raise ValueError("dimensions are mismatched")

    Xt = X.T
    ct = c.T

    # Bt = (X @ Xt + n * reg_param * I)^{-1} @ (X @ ct)
    Bt = np.linalg.inv(X @ Xt + n * reg_param * eye(p)) @ (X @ ct)

    return Bt.T


# Oracle dataset processing
def oracle_dataset(c: np.ndarray, oracle: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the optimization oracle to each column of the d x n matrix c
    """
    d, n = c.shape
    z_star_data = np.zeros(n)
    w_star_data = np.zeros((d, n))

    for i in range(n):
        z_i, w_i = oracle(c[:, i])
        z_star_data[i] = z_i
        w_star_data[:, i] = w_i

    return z_star_data, w_star_data


# SPO loss computation
def spo_loss(B_new: np.ndarray, X: np.ndarray, c: np.ndarray, oracle: Callable,
             z_star: Optional[np.ndarray] = None) -> float:
    """
    Compute the SPO loss of B_new on the training/holdout/test set (X, c)
    """
    if z_star is None:
        z_star, _ = oracle_dataset(c, oracle)

    n = len(z_star)
    spo_sum = 0

    for i in range(n):
        c_hat = B_new @ X[:, i]
        z_oracle, w_oracle = oracle(c_hat)
        spo_loss_cur = np.dot(c[:, i], w_oracle) - z_star[i]
        spo_sum += spo_loss_cur

    return spo_sum / n


# Compare models by percentage
def compare_models_percent(B_modA: np.ndarray, B_modB: np.ndarray, X: np.ndarray,
                           c: np.ndarray, oracle: Callable, eps: float = 0.000001) -> float:
    """
    Compute the percentage of times modA produces a solution with smaller true cost than modB
    """
    d, n = c.shape
    count = 0

    for i in range(n):
        c_hatA = B_modA @ X[:, i]
        c_hatB = B_modB @ X[:, i]
        _, w_oracleA = oracle(c_hatA)
        _, w_oracleB = oracle(c_hatB)

        diff = np.dot(c[:, i], w_oracleA) - np.dot(c[:, i], w_oracleB)
        if diff <= eps:
            count += 1

    return count / n


# SPO+ loss computation
def spo_plus_loss(B_new: np.ndarray, X: np.ndarray, c: np.ndarray, oracle: Callable,
                  z_star: Optional[np.ndarray] = None, w_star: Optional[np.ndarray] = None) -> float:
    """
    Compute the SPO plus loss of B_new on the training/holdout/test set (X, c)
    """
    if z_star is None or w_star is None:
        z_star, w_star = oracle_dataset(c, oracle)

    n = len(z_star)
    spo_plus_sum = 0

    for i in range(n):
        c_hat = B_new @ X[:, i]
        spoplus_cost_vec = 2 * c_hat - c[:, i]
        z_oracle, w_oracle = oracle(spoplus_cost_vec)
        spo_plus_cost = -z_oracle + 2 * np.dot(c_hat, w_star[:, i]) - z_star[i]
        spo_plus_sum += spo_plus_cost

    return spo_plus_sum / n


# Least squares loss
def least_squares_loss(B_new: np.ndarray, X: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the least squares loss of B_new on the training/holdout/test set (X, c)
    """
    p, n = X.shape
    residuals = B_new @ X - c
    error = (1 / n) * (1 / 2) * (np.linalg.norm(residuals) ** 2)
    return error


# Absolute (L1) loss
def absolute_loss(B_new: np.ndarray, X: np.ndarray, c: np.ndarray) -> float:
    """
    Compute the absolute (L1) loss
    """
    p, n = X.shape
    residuals = B_new @ X - c
    error = (1 / n) * np.sum(np.abs(residuals))
    return error


# Huber loss
def huber_loss(B_new: np.ndarray, X: np.ndarray, c: np.ndarray, delta: float) -> float:
    """
    Compute the Huber loss with parameter delta
    """
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimension mismatch in Huber loss calculation.")

    residuals = B_new @ X - c
    total_error = 0

    for i in range(n):
        for j in range(d):
            a_res = abs(residuals[j, i])
            if a_res <= delta:
                cur_error = 0.5 * a_res ** 2
            else:
                cur_error = delta * (a_res - 0.5 * delta)
            total_error += cur_error

    return (1 / n) * total_error


# Polynomial kernel data generation
def generate_poly_kernel_data(B_true: np.ndarray, n: int, degree: int,
                              inner_constant: float = 1, outer_constant: float = 1,
                              kernel_damp_normalize: bool = True, kernel_damp_factor: float = 1,
                              noise: bool = True, noise_half_width: float = 0,
                              normalize_c: bool = True, normalize_small_threshold: float = 0.0001) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Generate (X, c) from the polynomial kernel model
    """
    d, p = B_true.shape
    X_observed = np.random.randn(p, n)
    dot_prods = B_true @ X_observed

    c_observed = np.zeros((d, n))
    for j in range(d):
        if kernel_damp_normalize:
            cur_kernel_damp_factor = kernel_damp_factor / np.linalg.norm(B_true[j, :])
        else:
            cur_kernel_damp_factor = kernel_damp_factor

        for i in range(n):
            c_observed[j, i] = (cur_kernel_damp_factor * dot_prods[j, i] + inner_constant) ** degree + outer_constant
            if noise:
                epsilon = uniform.rvs(1 - noise_half_width, 2 * noise_half_width)
                c_observed[j, i] *= epsilon

    if normalize_c:
        for i in range(n):
            norm_c = np.linalg.norm(c_observed[:, i])
            if norm_c > 0:
                c_observed[:, i] /= norm_c
            c_observed[np.abs(c_observed[:, i]) < normalize_small_threshold, i] = 0

    return X_observed, c_observed


# Simplified polynomial kernel data generation
def generate_poly_kernel_data_simple(B_true: np.ndarray, n: int, polykernel_degree: int,
                                     noise_half_width: float) -> Tuple[np.ndarray, np.ndarray]:
    d, p = B_true.shape
    alpha_factor = 1 / np.sqrt(p)
    noise_on = noise_half_width != 0

    return generate_poly_kernel_data(B_true, n, polykernel_degree,
                                     kernel_damp_factor=alpha_factor,
                                     noise=noise_on, noise_half_width=noise_half_width,
                                     kernel_damp_normalize=False,
                                     normalize_c=False,
                                     inner_constant=3)


# Sigmoid function
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


# Sigmoid data generation
def generate_sigmoid_data(p_features: int, d_feasibleregion: int, n_sigmoid: int,
                          n_samples: int, bernoulli_prob: float) -> Tuple[np.ndarray, np.ndarray]:
    # True matrices parameterizing network
    B_true_1 = bernoulli.rvs(bernoulli_prob, size=(n_sigmoid, p_features))

    # Ensure input to sigmoid is normal with SD of 5
    col_norms = np.sqrt(np.sum(B_true_1 ** 2, axis=1))
    col_norms[col_norms < 0.01] = 5
    B_true_1 = np.diag(5 / col_norms) @ B_true_1

    B_true_2 = bernoulli.rvs(bernoulli_prob, size=(d_feasibleregion, n_sigmoid))

    X_observed = np.random.randn(p_features, n_samples)
    c_observed = B_true_2 @ sigmoid(B_true_1 @ X_observed)

    return X_observed, c_observed


# Sigmoid returns data generation
def generate_sigmoid_returns_data(p_features: int, d_feasibleregion: int, n_sigmoid: int,
                                  n_samples: int, bernoulli_prob: float, L_matrix: np.ndarray,
                                  sigma_noise: float, return_intercept: float) -> Tuple[np.ndarray, np.ndarray]:
    X, r = generate_sigmoid_data(p_features, d_feasibleregion, n_sigmoid, n_samples, bernoulli_prob)

    d, f = L_matrix.shape
    for i in range(n_samples):
        r[:, i] = return_intercept * np.ones(d) + r[:, i] + L_matrix @ np.random.randn(
            f) + sigma_noise * np.random.randn(d)

    c = -r
    return X, c


# RBF kernel data generation
def generate_rbf_kernel_data(B_true: np.ndarray, n: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate (X, c) from the RBF kernel model
    """
    d, p = B_true.shape
    X_observed = np.random.randn(p, n)

    c_observed = np.zeros((d, n))
    for j in range(d):
        for i in range(n):
            dist = np.linalg.norm(B_true[j, :] - X_observed[:, i])
            c_observed[j, i] = np.exp(-dist ** 2 / (2 * sigma ** 2))

    return X_observed, c_observed


# RBF kernel returns data generation
def generate_rbf_kernel_returns_data(B_true: np.ndarray, n: int, sigma_kernel: float,
                                     L_matrix: np.ndarray, sigma_noise: float,
                                     return_intercept: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate returns data with RBF kernel
    """
    X, r = generate_rbf_kernel_data(B_true, n, sigma_kernel)

    d, f = L_matrix.shape
    for i in range(n):
        r[:, i] = return_intercept * np.ones(d) + r[:, i] + L_matrix @ np.random.randn(
            f) + sigma_noise * np.random.randn(d)

    c = -r
    return X, c


# Polynomial kernel returns data generation
def generate_poly_kernel_returns_data(B_true: np.ndarray, n: int, polykernel_degree: int,
                                      L_matrix: np.ndarray, sigma_noise: float,
                                      return_intercept: float) -> Tuple[np.ndarray, np.ndarray]:
    d, p = B_true.shape
    alpha_factor = 0.05 / np.sqrt(p)

    X, r = generate_poly_kernel_data(B_true, n, polykernel_degree,
                                     kernel_damp_factor=alpha_factor,
                                     noise=False,
                                     kernel_damp_normalize=False,
                                     normalize_c=False,
                                     inner_constant=0.10 ** (1 / polykernel_degree),
                                     outer_constant=0)

    d, f = L_matrix.shape
    for i in range(n):
        r[:, i] = return_intercept * np.ones(d) + r[:, i] + L_matrix @ np.random.randn(
            f) + sigma_noise * np.random.randn(d)

    c = -r
    return X, c


# Gurobi environment setup
def setup_gurobi_env(quiet_mode: bool = True, method_type: str = "barrier",
                     use_time_limit: bool = True, time_limit: float = 60.0) -> gp.Env:
    env = gp.Env()

    if quiet_mode:
        env.setParam("OutputFlag", 0)

    if method_type == "barrier":
        env.setParam("Method", 2)
    elif method_type == "method3":
        env.setParam("Method", 3)
    elif method_type != "default":
        raise ValueError("Enter a valid method type for Gurobi.")

    if use_time_limit:
        env.setParam("TimeLimit", time_limit)

    return env

# Example usage (commented out as it requires Gurobi and a defined oracle)
# if __name__ == "__main__":
#     X = np.random.randn(5, 100)
#     c = np.random.randn(3, 100)
#     B = cost_least_squares(X, c)
#     print(B.shape)
#     env = setup_gurobi_env()