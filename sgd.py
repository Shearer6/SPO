import numpy as np
from dataclasses import dataclass
from typing import Union, List, Tuple, Callable, Optional
from utils import oracle_dataset,spo_loss,spo_plus_loss

# Class for storing SGD history
@dataclass
class SGDHistory:
    spo_train_history: np.ndarray
    spo_plus_train_history: np.ndarray
    spo_holdout_history: np.ndarray
    returned_iter: int


# Class for SGD algorithm parameters with default values
@dataclass
class SGDParms:
    grad_type: str = "stochastic"  # Using str instead of Symbol
    lambda_: float = 0.0  # 'lambda' is a reserved keyword in Python, using lambda_
    numiter: int = 2 #原来1000
    batchsize: int = 100 #原来10
    step_type: str = "long_dynamic"
    long_factor: float = 0.1
    holdout_set: bool = False
    holdout_period: int = 1
    history: bool = False
    history_period: int = 1


# Main SPO+ SGD function
def spoPlus_sgd(X: np.ndarray, c: np.ndarray, oracle: Callable,
                alg_parms: SGDParms = SGDParms(),
                X_holdout: np.ndarray = None, c_holdout: np.ndarray = None,
                B_init: np.ndarray = None) -> Tuple[np.ndarray, SGDHistory]:
    """
    Solve the SPO+ problem with stochastic gradient descent.
    Returns the B matrix and an SGDHistory object.
    """
    # Default empty arrays if not provided
    X_holdout = np.array([]) if X_holdout is None else X_holdout
    c_holdout = np.array([]) if c_holdout is None else c_holdout

    # Unpack parameters
    grad_type = alg_parms.grad_type
    lambda_ = alg_parms.lambda_
    numiter = alg_parms.numiter
    batchsize = alg_parms.batchsize
    step_type = alg_parms.step_type
    long_factor = alg_parms.long_factor
    holdout_set = alg_parms.holdout_set
    holdout_period = alg_parms.holdout_period
    history = alg_parms.history
    history_period = alg_parms.history_period

    # Dimension check
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")

    if B_init is None:
        B_init = np.zeros((d, p))

    # Pre-process to get z_star and w_star
    z_star_data, w_star_data = oracle_dataset(c, oracle)

    # Subgradient functions
    if grad_type == "deterministic":
        def subgrad(B_new: np.ndarray) -> np.ndarray:
            G_new = np.zeros((d, p))
            for i in range(n):
                spoplus_cost_vec = 2 * B_new @ X[:, i] - c[:, i]
                z_oracle, w_oracle = oracle(spoplus_cost_vec)
                w_star_diff = w_star_data[:, i] - w_oracle
                G_new += 2 * np.outer(w_star_diff, X[:, i])
            G_new = (1 / n) * G_new + lambda_ * B_new
            return G_new
    elif grad_type == "stochastic":
        def subgrad(B_new: np.ndarray) -> np.ndarray:
            G_new = np.zeros((d, p))
            for _ in range(batchsize):
                i = np.random.randint(0, n)
                spoplus_cost_vec = 2 * B_new @ X[:, i] - c[:, i]
                z_oracle, w_oracle = oracle(spoplus_cost_vec)
                w_star_diff = w_star_data[:, i] - w_oracle
                G_new += 2 * np.outer(w_star_diff, X[:, i])
            G_new = (1 / batchsize) * G_new + lambda_ * B_new
            return G_new
    elif grad_type == "deterministic_LS":
        def subgrad(B_new: np.ndarray) -> np.ndarray:
            G_new = np.zeros((d, p))
            for i in range(n):
                residuals = B_new @ X[:, i] - c[:, i]
                G_new += np.outer(residuals, X[:, i])
            G_new = (1 / n) * G_new + lambda_ * B_new
            return G_new
    elif grad_type == "stochastic_LS":
        def subgrad(B_new: np.ndarray) -> np.ndarray:
            G_new = np.zeros((d, p))
            for _ in range(batchsize):
                i = np.random.randint(0, n)
                residuals = B_new @ X[:, i] - c[:, i]
                G_new += np.outer(residuals, X[:, i])
            G_new = (1 / batchsize) * G_new + lambda_ * B_new
            return G_new
    else:
        raise ValueError("Enter a valid grad_type.")

    # Step-size functions
    if step_type == "short":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            return 2 / (lambda_ * (iter + 2))
    elif step_type == "short_practical":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            return 2 / (iter + 2)
    elif step_type == "long_static":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            return long_factor / np.sqrt(numiter + 1)
    elif step_type == "long_dynamic":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            return long_factor / np.sqrt(iter + 1)
    elif step_type == "long_static_normalized":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            G_norm = np.linalg.norm(G_new)
            return long_factor / (G_norm * np.sqrt(numiter + 1))
    elif step_type == "long_dynamic_normalized":
        def step_size(iter: int, G_new: np.ndarray) -> float:
            G_norm = np.linalg.norm(G_new)
            return long_factor / (G_norm * np.sqrt(iter + 1))
    else:
        raise ValueError("Enter a valid step_type.")

    # Pre-processing for holdout set and history
    if holdout_set:
        p_holdout, n_holdout = X_holdout.shape
        d_holdout, n_holdout_2 = c_holdout.shape
        if n_holdout != n_holdout_2 or p_holdout != p or d_holdout != d:
            raise ValueError("Dimensions of the input are mismatched.")

        z_star_holdout, w_star_holdout = oracle_dataset(c_holdout, oracle)

        def holdout_set_spo(B_new: np.ndarray) -> float:
            return spo_loss(B_new, X_holdout, c_holdout, oracle, z_star=z_star_holdout)

        B_cur_best_holdout = np.zeros((d, p))
        cur_best_spo_holdout = float('inf')
        spo_holdout_history = np.full(numiter, -1.0)

    if history:
        def training_set_spo(B_new: np.ndarray) -> float:
            return spo_loss(B_new, X, c, oracle, z_star=z_star_data)

        def training_set_spo_plus(B_new: np.ndarray) -> float:
            return spo_plus_loss(B_new, X, c, oracle, z_star=z_star_data, w_star=w_star_data)

        spo_train_history = np.full(numiter, -1.0)
        spo_plus_train_history = np.full(numiter, -1.0)

    # Subgradient method logic
    B_iter = B_init.copy()
    B_avg_iter = B_init.copy()
    step_size_sum = 0
    returned_iter = numiter - 1

    for iter in range(numiter):
        G_iter = subgrad(B_iter)
        step_iter = step_size(iter, G_iter)

        # Update average and current iterates
        step_size_sum += step_iter
        step_avg = step_iter / step_size_sum
        B_avg_iter = (1 - step_avg) * B_avg_iter + step_avg * B_iter
        B_iter -= step_iter * G_iter

        # Update holdout set and history
        if holdout_set and iter % holdout_period == 0:
            spo_holdout_iter = holdout_set_spo(B_avg_iter)
            spo_holdout_history[iter] = spo_holdout_iter
            if spo_holdout_iter < cur_best_spo_holdout:
                cur_best_spo_holdout = spo_holdout_iter
                B_cur_best_holdout = B_avg_iter.copy()
                returned_iter = iter

        if history and iter % history_period == 0:
            spo_train_history[iter] = training_set_spo(B_avg_iter)
            spo_plus_train_history[iter] = training_set_spo_plus(B_avg_iter)

    # Generate history object
    if holdout_set and history:
        sgd_history_trace = SGDHistory(spo_train_history, spo_plus_train_history, spo_holdout_history, returned_iter)
    elif not holdout_set and history:
        sgd_history_trace = SGDHistory(spo_train_history, spo_plus_train_history, np.array([]), returned_iter)
    else:
        sgd_history_trace = SGDHistory(np.array([]), np.array([]), np.array([]), returned_iter)

    # Return results
    if holdout_set:
        return B_cur_best_holdout, sgd_history_trace
    else:
        return B_avg_iter, sgd_history_trace


# Wrapper for least-squares SGD
def leastSquares_sgd(X: np.ndarray, c: np.ndarray, oracle: Callable,
                     alg_parms: SGDParms = SGDParms(grad_type="stochastic_LS"),
                     X_holdout: np.ndarray = None, c_holdout: np.ndarray = None,
                     B_init: np.ndarray = None) -> Tuple[np.ndarray, SGDHistory]:
    alg_parms = SGDParms(**{**vars(alg_parms), "grad_type": "stochastic_LS"})
    return spoPlus_sgd(X, c, oracle, alg_parms=alg_parms, X_holdout=X_holdout,
                       c_holdout=c_holdout, B_init=B_init)


# Class for SGD path parameters
@dataclass
class SGDPathParms:
    lambda_max: Union[str, float] = "practical"
    lambda_min_ratio: float = 0.0001
    num_lambda: int = 100
    grad_type: str = "stochastic"
    second_moment_bound: str = "practical"
    obj_accuracy: float = 0.0001
    batchsize: int = 10
    iteration_limit: int = 10000
    step_type: str = "short"
    verbose: bool = False


# SPO+ SGD path function
def spoPlus_sgd_path(X: np.ndarray, c: np.ndarray, oracle: Callable,
                     path_alg_parms: SGDPathParms = SGDPathParms()) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Solves the ridge regularization path using SGD over a grid of lambda values.
    """
    # Unpack parameters
    lambda_max = path_alg_parms.lambda_max
    lambda_min_ratio = path_alg_parms.lambda_min_ratio
    num_lambda = path_alg_parms.num_lambda
    grad_type = path_alg_parms.grad_type
    second_moment_bound = path_alg_parms.second_moment_bound
    obj_accuracy = path_alg_parms.obj_accuracy
    batchsize = path_alg_parms.batchsize
    iteration_limit = path_alg_parms.iteration_limit
    step_type = path_alg_parms.step_type
    verbose = path_alg_parms.verbose

    # Get dimensions
    p, n = X.shape
    d, n2 = c.shape

    # Compute second moment bound
    if second_moment_bound == "practical":
        second_moment_bound_val = (1 / n) * (np.linalg.norm(X) ** 2)
    elif second_moment_bound == "theory":
        second_moment_bound_val = 16 * (d / n) * (np.linalg.norm(X) ** 2)
    else:
        raise ValueError("Enter a valid second_moment_bound: either 'practical' or 'theory'")

    # Compute lambda_max
    if lambda_max == "practical":
        lambda_max_val = (d / n) * (np.linalg.norm(X) ** 2)
    elif lambda_max == "theory":
        lambda_max_val = (2 * second_moment_bound_val) / obj_accuracy
    elif not isinstance(lambda_max, float) or lambda_max <= 0:
        raise ValueError("Enter a valid lambda_max: either 'practical', 'theory', or a positive float")
    else:
        lambda_max_val = lambda_max

    # Construct lambda sequence
    lambda_min = lambda_max_val * lambda_min_ratio
    log_lambdas = np.linspace(np.log(lambda_min), np.log(lambda_max_val), num_lambda)
    lambdas = np.exp(log_lambdas)

    # Run SGD on path
    B_soln_list = []
    B_spo_plus = np.zeros((d, p))
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        if verbose:
            print(f"Trying lambda = {cur_lambda}")

        num_iters = int(np.ceil((2 * second_moment_bound_val) / (cur_lambda * obj_accuracy))) + 1
        num_iters = min(num_iters, iteration_limit)

        sgd_alg_parms = SGDParms(grad_type=grad_type, lambda_=cur_lambda, numiter=num_iters,
                                 batchsize=batchsize, step_type=step_type)

        B_spo_plus, _ = spoPlus_sgd(X, c, oracle, alg_parms=sgd_alg_parms, B_init=B_spo_plus)
        B_soln_list.append(B_spo_plus.copy())

    return B_soln_list, lambdas


# Wrapper for least-squares SGD path
def leastSquares_sgd_path(X: np.ndarray, c: np.ndarray, oracle: Callable,
                          path_alg_parms: SGDPathParms = SGDPathParms(grad_type="stochastic_LS")) -> Tuple[
    List[np.ndarray], np.ndarray]:
    path_alg_parms = SGDPathParms(**{**vars(path_alg_parms), "grad_type": "stochastic_LS"})
    return spoPlus_sgd_path(X, c, oracle, path_alg_parms=path_alg_parms)

# Example usage (commented out as oracle and other dependencies are not defined)
# if __name__ == "__main__":
#     X = np.random.rand(5, 100)  # p x n
#     c = np.random.rand(3, 100)  # d x n
#     def dummy_oracle(cost_vec): return (0, np.zeros_like(cost_vec))
#     B, history = spoPlus_sgd(X, c, dummy_oracle)
#     print(B.shape, history.returned_iter)