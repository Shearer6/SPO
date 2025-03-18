import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable, Union
import matplotlib.pyplot as plt
from sgd import spoPlus_sgd_path, leastSquares_sgd_path, spoPlus_sgd, SGDPathParms, SGDParms, SGDHistory
from reformulation import sp_reformulation_path_jump, leastSquares_path_jump, ReformulationPathParms, ShortestPathGraph
from utils import spo_loss, spo_plus_loss, least_squares_loss, absolute_loss, huber_loss, oracle_dataset


# Define validation set algorithm parameters
@dataclass
class ValParms:
    algorithm_type: str = "spo_plus_sgd"  # Using str instead of Symbol
    validation_set_percent: float = 0.2
    validation_loss: str = "spo_loss"
    plot_results: bool = False
    resolve_sgd: bool = False
    resolve_sgd_accuracy: float = 0.00001
    resolve_iteration_limit: int = 50000


# Main validation set algorithm function
def validation_set_alg(X: np.ndarray, c: np.ndarray, oracle: Callable,
                       sp_graph: Optional[ShortestPathGraph] = None,
                       train_ind: Optional[List[int]] = None,
                       validation_ind: Optional[List[int]] = None,
                       val_alg_parms: ValParms = ValParms(),
                       path_alg_parms: Union[SGDPathParms, ReformulationPathParms] = SGDPathParms()) -> Tuple[
    np.ndarray, float]:
    """
    Applies the validation set approach to tune the regularization parameter for SPO+ or least squares.
    """
    # Unpack validation set parameters
    algorithm_type = val_alg_parms.algorithm_type
    validation_set_percent = val_alg_parms.validation_set_percent
    plot_results = val_alg_parms.plot_results
    validation_loss = val_alg_parms.validation_loss
    resolve_sgd = val_alg_parms.resolve_sgd
    resolve_sgd_accuracy = val_alg_parms.resolve_sgd_accuracy
    resolve_iteration_limit = val_alg_parms.resolve_iteration_limit

    # Check validity of path_alg_parms
    if not isinstance(path_alg_parms, (SGDPathParms, ReformulationPathParms)):
        raise ValueError("path_alg_parms must be either SGDPathParms or ReformulationPathParms")

    # Check and unpack sp_graph if provided
    if sp_graph is not None:
        sources = sp_graph.sources
        destinations = sp_graph.destinations
        start_node = sp_graph.start_node
        end_node = sp_graph.end_node

    # Get dimensions of input
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")

    # Split into train/validation if not given
    if train_ind is None and validation_ind is None:
        validation_size = int(n * validation_set_percent)
        validation_ind = np.random.choice(n, size=validation_size, replace=False)
        train_ind = np.setdiff1d(np.arange(n), validation_ind)

    X_train = X[:, train_ind]
    X_validation = X[:, validation_ind]
    c_train = c[:, train_ind]
    c_validation = c[:, validation_ind]

    # Train models based on algorithm type
    if algorithm_type == "spo_plus_sgd":
        B_soln_list, lambdas = spoPlus_sgd_path(X_train, c_train, oracle, path_alg_parms=path_alg_parms)
    elif algorithm_type == "ls_sgd":
        B_soln_list, lambdas = leastSquares_sgd_path(X_train, c_train, oracle, path_alg_parms=path_alg_parms)
    elif algorithm_type == "sp_spo_plus_reform":
        if sp_graph is None:
            raise ValueError("sp_graph is required for sp_spo_plus_reform algorithm")
        B_soln_list, lambdas = sp_reformulation_path_jump(X_train, c_train, sp_graph, sp_oracle=oracle,
                                                          reform_alg_parms=path_alg_parms)
    elif algorithm_type == "ls_jump":
        B_soln_list, lambdas = leastSquares_path_jump(X_train, c_train, reform_alg_parms=path_alg_parms)
    else:
        raise ValueError("Enter a valid algorithm type: 'spo_plus_sgd', 'ls_sgd', 'sp_spo_plus_reform', or 'ls_jump'")

    # Get best model by evaluating validation loss
    num_lambda = len(lambdas)
    validation_loss_list = np.zeros(num_lambda)

    # Define loss function dynamically
    def get_loss_function(loss_type: str, B: np.ndarray, X_val: np.ndarray, c_val: np.ndarray,
                          oracle: Callable) -> float:
        if loss_type == "spo_loss":
            return spo_loss(B, X_val, c_val, oracle)
        elif loss_type == "spo_plus_loss":
            return spo_plus_loss(B, X_val, c_val, oracle)
        elif loss_type == "least_squares_loss":
            return least_squares_loss(B, X_val, c_val)
        elif loss_type == "absolute_loss":
            return absolute_loss(B, X_val, c_val)
        elif loss_type == "huber_loss":
            if isinstance(path_alg_parms, ReformulationPathParms):
                return huber_loss(B, X_val, c_val, path_alg_parms.huber_delta)
            else:
                raise ValueError("Huber loss requires ReformulationPathParms with huber_delta")
        elif loss_type == "hamming_loss":
            z_validation, w_validation = oracle_dataset(c_val, oracle)
            c_ham_validation = np.ones_like(w_validation) - w_validation
            return spo_loss(B, X_val, c_ham_validation, oracle)
        else:
            raise ValueError(
                "Enter a valid validation set loss function: 'spo_loss', 'spo_plus_loss', 'least_squares_loss', 'absolute_loss', 'huber_loss', or 'hamming_loss'")

    for i in range(num_lambda):
        validation_loss_list[i] = get_loss_function(validation_loss, B_soln_list[i], X_validation, c_validation, oracle)

    # Plot results if requested
    if plot_results:
        plt.figure(figsize=(6, 4))
        plt.plot(np.log(lambdas), validation_loss_list, label="Validation Loss")
        plt.xlabel("Log(Lambda)")
        plt.ylabel("Validation Set SPO Loss")
        plt.legend()
        plt.savefig("plots/validation_set_plot_spo_loss.svg", format="svg")
        plt.close()

        # Plot surrogate loss (SPO+ or least squares)
        surrogate_loss = np.zeros(num_lambda)
        if algorithm_type in ["spo_plus_sgd", "sp_spo_plus_reform"]:
            for i in range(num_lambda):
                surrogate_loss[i] = spo_plus_loss(B_soln_list[i], X_validation, c_validation, oracle)
            ylabel = "Validation Set SPO+ Loss"
            filename = "plots/validation_set_plot_spoplus_loss.svg"
        elif algorithm_type in ["ls_sgd", "ls_jump"]:
            for i in range(num_lambda):
                surrogate_loss[i] = least_squares_loss(B_soln_list[i], X_validation, c_validation)
            ylabel = "Validation Set Least Squares Loss"
            filename = "plots/validation_set_plot_leastsquares_loss.svg"

        plt.figure(figsize=(6, 4))
        plt.plot(np.log(lambdas), surrogate_loss, label="Surrogate Loss")
        plt.xlabel("Log(Lambda)")
        plt.ylabel(ylabel)
        plt.legend()
        plt.savefig(filename, format="svg")
        plt.close()

    # Select best model
    best_ind = np.argmin(validation_loss_list)
    best_lambda = lambdas[best_ind]
    best_B_matrix = B_soln_list[best_ind]

    # Resolve with SGD if requested
    if resolve_sgd and algorithm_type == "spo_plus_sgd":
        n_train = len(train_ind)
        second_moment_bound = (1 / n_train) * (np.linalg.norm(X_train) ** 2)
        num_iters = int(np.ceil((2 * second_moment_bound) / (best_lambda * resolve_sgd_accuracy))) + 1
        num_iters = min(num_iters, resolve_iteration_limit)

        sgd_alg_parms = SGDParms(
            grad_type=path_alg_parms.grad_type,
            lambda_=best_lambda,
            numiter=num_iters,
            batchsize=path_alg_parms.batchsize,
            step_type="short"
        )

        B_spo_plus_final, _ = spoPlus_sgd(X_train, c_train, oracle, alg_parms=sgd_alg_parms, B_init=best_B_matrix)
    else:
        B_spo_plus_final = best_B_matrix

    return B_spo_plus_final, best_lambda

# Example usage (commented out due to dependencies)
# if __name__ == "__main__":
#     X = np.random.randn(5, 100)
#     c = np.random.randn(3, 100)
#     def dummy_oracle(cost_vec): return (0, np.zeros_like(cost_vec))
#     B_final, best_lambda = validation_set_alg(X, c, dummy_oracle)
#     print(B_final.shape, best_lambda)