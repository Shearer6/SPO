import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Union, Tuple, Optional, Callable
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csr_matrix
from utils import oracle_dataset

# Placeholder for missing function (assumed to be defined elsewhere)
def sp_flow_jump_setup(sources: List[int], destinations: List[int], start_node: int,
                       end_node: int, solver: str = "Gurobi") -> Callable:
    raise NotImplementedError("sp_flow_jump_setup needs to be implemented.")





# Reformulation algorithm parameters
@dataclass
class ReformulationPathParms:
    lambda_max: Optional[float] = None  # None replaces Julia's missing
    lambda_min_ratio: float = 0.0001
    num_lambda: int = 100
    solver: str = "Gurobi"  # Using str instead of Symbol
    gurobiEnv: gp.Env = gp.Env()
    regularization: str = "ridge"
    regularize_first_column_B: bool = False
    upper_bound_B_present: bool = False
    upper_bound_B: float = 1e6
    po_loss_function: str = "leastSquares"
    huber_delta: float = 0.0
    verbose: bool = False
    algorithm_type: str = "fake_algorithm"


# Shortest path graph specification
@dataclass
class ShortestPathGraph:
    sources: List[int]
    destinations: List[int]
    start_node: int
    end_node: int
    acyclic: bool = False


# Main SPO+ reformulation path function
def sp_reformulation_path_jump(X: np.ndarray, c: np.ndarray, sp_graph: ShortestPathGraph,
                               sp_oracle: Optional[Callable] = None,
                               reform_alg_parms: ReformulationPathParms = ReformulationPathParms()) -> Tuple[
    List[np.ndarray], np.ndarray]:
    """
    Solves the empirical risk SPO+ problem for the shortest path problem using a reformulation approach.
    """
    # Unpack parameters
    lambda_max = reform_alg_parms.lambda_max
    lambda_min_ratio = reform_alg_parms.lambda_min_ratio
    num_lambda = reform_alg_parms.num_lambda
    solver = reform_alg_parms.solver
    gurobiEnv = reform_alg_parms.gurobiEnv
    regularization = reform_alg_parms.regularization
    regularize_first_column_B = reform_alg_parms.regularize_first_column_B
    upper_bound_B_present = reform_alg_parms.upper_bound_B_present
    upper_bound_B = reform_alg_parms.upper_bound_B
    verbose = reform_alg_parms.verbose
    algorithm_type = reform_alg_parms.algorithm_type

    sources = sp_graph.sources
    destinations = sp_graph.destinations
    start_node = sp_graph.start_node
    end_node = sp_graph.end_node
    acyclic = sp_graph.acyclic

    # Dimension check
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")

    # Process graph
    nodes = np.unique(np.union1d(sources, destinations))
    n_nodes = len(nodes)
    n_edges = len(sources)
    if n_edges != d:
        raise ValueError("Dimensions of the input are mismatched.")

    # Sparse incidence matrix
    I_vec = np.concatenate([sources, destinations])
    J_vec = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    V_vec = np.concatenate([-np.ones(n_edges), np.ones(n_edges)])
    A_mat = csr_matrix((V_vec, (I_vec, J_vec)), shape=(max(nodes) + 1, n_edges)).toarray()
    A_mat_trans = A_mat.T

    # Set up RHS
    b_vec = np.zeros(n_nodes)
    b_vec[start_node] = -1
    b_vec[end_node] = 1
    b_vec = b_vec.astype(int)

    # Get oracle and w*
    if sp_oracle is None:
        sp_oracle = sp_flow_jump_setup(sources, destinations, start_node, end_node, solver=solver)
    z_star_data, w_star_data = oracle_dataset(c, sp_oracle)

    # Define CVXPY variables
    p_var = cp.Variable((n_nodes, n))
    B_var = cp.Variable((d, p))
    if upper_bound_B_present:
        B_var = cp.Variable((d, p), value=np.zeros((d, p)), bounds=(-upper_bound_B, upper_bound_B))
    s_var = cp.Variable((d, n), nonneg=True) if not acyclic else None

    # Constraints
    constraints = []
    for i in range(n):
        if acyclic:
            constraints.append(-A_mat_trans @ p_var[:, i] >= c[:, i] - 2 * B_var @ X[:, i])
        else:
            constraints.append(s_var[:, i] - A_mat_trans @ p_var[:, i] >= c[:, i] - 2 * B_var @ X[:, i])

    # Objective without regularization
    if regularization == "ridge":
        obj_expr_noreg = 0
    elif regularization == "lasso":
        obj_expr_noreg = 0
    else:
        raise ValueError("enter valid regularization: 'ridge' or 'lasso'")

    for i in range(n):
        if acyclic:
            cur_expr = -b_vec @ p_var[:, i] + 2 * (w_star_data[:, i].T @ B_var @ X[:, i]) - z_star_data[i]
        else:
            cur_expr = -b_vec @ p_var[:, i] + cp.sum(s_var[:, i]) + 2 * (w_star_data[:, i].T @ B_var @ X[:, i]) - \
                       z_star_data[i]
        obj_expr_noreg += cur_expr

    # Lambda sequence
    if lambda_max is None:
        lambda_max = (d / n) * (np.linalg.norm(X) ** 2)

    if num_lambda == 1 and lambda_max == 0:
        lambdas = [0.0]
    else:
        lambda_min = lambda_max * lambda_min_ratio
        log_lambdas = np.linspace(np.log(lambda_min), np.log(lambda_max), num_lambda)
        lambdas = np.exp(log_lambdas)

    # Lasso theta variables
    if regularization == "lasso":
        theta_var = cp.Variable((d, p))
        constraints.extend([theta_var >= B_var, theta_var >= -B_var])

    # Solve for path
    B_soln_list = []
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        if verbose:
            print(f"Trying lambda = {cur_lambda}")

        if regularization == "ridge" and regularize_first_column_B:
            reg_term = n * (cur_lambda / 2) * cp.sum_squares(B_var)
        elif regularization == "ridge" and not regularize_first_column_B:
            reg_term = n * (cur_lambda / 2) * cp.sum_squares(B_var[:, 1:])
        elif regularization == "lasso" and regularize_first_column_B:
            reg_term = n * cur_lambda * cp.sum(theta_var)
        elif regularization == "lasso" and not regularize_first_column_B:
            reg_term = n * cur_lambda * cp.sum(theta_var[:, 1:])
        else:
            raise ValueError("enter valid regularization: 'ridge' or 'lasso'")

        obj_expr_full = obj_expr_noreg + reg_term
        prob = cp.Problem(cp.Minimize(obj_expr_full), constraints)

        # Solve
        if solver == "Gurobi":
            prob.solve(solver=cp.GUROBI, env=gurobiEnv)
        else:
            prob.solve()  # Default CVXPY solver (e.g., ECOS, SCS)

        B_ast = B_var.value
        mod_status = prob.status

        if B_ast is None or np.any(np.isnan(B_ast)):
            B_ast = np.zeros((d, p))
            print(f"We got NaNs in B_ast and the reason is: {mod_status}. The algorithm is {algorithm_type}.")
        elif mod_status != "optimal":
            print(
                f"There are no NaNs in B_ast, but the model status is: {mod_status}. The algorithm is {algorithm_type}.")

        B_soln_list.append(B_ast.copy())

    return B_soln_list, lambdas


# Least squares path function
def leastSquares_path_jump(X: np.ndarray, c: np.ndarray,
                           reform_alg_parms: ReformulationPathParms = ReformulationPathParms()) -> Tuple[
    List[np.ndarray], np.ndarray]:
    """
    Solves the empirical risk least squares problem using a reformulation approach.
    """
    # Unpack parameters
    lambda_max = reform_alg_parms.lambda_max
    lambda_min_ratio = reform_alg_parms.lambda_min_ratio
    num_lambda = reform_alg_parms.num_lambda
    solver = reform_alg_parms.solver
    gurobiEnv = reform_alg_parms.gurobiEnv
    regularization = reform_alg_parms.regularization
    regularize_first_column_B = reform_alg_parms.regularize_first_column_B
    upper_bound_B_present = reform_alg_parms.upper_bound_B_present
    upper_bound_B = reform_alg_parms.upper_bound_B
    po_loss_function = reform_alg_parms.po_loss_function
    huber_delta = reform_alg_parms.huber_delta
    verbose = reform_alg_parms.verbose
    algorithm_type = reform_alg_parms.algorithm_type

    # Dimension check
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")

    # Define CVXPY variables
    B_var = cp.Variable((d, p))
    if upper_bound_B_present:
        B_var = cp.Variable((d, p), value=np.zeros((d, p)), bounds=(-upper_bound_B, upper_bound_B))

    # Objective without regularization
    if po_loss_function == "leastSquares":
        obj_expr_noreg = cp.sum_squares(c - B_var @ X)
    elif po_loss_function == "huber":
        w_var = cp.Variable((d, n), nonneg=True)
        v_var = cp.Variable((d, n))
        constraints = [
            c - B_var @ X <= v_var + w_var,
            -(c - B_var @ X) <= v_var + w_var,
            w_var <= huber_delta * np.ones((d, n))
        ]
        obj_expr_noreg = cp.sum_squares(w_var) + 2 * huber_delta * cp.sum(v_var)
    elif po_loss_function == "absolute":
        w_var = cp.Variable((d, n), nonneg=True)
        constraints = [
            c - B_var @ X <= w_var,
            -(c - B_var @ X) <= w_var
        ]
        obj_expr_noreg = 2 * cp.sum(w_var)
    else:
        raise ValueError("Enter a valid loss function: 'leastSquares', 'huber', or 'absolute'")

    # Lambda sequence
    if lambda_max is None:
        lambda_max = (d / n) * (np.linalg.norm(X) ** 2)

    if num_lambda == 1 and lambda_max == 0:
        lambdas = [0.0]
    else:
        lambda_min = lambda_max * lambda_min_ratio
        log_lambdas = np.linspace(np.log(lambda_min), np.log(lambda_max), num_lambda)
        lambdas = np.exp(log_lambdas)

    # Lasso theta variables
    if regularization == "lasso":
        theta_var = cp.Variable((d, p))
        constraints.extend([theta_var >= B_var, theta_var >= -B_var])
    else:
        constraints = []

    # Solve for path
    B_soln_list = []
    for i in range(num_lambda):
        cur_lambda = lambdas[i]
        if verbose:
            print(f"Trying lambda = {cur_lambda}")

        if regularization == "ridge" and regularize_first_column_B:
            reg_term = n * cur_lambda * cp.sum_squares(B_var)
        elif regularization == "ridge" and not regularize_first_column_B:
            reg_term = n * cur_lambda * cp.sum_squares(B_var[:, 1:])
        elif regularization == "lasso" and regularize_first_column_B:
            reg_term = 2 * n * cur_lambda * cp.sum(theta_var)
        elif regularization == "lasso" and not regularize_first_column_B:
            reg_term = 2 * n * cur_lambda * cp.sum(theta_var[:, 1:])
        else:
            raise ValueError("enter valid regularization: 'ridge' or 'lasso'")

        obj_expr_full = obj_expr_noreg + reg_term
        prob = cp.Problem(cp.Minimize(obj_expr_full), constraints)

        # Solve
        if solver == "Gurobi":
            prob.solve(solver=cp.GUROBI, env=gurobiEnv)
        else:
            prob.solve()  # Default CVXPY solver

        B_ast = B_var.value
        mod_status = prob.status

        if B_ast is None or np.any(np.isnan(B_ast)):
            B_ast = np.zeros((d, p))
            print(f"We got NaNs in B_ast and the reason is: {mod_status}. The algorithm is {algorithm_type}.")
        elif mod_status != "optimal":
            print(
                f"There are no NaNs in B_ast, but the model status is: {mod_status}. The algorithm is {algorithm_type}.")

        B_soln_list.append(B_ast.copy())

    return B_soln_list, lambdas

# Example usage (commented out due to dependencies)
# if __name__ == "__main__":
#     X = np.random.randn(5, 100)
#     c = np.random.randn(3, 100)
#     sp_graph = ShortestPathGraph(sources=[0, 1], destinations=[1, 2], start_node=0, end_node=2)
#     B_list, lambdas = sp_reformulation_path_jump(X, c, sp_graph)
#     print(len(B_list), lambdas)