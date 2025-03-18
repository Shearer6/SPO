import numpy as np
import cvxpy as cp
import gurobipy as gp
from typing import Tuple, Callable


def portfolio_simplex_oracle_jump_basic(c: np.ndarray, Sigma: np.ndarray, gamma: float) -> Tuple[float, np.ndarray]:
    """
    Optimization oracle for the portfolio problem:
        min  c^T w
        s.t. w^T Sigma w <= gamma
             e^T w <= 1
             w >= 0
    Here c is the negative return vector, Sigma is the covariance matrix, and gamma is the risk level.
    """
    d = len(c)

    # Define variables
    w = cp.Variable(d)

    # Define constraints
    constraints = [
        w >= 0,
        cp.sum(w) <= 1,
        cp.quad_form(w, Sigma) <= gamma
    ]

    # Define objective
    objective = cp.Minimize(c @ w)

    # Define and solve problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)  # Use Gurobi solver, suppress output

    # Extract results
    z_ast = prob.value
    w_ast = w.value

    return z_ast, w_ast


def portfolio_simplex_jump_setup(Sigma: np.ndarray, gamma: float, gurobiEnv: gp.Env = gp.Env()) -> Callable[
    [np.ndarray], Tuple[float, np.ndarray]]:
    """
    Smarter implementation of the portfolio oracle. Constructs the feasible region ahead of time
    and returns a function that takes c as input.
    """
    d, d2 = Sigma.shape

    if d != d2:
        raise ValueError("Sigma dimensions don't match")

    # Define variables
    w = cp.Variable(d)

    # Define constraints (fixed for all c)
    constraints = [
        w >= 0,
        cp.sum(w) <= 1,
        cp.quad_form(w, Sigma) <= gamma
    ]

    # Define a parameter for c
    c_param = cp.Parameter(d)

    # Define objective (to be updated with c)
    objective = cp.Minimize(c_param @ w)

    # Define problem
    prob = cp.Problem(objective, constraints)

    def local_portfolio_oracle(c: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Inner function that updates the objective with a new c and solves the problem.
        """
        c_param.value = c
        prob.solve(solver=cp.GUROBI, verbose=False, env=gurobiEnv)
        z_ast = prob.value
        w_ast = w.value
        return z_ast, w_ast

    return local_portfolio_oracle



# Example usage (for testing)
if __name__ == "__main__":
    # Example data
    np.random.seed(342)
    d = 10
    factors = 4
    F = 0.05 * np.random.rand(d, factors)
    mu = np.random.rand(d)
    r = mu + F @ np.random.randn(factors) + 0.2 * np.random.randn(d)
    Sigma = F @ F.T + np.eye(d) * 0.2
    gamma = 0.1

    # Test basic oracle
    z1, w1 = portfolio_simplex_oracle_jump_basic(-r, Sigma, gamma)
    print(f"Basic Oracle: z = {z1}, w = {w1}")

    # Test setup oracle
    oracle = portfolio_simplex_jump_setup(Sigma, gamma)
    z2, w2 = oracle(-r)
    print(f"Setup Oracle: z = {z2}, w = {w2}")

    # Check approximate equality
    print(f"z1 ≈ z2: {np.isclose(z1, z2, atol=0.0001)}")
    print(f"w1 ≈ w2: {np.allclose(w1, w2, atol=0.0001)}")