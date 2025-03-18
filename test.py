import numpy as np
import unittest
import cvxpy as cp
import gurobipy as gp
from typing import Tuple, Callable


# portfolio_oracle.py content (translated from portfolio_oracle.jl)
def portfolio_simplex_oracle_jump_basic(c: np.ndarray, Sigma: np.ndarray, gamma: float) -> Tuple[float, np.ndarray]:
    """
    Optimization oracle for the portfolio problem:
        min  c^T w
        s.t. w^T Sigma w <= gamma
             e^T w <= 1
             w >= 0
    """
    d = len(c)

    w = cp.Variable(d)
    constraints = [
        w >= 0,
        cp.sum(w) <= 1,
        cp.quad_form(w, Sigma) <= gamma
    ]
    objective = cp.Minimize(c @ w)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.GUROBI, verbose=False)

    z_ast = prob.value
    w_ast = w.value

    return z_ast, w_ast


def portfolio_simplex_jump_setup(Sigma: np.ndarray, gamma: float, gurobiEnv: gp.Env = gp.Env()) -> Callable[
    [np.ndarray], Tuple[float, np.ndarray]]:
    """
    Smarter implementation of the portfolio oracle, constructing the feasible region ahead of time.
    """
    d, d2 = Sigma.shape

    if d != d2:
        raise ValueError("Sigma dimensions don't match")

    w = cp.Variable(d)
    constraints = [
        w >= 0,
        cp.sum(w) <= 1,
        cp.quad_form(w, Sigma) <= gamma
    ]
    c_param = cp.Parameter(d)
    objective = cp.Minimize(c_param @ w)
    prob = cp.Problem(objective, constraints)

    def local_portfolio_oracle(c: np.ndarray) -> Tuple[float, np.ndarray]:
        c_param.value = c
        prob.solve(solver=cp.GUROBI, verbose=False, env=gurobiEnv)
        z_ast = prob.value
        w_ast = w.value
        return z_ast, w_ast

    return local_portfolio_oracle


# Test script (translated from the provided Julia code)
class TestPortfolioOracle(unittest.TestCase):
    def test_portfolio_oracle(self):
        # Set random seed for reproducibility
        np.random.seed(342)

        # Parameters
        d = 10
        factors = 4

        # Generate random data
        F = 0.05 * np.random.rand(d, factors)
        mu = np.random.rand(d)
        r = mu + F @ np.random.randn(factors) + 0.2 * np.random.randn(d)

        print(f"R is: {r}")

        Sigma = F @ F.T + np.eye(d) * 0.2  # Sigma = F*F' + 0.2*I

        # Call the portfolio simplex oracle functions
        z1, w1 = portfolio_simplex_oracle_jump_basic(-r, Sigma, 0.1)

        oracle = portfolio_simplex_jump_setup(Sigma, 0.1)
        z2, w2 = oracle(-r)

        # Test approximate equality
        self.assertTrue(np.isclose(z1, z2, atol=0.0001), "z1 and z2 are not approximately equal")
        self.assertTrue(np.allclose(w1, w2, atol=0.0001), "w1 and w2 are not approximately equal")


if __name__ == "__main__":
    unittest.main()