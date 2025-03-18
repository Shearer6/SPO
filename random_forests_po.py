import numpy as np
from dataclasses import dataclass
from typing import List
from sklearn.ensemble import RandomForestRegressor


# Define Random Forest algorithm parameters with default values
@dataclass
class RFParms:
    num_trees: int = 100
    num_features_per_split: int = -1


def train_random_forests_po(X: np.ndarray, c: np.ndarray, rf_alg_parms: RFParms = RFParms()) -> List[
    RandomForestRegressor]:
    """
    Train d different Random Forests regression models, each to predict a different
    component of the cost vector as a function of the features.

    Args:
        X: p x n training set feature matrix
        c: d x n training set matrix of cost vectors
        rf_alg_parms: RF parameters object with num_trees and num_features_per_split

    Returns:
        List of d RandomForestRegressor models, one for each cost component
    """
    # Unpack parameters
    num_trees = rf_alg_parms.num_trees
    num_features_per_split = rf_alg_parms.num_features_per_split

    # Dimension check
    p, n = X.shape
    d, n2 = c.shape
    if n != n2:
        raise ValueError("Dimensions of the input are mismatched.")

    # Transpose X to n x p (samples x features) as expected by scikit-learn
    X_t = X.T.copy()

    # Set default number of features per split if not specified
    if num_features_per_split < 1:
        num_features_per_split = int(np.ceil(p / 3))

    # Train one RF model for each component of c
    rf_model_list = []
    for j in range(d):
        c_vec = c[j, :]  # Target vector for the j-th component
        rf_model = RandomForestRegressor(
            n_estimators=num_trees,
            max_features=num_features_per_split,
            random_state=None  # No fixed seed here; could add one if needed
        )
        rf_model.fit(X_t, c_vec)
        rf_model_list.append(rf_model)

    return rf_model_list


def predict_random_forests_po(rf_model_list: List[RandomForestRegressor], X_new: np.ndarray) -> np.ndarray:
    """
    Predict cost vectors using a list of Random Forest models.

    Args:
        rf_model_list: List of d RF models, each predicting a cost component
        X_new: p x n feature matrix of new observations

    Returns:
        d x n matrix of predictions
    """
    # Dimensions
    p, n = X_new.shape
    d = len(rf_model_list)

    # Transpose X_new to n x p (samples x features) as expected by scikit-learn
    X_new_t = X_new.T.copy()

    # Predict for each component
    preds = np.zeros((d, n))
    for j in range(d):
        preds[j, :] = rf_model_list[j].predict(X_new_t)

    return preds


# Example usage (for testing)
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    p, n, d = 5, 100, 3
    X = np.random.randn(p, n)
    c = np.random.randn(d, n)

    # Train RF models
    rf_parms = RFParms(num_trees=50, num_features_per_split=2)
    rf_models = train_random_forests_po(X, c, rf_parms)

    # Predict on new data
    X_new = np.random.randn(p, 20)
    preds = predict_random_forests_po(rf_models, X_new)

    print(f"Trained {len(rf_models)} RF models")
    print(f"Predictions shape: {preds.shape}")
    print(f"Sample predictions:\n{preds[:, :5]}")