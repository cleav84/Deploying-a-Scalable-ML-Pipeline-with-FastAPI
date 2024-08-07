import pytest
import numpy as np
from sklearn.datasets import make_classification
from ml.model import train_model, inference, compute_model_metrics


def test_train_model():
    """
    # Test if the train_model function trains a model correctly and checks correct length of predictions
    """
    X_train, y_train = make_classification(
        n_samples = 1000,
        n_features = 20,
        n_informative = 2,
        n_redundant = 10,
        random_state = 42
    )
    model = train_model(X_train, y_train)
    assert model is not None
    assert hasattr(model, 'predict')

    # Check for predictions and correct length
    X_test, _ = make_classification(
        n_samples = 10,
        n_features = 20,
        n_informative = 2,
        n_redundant = 10,
        random_state = 42
    )
    preds = inference(model, X_test)
    assert len(preds) == 10


def test_compute_model_metrics():
    """
    # Test if the compute_model_metrics function returns correct metrics
    """
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 0, 1, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert precision == pytest.approx(0.8333, rel=1e-3)
    assert recall == pytest.approx(0.8333, rel=1e-3)
    assert fbeta == pytest.approx(0.8333, rel=1e-3)


def test_inference():
    """
    # Test if the inference function returns the correct predictions
    """
    X_train, y_train = make_classification(
        n_samples = 1000,
        n_features = 20,
        n_informative = 2,
        n_redundant = 10,
        random_state = 42
    )
    model = train_model(X_train, y_train)

    X_test, _ = make_classification(
        n_samples = 10,
        n_features = 20,
        n_informative = 2,
        n_redundant = 10,
        random_state = 42
    )
    preds = inference(model, X_test)
    assert len(preds) == 10
    assert all(isinstance(pred, np.integer) for pred in preds)
