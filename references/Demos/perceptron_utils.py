"""
perceptron_utils.py
Shared utility functions for perceptron demos

Contains common mathematical functions used across multiple
perceptron demonstration applications.
"""

import numpy as np


def sigmoid(z):
    """
    Sigmoid activation function.

    Args:
        z: Input value or array (weighted sum)

    Returns:
        Sigmoid output in range (0, 1)
    """
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid function.

    Used in backpropagation for computing gradients.

    Args:
        z: Input value or array

    Returns:
        Derivative of sigmoid at z: σ'(z) = σ(z) * (1 - σ(z))
    """
    sig = sigmoid(z)
    return sig * (1 - sig)


def perceptron_forward(x1, x2, w1, w2, b):
    """
    Forward pass through perceptron.

    Args:
        x1, x2: Input values
        w1, w2: Weight values
        b: Bias term

    Returns:
        Tuple of (weighted_sum, sigmoid_output)
    """
    z = w1 * x1 + w2 * x2 + b
    output = sigmoid(z)
    return z, output


def mean_squared_error(predictions, targets):
    """
    Calculate mean squared error loss.

    Args:
        predictions: Array of predicted values
        targets: Array of target values

    Returns:
        MSE loss value
    """
    return np.mean((predictions - targets) ** 2)


def binary_cross_entropy(predictions, targets, epsilon=1e-15):
    """
    Calculate binary cross-entropy loss.

    Args:
        predictions: Array of predicted probabilities
        targets: Array of target values (0 or 1)
        epsilon: Small value to prevent log(0)

    Returns:
        BCE loss value
    """
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
