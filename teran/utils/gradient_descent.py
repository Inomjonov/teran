import numpy as np

def gradient_descent(X, y, weights, bias, learning_rate, epochs):
    n_samples = X.shape[0]

    for _ in range(epochs):
        y_pred = np.dot(X, weights) + bias
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias
