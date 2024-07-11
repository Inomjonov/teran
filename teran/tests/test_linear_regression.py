import unittest
import pandas as pd
import numpy as np
from teran.models.linear_regression import LinearRegression

class TestLinearRegression(unittest.TestCase):
    def setUp(self):
        # Create a sample dataset for testing
        np.random.seed(0)
        self.X_train = pd.DataFrame(np.random.rand(100, 5))  # 100 samples, 5 features
        self.y_train = pd.Series(np.random.rand(100) * 10)   # 100 targets

    def test_fit_predict(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        self.assertEqual(predictions.shape, (100, 1))  # Ensure predictions shape is correct

    def test_compute_loss(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_train)
        loss = model._compute_loss(self.y_train.values.reshape(-1, 1), predictions)
        self.assertIsInstance(loss, float)  # Ensure loss calculation returns a float

if __name__ == '__main__':
    unittest.main()
