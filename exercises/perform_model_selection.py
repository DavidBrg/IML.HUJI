from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = pd.Series(np.linspace(-1.2, 2, n_samples))
    y_true = (X + 3) * (X + 2) * (X + 1) * (X - 1) * (X - 2)
    y_noisy = y_true + np.random.randn(n_samples) * noise
    X_train, y_train, X_test, y_test = split_train_test(X, y_noisy, (2 / 3))
    X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
    go.Figure([go.Scatter(name='Model', x=X, y=y_true, mode='markers'),
               go.Scatter(name='Train', x=X_train, y=y_train, mode='markers'),
               go.Scatter(name='Test', x=X_test, y=y_test, mode='markers')]).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_range = np.arange(11)
    train_errors = [0] * len(k_range)
    validation_errors = [0] * len(k_range)
    for k in k_range:
        train_errors[k], validation_errors[k] = cross_validate(PolynomialFitting(k), X_train, y_train,
                                                               mean_square_error)
    go.Figure([go.Scatter(name='Train errors', x=k_range, y=train_errors, mode='markers+lines'),
               go.Scatter(name='Validation errors', x=k_range, y=validation_errors, mode='markers+lines')]).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_index = int(np.argmin(validation_errors))
    pf = PolynomialFitting(min_index)
    pf.fit(X_train, y_train)
    print(f"Best k is: {min_index}\n"
          f"Validation error of: {round(validation_errors[min_index], 2)}\n"
          f"Test set error of: {round(mean_square_error(y_test, pf.predict(X_test)), 2)}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    X_train, y_train, X_test, y_test = X[:n_samples, :], y[:n_samples], X[n_samples:, :], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    reg_params = np.linspace(0, 2.5, num=n_evaluations)
    ridge_train_errors = [0] * n_evaluations
    ridge_validation_errors = [0] * n_evaluations
    lasso_train_errors = [0] * n_evaluations
    lasso_validation_errors = [0] * n_evaluations
    for i, k in enumerate(reg_params):
        ridge_train_errors[i], ridge_validation_errors[i] = cross_validate(RidgeRegression(k), X_train, y_train,
                                                                           mean_square_error)
        lasso_train_errors[i], lasso_validation_errors[i] = cross_validate(Lasso(k), X_train, y_train,
                                                                           mean_square_error)
    go.Figure([go.Scatter(name='Ridge train errors', x=reg_params, y=ridge_train_errors, mode='lines'),
               go.Scatter(name='Ridge validation errors', x=reg_params, y=ridge_validation_errors, mode='lines'),
               go.Scatter(name='Lasso train errors', x=reg_params, y=lasso_train_errors, mode='lines'),
               go.Scatter(name='Lasso validation errors', x=reg_params, y=lasso_validation_errors,
                          mode='lines')]).show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_reg_param = reg_params[np.argmin(ridge_validation_errors)]
    lasso_reg_param = reg_params[np.argmin(lasso_validation_errors)]

    ridge = RidgeRegression(ridge_reg_param)
    lasso = Lasso(lasso_reg_param)
    lr = LinearRegression()

    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    print(f"Ridge best regularization parameter: {ridge_reg_param}")
    print(f"Lasso best regularization parameter: {lasso_reg_param}")

    print(f"Ridge test error: {ridge.loss(X_test, y_test)}")
    print(f"Lasso test error: {mean_square_error(y_test, lasso.predict(X_test))}")
    print(f"Linear Regression test error: {lr.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
