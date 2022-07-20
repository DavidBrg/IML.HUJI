import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.model_selection import cross_validate


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values, weights = [], []

    def gd_state_recorder_callback(**kwargs):
        values.append(kwargs['val'])
        weights.append(kwargs['weights'])

    return gd_state_recorder_callback, values, weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        for module in [L1, L2]:
            callback, values, weights = get_gd_state_recorder_callback()
            gd = GradientDescent(FixedLR(eta), callback=callback)
            gd.fit(module(init), None, None)
            name = f"| Module: {module.__name__} | Eta: {eta}"
            plot_descent_path(module, np.array(weights), name).show()
            go.Figure(go.Scatter(x=np.arange(len(values)), y=values, mode="markers"))\
                .update_layout(title=name, xaxis_title="Iteration", yaxis_title="Norm").show()
            print(f"Minimal value for {name}: {min(values)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    subplots = []
    min_gamma_l1_norm = (0, np.inf)
    for gamma in gammas:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, gamma), callback=callback)
        gd.fit(L1(init), None, None)
        name = f"ExponentialLR | Eta: {eta} | Gamma: {gamma}"
        subplots.append(go.Scatter(x=np.arange(len(values)), y=values, mode="markers+lines", name=name))
        local_min = min(values)
        min_gamma_l1_norm = (gamma, local_min) if local_min < min_gamma_l1_norm[1] else min_gamma_l1_norm

    # Plot algorithm's convergence for the different values of gamma
    go.Figure(subplots).update_layout(xaxis_title="Iteration", yaxis_title="Norm").show()
    print(f"Lowest l1 norm achieved was: {min_gamma_l1_norm[1]} for gamma value {min_gamma_l1_norm[0]}")

    # Plot descent path for gamma=0.95
    for module in {L1, L2}:
        callback, values, weights = get_gd_state_recorder_callback()
        gd = GradientDescent(ExponentialLR(eta, 0.95), callback=callback)
        gd.fit(module(init), None, None)
        name = f"| Module: {module.__name__} | Eta: {eta} | Gamma: 0.95"
        plot_descent_path(module, np.array(weights), name).show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()
    max_iter = 20000
    lambdas = [.001, .002, .005, .01, .02, .05, .1]
    lr = 1e-4
    alpha = .5

    # Plotting convergence rate of logistic regression over SA heart disease data
    lreg = LogisticRegression(solver=GradientDescent(learning_rate=FixedLR(lr), max_iter=max_iter))
    lreg.fit(X_train, y_train)
    fpr, tpr, thresholds = roc_curve(y_train, lreg.predict_proba(X_train))
    go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                          name="Random Class Assignment"),
               go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                          marker_size=5, hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: "
                                                       "%{y:.3f}")],
              layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                               xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                               yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()

    # Calculate best model's test error
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    lr_reg_l1 = LogisticRegression(alpha=best_alpha)
    lr_reg_l1.fit(X_train, y_train)
    print(f"Best alpha is: {best_alpha} with a test error of {lr_reg_l1.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    for penalty in ["l1", "l2"]:
        validation_scores = []
        for lam in lambdas:
            lreg = LogisticRegression(True, GradientDescent(FixedLR(lr), max_iter=max_iter), penalty, lam, alpha)
            train_score, validation_score = cross_validate(lreg, X_train, y_train, misclassification_error)
            validation_scores.append(validation_score)
        min_lambda = lambdas[np.argmin(validation_scores)]
        best_lr = LogisticRegression(True, GradientDescent(FixedLR(lr), max_iter=max_iter), penalty, min_lambda, alpha)
        best_lr.fit(X_train, y_train)
        print(f"Lambda that achieved minimal validation score: {min_lambda} with {penalty} penalty and test error of:"
              f" {best_lr.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
