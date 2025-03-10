import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    train_loss, test_loss = [], []
    num_learners_range = np.arange(1, n_learners + 1)
    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)
    for t in num_learners_range:
        train_loss.append(adaboost.partial_loss(train_X, train_y, t))
        test_loss.append(adaboost.partial_loss(test_X, test_y, t))
    go.Figure([go.Scatter(x=num_learners_range, y=train_loss, mode='lines', name='Train Loss'),
               go.Scatter(x=num_learners_range, y=test_loss, mode='lines', name='Test Loss')]).update_layout(
        title="Train and Test Loss as a function of number of adaboost iterations",
        xaxis_title="Num Iterations", yaxis_title="Loss").show()

    # Question 2: Plotting decision surfaces
    iters = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    subplots = make_subplots(rows=2, cols=2, subplot_titles=[f"{i} Classifiers" for i in iters],
                             horizontal_spacing=0.01,
                             vertical_spacing=.03)
    subplots.update_layout(title=f"Decision Boundaries as function of different iterations",
                           margin=dict(t=100, )).update_xaxes(visible=False).update_yaxes(visible=False)
    for i, t in enumerate(iters):
        subplots.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, t), lims[0], lims[1],
                                              showscale=False),
                             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                        marker=dict(color=test_y.astype(int), symbol=class_symbols[test_y.astype(int)],
                                                    colorscale=class_colors(2),
                                                    line=dict(color="black", width=1)))], rows=(i // 2) + 1,
                            cols=(i % 2) + 1)
    subplots.show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_index = np.argmin(test_loss) + 1
    go.Figure([decision_surface(lambda X: adaboost.partial_predict(X, min_loss_index), lims[0], lims[1],
                                showscale=False),
               go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=test_y.astype(int), symbol=class_symbols[test_y.astype(int)],
                                      colorscale=class_colors(2),
                                      line=dict(color="black", width=1)))]).update_layout(
        title=f"Decision boundaries of ensemble size {min_loss_index} that gave the best accuracy which is:"
              f"{1 - test_loss[min_loss_index - 1]}").update_xaxes(
        visible=False).update_yaxes(visible=False).show()

    # Question 4: Decision surface with weighted samples
    D = adaboost.D_ / np.max(adaboost.D_) * 5
    go.Figure([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
               go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                          marker=dict(color=train_y.astype(int), colorscale=class_colors(2),
                                      line=dict(color="black", width=1), size=D))]).update_layout(
        title=f"Training set decision boundaries and points size as a function of each point's weight", width=1000,
        height=500).update_xaxes(visible=False).update_yaxes(visible=False).show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
