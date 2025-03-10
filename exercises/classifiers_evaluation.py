from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def loss_callback(p, data, sample):
            losses.append(p.loss(X, y))

        Perceptron(callback=loss_callback).fit(X, y)

        fig = go.Figure(go.Scatter(x=list(range(len(losses))), y=losses, mode='lines'),
                        layout=go.Layout(title=f"{n} Loss as function of fitting iteration",
                                         xaxis_title="Iteration", yaxis_title="Loss"))
        fig.write_image(f"{n}_Loss_as_function_of_fitting_iteration.png")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset("../datasets/" + f)

        # Fit models and predict over training set
        lda = LDA()
        gnb = GaussianNaiveBayes()

        lda.fit(X, y)
        gnb.fit(X, y)

        y_lda = lda.predict(X)
        y_gnb = gnb.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
        subplots = make_subplots(rows=1, cols=2, subplot_titles=[f"GNB Model, Accuracy: {accuracy(y, y_gnb)}",
                                                                 f"LDA Model, Accuracy: {accuracy(y, y_lda)}"],
                                 horizontal_spacing=0.01, vertical_spacing=.03)
        subplots.update_layout(title=f"Decision Boundaries Of Models on {f} Dataset",
                               margin=dict(t=100, )).update_xaxes(visible=False).update_yaxes(visible=False)
        data = [[], []]
        # Add traces for data-points setting symbols and colors
        for i, model in enumerate([gnb, lda]):
            y_predict = model.fit(X, y).predict(X)
            data[i] = [decision_surface(model.fit(X, y).predict, lims[0], lims[1], showscale=False),
                       go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                  marker=dict(color=y_predict, symbol=class_symbols[y], colorscale=class_colors(3),
                                              line=dict(color="black", width=1)))]

            # Add `X` dots specifying fitted Gaussians' means
            data[i].append(go.Scatter(x=model.mu_[:, 0], y=model.mu_[:, 1], mode="markers", showlegend=False,
                                      marker=dict(color="black", symbol="x")))

            # Add ellipses depicting the covariances of the fitted Gaussians
            for j in range(len(model.mu_)):
                if model == gnb:
                    cov = np.diag(gnb.vars_[j])
                else:
                    cov = lda.cov_
                data[i].append(get_ellipse(model.mu_[j], cov))

            subplots.add_traces(data[i], rows=1, cols=1 + i)

        subplots.write_image(f"{f}_decision_boundary.png")
        subplots.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
