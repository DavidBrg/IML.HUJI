from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    univariate_gaussian = UnivariateGaussian()
    mu = 10
    sigma = 1
    X = np.random.normal(mu, sigma, 1000)
    univariate_gaussian.fit(X)
    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    for m in ms:
        univariate_gaussian.fit(X[:m])
        estimated_mean.append(np.abs(univariate_gaussian.mu_ - mu))
    go.Figure(go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'Absolute Difference'),
              layout=go.Layout(title=r"$\text{Absolute distance between the estimated and true value of the"
                                     r" expectation as a function of the sample size}$",
                               xaxis_title="$m\\text{ - number of samples}$", yaxis_title="r$|\hat\mu-\mu|$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    X.sort()
    go.Figure(go.Scatter(x=X, y=univariate_gaussian.pdf(X), mode='lines'),
              layout=go.Layout(title='Empirical PDF Function Under The Fitted Model',
                               xaxis_title="Sample Values", yaxis_title="pdf(x)", height=300)).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)
    multivariate_gaussian.fit(X)
    print(multivariate_gaussian.mu_)
    print(multivariate_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    ms = np.linspace(-10, 10, 200)
    likelihood = []
    for x in ms:
        likelihood.append([multivariate_gaussian.log_likelihood(np.array([x, 0, y, 0]), cov, X) for y in ms])
    go.Figure(go.Heatmap(x=ms, y=ms, z=likelihood), layout=go.Layout(
        title=r"$\text{Heatmap of the previously given covariance and }\hat\mu = [f1, 0, f3, 0]$",
        xaxis_title=r"$f1$", yaxis_title=r"$f3$", height=500, width=500)).show()

    # Question 6 - Maximum likelihood
    max_likelihood = likelihood[0][0]
    res = [0, 0]
    for i in range(len(likelihood)):
        for j in range(len(likelihood[0])):
            if likelihood[i][j] > max_likelihood:
                max_likelihood = likelihood[i][j]
                res = [i, j]
    # Maximum lo likelihood couple
    print(f"{round(ms[res[0]], 3)} {round(ms[res[1]], 3)}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
