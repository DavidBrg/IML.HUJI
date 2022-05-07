from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # # Question 1 - Draw samples and print fitted model
    # univariate_gaussian = UnivariateGaussian()
    # mu = 10
    # sigma = 1
    # X = np.random.normal(mu, sigma, 1000)
    # univariate_gaussian.fit(X)
    # print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")
    #
    # # Question 2 - Empirically showing sample mean is consistent
    # ms = np.linspace(10, 1000, 100).astype(int)
    # estimated_mean = []
    # for m in ms:
    #     univariate_gaussian.fit(X[:m])
    #     estimated_mean.append(np.abs(univariate_gaussian.mu_ - mu))
    # go.Figure(go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'Absolute Difference'),
    #           layout=go.Layout(title=r"$\text{Absolute distance between the estimated and true value of the"
    #                                  r" expectation as a function of the sample size}$",
    #                            xaxis_title="$m\\text{ - number of samples}$", yaxis_title="r$|\hat\mu-\mu|$",
    #                            height=300)).show()
    #
    # # Question 3 - Plotting Empirical PDF of fitted model
    # X.sort()
    # go.Figure(go.Scatter(x=X, y=univariate_gaussian.pdf(X), mode='lines'),
    #           layout=go.Layout(title='Empirical PDF Function Under The Fitted Model',
    #                            xaxis_title="Sample Values", yaxis_title="pdf(x)", height=300)).show()

    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(loc=mu, scale=sigma, size=1000)
    fit = UnivariateGaussian().fit(X)
    print((fit.mu_, fit.var_))

    # Question 2 - Empirically showing sample mean is consistent
    mu_hat = [np.abs(mu - UnivariateGaussian().fit(X[:n]).mu_)
              for n in np.arange(1, len(X) / 10, dtype=np.int) * 10]
    go.Figure(go.Scatter(x=list(range(len(mu_hat))), y=mu_hat, mode="markers", marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="Deviation of Sample Mean Estimation As Function of Sample Size",
                          xaxis_title=r"$\text{Sample Size }n$",
                          yaxis_title=r"$\text{Sample Mean Estimator }\hat{\mu}_n$")) \
        .write_image("mean.deviation.over.sample.size.png")

    # Question 3 - Plotting Empirical PDF of fitted model
    pdfs = np.c_[X, fit.pdf(X)]
    pdfs = pdfs[pdfs[:, 1].argsort()]
    go.Figure(go.Scatter(x=pdfs[:, 0], y=pdfs[:, 1], mode="markers", marker=dict(color="black")),
              layout=dict(template="simple_white",
                          title="Empirical PDF Using Fitted Model",
                          xaxis_title=r"$x$",
                          yaxis_title=r"$\mathcal{N}(x;\hat{\mu},\hat{\sigma}^2)$")) \
        .write_image("empirical.pdf.png")


def test_multivariate_gaussian():
    # # Question 4 - Draw samples and print fitted model
    # multivariate_gaussian = MultivariateGaussian()
    # mu = np.array([0, 0, 4, 0])
    # cov = np.array([[1, 0.2, 0, 0.5],
    #                 [0.2, 2, 0, 0],
    #                 [0, 0, 1, 0],
    #                 [0.5, 0, 0, 1]])
    # X = np.random.multivariate_normal(mu, cov, 1000)
    # multivariate_gaussian.fit(X)
    # print(multivariate_gaussian.mu_)
    # print(multivariate_gaussian.cov_)
    #
    # # Question 5 - Likelihood evaluation
    # ms = np.linspace(-10, 10, 200)
    # likelihood = []
    # for x in ms:
    #     likelihood.append([multivariate_gaussian.log_likelihood(np.array([x, 0, y, 0]), cov, X) for y in ms])
    # go.Figure(go.Heatmap(x=ms, y=ms, z=likelihood), layout=go.Layout(
    #     title=r"$\text{Heatmap of the previously given covariance and }\hat\mu = [f1, 0, f3, 0]$",
    #     xaxis_title=r"$f1$", yaxis_title=r"$f3$", height=500, width=500)).show()
    #
    # # Question 6 - Maximum likelihood
    # max_likelihood = likelihood[0][0]
    # res = [0, 0]
    # for i in range(len(likelihood)):
    #     for j in range(len(likelihood[0])):
    #         if likelihood[i][j] > max_likelihood:
    #             max_likelihood = likelihood[i][j]
    #             res = [i, j]
    # # Maximum lo likelihood couple
    # print(f"{round(ms[res[0]], 3)} {round(ms[res[1]], 3)}")

    # Question 4 - Draw samples and print fitted model
    cov = np.array([[1, .2, 0, .5], [.2, 2, 0, 0], [0, 0, 1, 0], [.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mean=[0, 0, 4, 0], cov=cov, size=1000)
    fit = MultivariateGaussian().fit(X)
    print(np.round(fit.mu_, 3))
    print(np.round(fit.cov_, 3))
    fit.pdf(X)
    # Question 5 - Likelihood evaluation
    ll = np.zeros((200, 200))
    f_vals = np.linspace(-10, 10, ll.shape[0])
    for i, f1 in enumerate(f_vals):
        for j, f3 in enumerate(f_vals):
            ll[i, j] = MultivariateGaussian.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)

    go.Figure(go.Heatmap(x=f_vals, y=f_vals, z=ll),
              layout=dict(template="simple_white",
                          title="Log-Likelihood of Multivariate Gaussian As Function of Expectation of Features 1,3",
                          xaxis_title=r"$\mu_3$",
                          yaxis_title=r"$\mu_1$")) \
        .write_image("log likelihood.heatmap.png")

    # Question 6 - Maximum likelihood
    print("Setup of maximum likelihood (features 1 and 3):",
          np.round(f_vals[list(np.unravel_index(ll.argmax(), ll.shape))], 3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
