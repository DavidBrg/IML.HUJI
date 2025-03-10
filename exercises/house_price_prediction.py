from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # # get all features
    # features = pd.read_csv(filename).dropna().drop_duplicates()
    # # remove unnecessary features:
    # features.drop(columns=["id", "date", "lat", "long"], inplace=True)
    # # limit values in features:
    # feature_limits = {"price": [1],
    #                   "bedrooms": [0],
    #                   "bathrooms": [0],
    #                   "sqft_living": [1],
    #                   "sqft_lot": [1],
    #                   "floors": [0],
    #                   "waterfront": [0, 1],
    #                   "view": [0, 4],
    #                   "condition": [1, 5],
    #                   "grade": [1, 15],
    #                   "sqft_above": [1],
    #                   "sqft_basement": [0],
    #                   "yr_built": [1],
    #                   "yr_renovated": [0],
    #                   "zipcode": [1],
    #                   "sqft_living15": [1],
    #                   "sqft_lot15": [1]}
    # for f, limits in feature_limits.items():
    #     if len(limits) == 2:
    #         features = features[features[f] <= limits[1]]
    #     features = features[features[f] >= limits[0]]
    # # get dummies
    # features["zipcode"] = features["zipcode"].astype(str)
    # features = pd.get_dummies(features, columns=["zipcode"])
    # latest_year = np.max(features["yr_built"])
    # features["yr_renovated"] = np.where(((features["yr_built"] > latest_year - 30) | (features["yr_renovated"] >= latest_year - 15)), 1, 0)
    # y = features["price"]
    # features.drop("price", axis=1, inplace=True)
    # return features, y

    df = pd.read_csv(filename).dropna().drop_duplicates()
    df = df.drop(["id", "lat", "long", "date", "sqft_lot15", "sqft_living15"], axis=1)
    df["zipcode"] = df["zipcode"].astype(int)

    for c in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built"]:
        df = df[df[c] > 0]
    for c in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[c] >= 0]

    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]

    df["recently_renovated"] = np.where(df["yr_renovated"] >= np.percentile(df.yr_renovated.unique(), 70), 1, 0)
    df = df.drop("yr_renovated", axis=1)

    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df = df.drop("yr_built", axis=1)

    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built_', columns=['decade_built'])

    # Removal of outliers (Notice that there exists methods for better defining outliers
    # but for this course this suffices
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]

    return df.drop("price", axis=1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # for feature in X:
    #     if feature.startswith("zipcode"):
    #         continue
    #     correlation = X[feature].to_frame().corrwith(y)[0]
    #     px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", title=f"{feature} & price correlation: {correlation}",
    #                labels={"x": f"{feature}", "y": "price"}).write_image(f"{output_path}/{feature}_price_corr.png")

    X = X.loc[:, ~(X.columns.str.contains('^zipcode_', case=False) |
                   X.columns.str.contains('^decade_built_', case=False))]

    for f in X:
        rho = np.cov(X[f], y)[0, 1] / (np.std(X[f]) * np.std(y))

        fig = px.scatter(pd.DataFrame({'x': X[f], 'y': y}), x="x", y="y", trendline="ols",
                         color_discrete_sequence=["black"],
                         title=f"Correlation Between {f} Values and Response <br>Pearson Correlation {rho}",
                         labels={"x": f"{f} Values", "y": "Response Values"})
        fig.write_image(output_path + f"/pearson.correlation.{f}.png")


if __name__ == '__main__':
    # np.random.seed(0)
    # # Question 1 - Load and preprocessing of housing prices dataset
    # data, price = load_data("../datasets/house_prices.csv")
    #
    # # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(data, price)
    #
    # # Question 3 - Split samples into training- and testing sets.
    # train_data, train_price, test_data, test_price = split_train_test(data, price)
    #
    # # Question 4 - Fit model over increasing percentages of the overall training data
    # # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    # #   1) Sample p% of the overall training data
    # #   2) Fit linear model (including intercept) over sampled set
    # #   3) Test fitted model over test set
    # #   4) Store average and variance of loss over test set
    # # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    # mean = []
    # std = []
    # percents = np.arange(10, 101, 1).astype(int)
    # for p in percents:
    #     ploss = []
    #     for i in range(10):
    #         lr = LinearRegression()
    #         ptrain_data, ptrain_price, ptest_data, ptest_price = split_train_test(train_data, train_price, p / 100)
    #         lr.fit(ptrain_data.to_numpy(), ptrain_price.to_numpy())
    #         ploss.append(lr.loss(test_data.to_numpy(), test_price.to_numpy()))
    #     mean.append(np.mean(ploss))
    #     std.append(np.std(ploss))
    # mean = np.array(mean)
    # std = np.array(std)
    #
    # go.Figure([go.Scatter(x=percents, y=mean, name="Mean Loss", mode="markers+lines", marker=dict(color="black", size=1)),
    #            go.Scatter(x=percents, y=mean - 2 * std, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False),
    #            go.Scatter(x=percents, y=mean + 2 * std, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False)],
    #           layout=go.Layout(title=r"$\text{Mean Loss of house pricing as a function percentage of training data}$", xaxis={"title": "Percent"}, yaxis={"title": "Mean Loss"})).show()

    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    df, price = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df.loc[:, df.columns != "price"], price)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(df, price)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    ps = list(range(10, 101))
    results = np.zeros((len(ps), 10))
    for i, p in enumerate(ps):
        for j in range(results.shape[1]):
            _X = train_X.sample(frac=p / 100.0)
            _y = train_y.loc[_X.index]
            results[i, j] = LinearRegression(include_intercept=True).fit(_X, _y).loss(test_X, test_y)

    m, s = results.mean(axis=1), results.std(axis=1)
    fig = go.Figure([go.Scatter(x=ps, y=m - 2 * s, fill=None, mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m + 2 * s, fill='tonexty', mode="lines", line=dict(color="lightgrey")),
                     go.Scatter(x=ps, y=m, mode="markers+lines", marker=dict(color="black"))],
                    layout=go.Layout(title="Test MSE as Function Of Training Size",
                                     xaxis=dict(title="Percentage of Training Set"),
                                     yaxis=dict(title="MSE Over Test Set"),
                                     showlegend=False))
    fig.write_image("mse.over.training.percentage.png")
