import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    # samples = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    # samples['DayOfYear'] = samples['Date'].dt.dayofyear
    # samples = samples[samples["Temp"] >= -70]
    # return samples

    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df = df[df.Temp > 0]
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    df["DM"] = df["Date"].dt.strftime('%d-%m')
    return df


if __name__ == '__main__':
    # np.random.seed(0)
    # # Question 1 - Load and preprocessing of city temperature dataset
    # data = load_data("../datasets/City_Temperature.csv")
    #
    # # Question 2 - Exploring data for specific country
    # # graph 1
    # isr_data = data[data['Country'] == 'Israel']
    # isr_data['Year'] = isr_data['Year'].astype(str)
    # px.scatter(isr_data, x='DayOfYear', y='Temp', title='Temperature as a function of day of year', color='Year').show()
    #
    # # graph 2
    # isr_month_tmp_std = isr_data.groupby('Month').agg(std_Temp=('Temp', 'std')).reset_index()
    # px.bar(isr_month_tmp_std, x='Month', y='std_Temp', title="Israel temperature std as a function of month is the year").show()
    #
    # # Question 3 - Exploring differences between countries
    # country_month_to_temp_std = data.groupby(['Country', 'Month']).agg(std_Temp=('Temp', 'std'), mean_Temp=('Temp', 'mean')).reset_index()
    # px.line(country_month_to_temp_std, x='Month', y='mean_Temp', color='Country', error_y='std_Temp', title="Temperature std as a function of month and country").show()
    #
    # # Question 4 - Fitting model for different values of `k`
    # isr_temps = isr_data.pop('Temp')
    # train_X, train_y, test_X, test_y = split_train_test(isr_data, isr_temps)
    # kloss = []
    # degrees = np.arange(1, 11, 1).astype(int)
    # for k in degrees:
    #     pf = PolynomialFitting(k)
    #     pf.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
    #     kloss.append(round(pf.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy()), 2))
    #     print(f"Degree: {k}, Loss: {kloss[k - 1]}")
    # px.bar(isr_data, x=degrees, y=kloss, labels={'x': 'Degree', 'y': 'Loss'}, title='Loss as a function of the degree').show()
    #
    # # Question 5 - Evaluating fitted model on different countries
    # pf4 = PolynomialFitting(4)
    # pf4.fit(isr_data["DayOfYear"], isr_temps)
    # countries = data["Country"].loc[data["Country"] != "Israel"].unique()
    # closs = []
    # for country in countries:
    #     cdata = data[data['Country'] == country]
    #     closs.append(pf4.loss(cdata['DayOfYear'], cdata['Temp']))
    # px.bar(pd.DataFrame({"Country": countries, "Loss": closs}), x="Country", y="Loss", title='Temperature loss values as a function of country on Israeli trained fitting').show()

    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("../datasets/city_temperature.csv")

    # Question 2 - Exploring data for specific country
    sub_df = df[df.Country == "Israel"]
    px.scatter(sub_df, x="DayOfYear", y="Temp", color="Year") \
        .write_image("israel.daily.temperatures.png")
    px.bar(sub_df.groupby(["Month"], as_index=False).agg(std=("Temp", "std")),
           title="Temperature Standard Deviation Over Years", x="Month", y="std") \
        .write_image("israel.monthly.average.temperature.png")

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(["Country", "Month"], as_index=False).agg(mean=("Temp", "mean"), std=("Temp", "std")),
            x="Month", y="mean", error_y="std", color="Country") \
        .update_layout(title="Average Monthly Temperatures",
                       xaxis_title="Month",
                       yaxis_title="Mean Temperature") \
        .write_image("mean.temp.different.countries.png")

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(sub_df.DayOfYear, sub_df.Temp)
    ks = list(range(1, 11))
    loss = np.zeros_like(ks, dtype=float)
    for i, k in enumerate(ks):
        model = PolynomialFitting(k=k).fit(train_X.to_numpy(), train_y.to_numpy())
        loss[i] = np.round(model.loss(test_X.to_numpy(), test_y.to_numpy()), 2)

    loss = pd.DataFrame(dict(k=ks, loss=loss))
    px.bar(loss, x="k", y="loss", text="loss",
           title=r"$\text{Test Error For Different Values of }k$") \
        .write_image("israel.different.k.png")
    print(loss)

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(k=5).fit(sub_df.DayOfYear.to_numpy(), sub_df.Temp.to_numpy())
    px.bar(pd.DataFrame([{"country": c, "loss": round(model.loss(df[df.Country == c].DayOfYear, df[df.Country == c].Temp), 2)}
                         for c in ["Jordan", "South Africa", "The Netherlands"]]),
           x="country", y="loss", text="loss", color="country",
           title="Loss Over Countries For Model Fitted Over Israel") \
        .write_image("test.other.countries.png")
