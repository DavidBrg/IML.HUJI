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
    samples = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    samples['DayOfYear'] = samples['Date'].dt.dayofyear
    samples = samples[samples["Temp"] >= -70]
    return samples


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    # graph 1
    isr_data = data[data['Country'] == 'Israel']
    isr_data['Year'] = isr_data['Year'].astype(str)
    px.scatter(isr_data, x='DayOfYear', y='Temp', title='Temperature as a function of day of year', color='Year').show()

    # graph 2
    isr_month_tmp_std = isr_data.groupby('Month').agg(std_Temp=('Temp', 'std')).reset_index()
    px.bar(isr_month_tmp_std, x='Month', y='std_Temp', title="Israel temperature std as a function of month is the year").show()

    # Question 3 - Exploring differences between countries
    country_month_to_temp_std = data.groupby(['Country', 'Month']).agg(std_Temp=('Temp', 'std'), mean_Temp=('Temp', 'mean')).reset_index()
    px.line(country_month_to_temp_std, x='Month', y='mean_Temp', color='Country', error_y='std_Temp', title="Temperature std as a function of month and country").show()

    # Question 4 - Fitting model for different values of `k`
    isr_temps = isr_data.pop('Temp')
    train_X, train_y, test_X, test_y = split_train_test(isr_data, isr_temps)
    kloss = []
    degrees = np.arange(1, 11, 1).astype(int)
    for k in degrees:
        pf = PolynomialFitting(k)
        pf.fit(train_X['DayOfYear'].to_numpy(), train_y.to_numpy())
        kloss.append(round(pf.loss(test_X['DayOfYear'].to_numpy(), test_y.to_numpy()), 2))
        print(f"Degree: {k}, Loss: {kloss[k - 1]}")
    px.bar(isr_data, x=degrees, y=kloss, labels={'x': 'Degree', 'y': 'Loss'}, title='Loss as a function of the degree').show()

    # Question 5 - Evaluating fitted model on different countries
    pf4 = PolynomialFitting(4)
    pf4.fit(isr_data["DayOfYear"], isr_temps)
    countries = data["Country"].loc[data["Country"] != "Israel"].unique()
    closs = []
    for country in countries:
        cdata = data[data['Country'] == country]
        closs.append(pf4.loss(cdata['DayOfYear'], cdata['Temp']))
    px.bar(pd.DataFrame({"Country": countries, "Loss": closs}), x="Country", y="Loss", title='Temperature loss values as a function of country on Israeli trained fitting').show()
