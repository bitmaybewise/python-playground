import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import numpy as np
import random

random.seed(42)  # set random seed for repeatability

x = pd.to_datetime(
    pd.date_range("1995-01", "2004-02", freq="M").strftime("%Y-%b").tolist()
)
# create logistic curve
y = [1 / (1 + np.e ** (-0.03 * (val - 50))) for val in range(len(x))]
# add sinusoidal variation
y = [
    y[idx] + y[idx] * 0.01 * np.sin((idx - 2) * (360 / 12) * (np.pi / 180))
    for idx in range(len(y))
]
# add noise
y = [val + random.uniform(-0.01, 0.01) for val in y]
y = [int(500 * val) for val in y]  # scale up

df = pd.DataFrame({"ds": pd.to_datetime(x), "y": y})


def set_cap(row, df):
    if row.year < 2007:
        return 500
    else:
        pop_2007 = 500
        idx_2007 = df[df["ds"].dt.year == 2007].index[0]
        idx_date = df[df["ds"] == row].index[0]
        return pop_2007 + 2 * (idx_date - idx_2007)


df["cap"] = df["ds"].apply(set_cap, args=(df,))

model = Prophet(
    growth="logistic", seasonality_mode="multiplicative", yearly_seasonality=3
)
model.fit(df)
future = model.make_future_dataframe(periods=12 * 10, freq="M")
future["cap"] = future["ds"].apply(set_cap, args=(future,))
forecast = model.predict(future)
fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast, cp_linestyle="")
plt.show()
