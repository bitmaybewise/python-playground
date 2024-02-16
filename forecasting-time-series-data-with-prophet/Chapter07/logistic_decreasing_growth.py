import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
import numpy as np
import random

random.seed(42)  # set random seed for repeatability

x = pd.to_datetime(
    pd.date_range("1995-01", "2035-02", freq="M").strftime("%Y-%b").tolist()
)
# create logistic curve
y = [1 - 1 / (1 + np.e ** (-0.03 * (val - 50))) for val in range(len(x))]
# add sinusoidal variation
y = [
    y[idx] + y[idx] * 0.05 * np.sin((idx - 2) * (360 / 12) * (np.pi / 180))
    for idx in range(len(y))
]
# add noise
y = [val + 5 * val * random.uniform(-0.01, 0.01) for val in y]
y = [int(500 * val) for val in y]  # scale up

df2 = pd.DataFrame({"ds": pd.to_datetime(x), "y": y})
df2 = df2[df2["ds"].dt.year < 2006]
df2["cap"] = 500
df2["floor"] = 0

model = Prophet(
    growth="logistic", yearly_seasonality=3, seasonality_mode="multiplicative"
)
model.fit(df2)
future = model.make_future_dataframe(periods=12 * 10, freq="M")
future["cap"] = 500
future["floor"] = 0
forecast = model.predict(future)
fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast, cp_linestyle="")
plt.show()
