import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv("../data/divvy_daily.csv")
df = df[["date", "rides"]]
df["date"] = pd.to_datetime(df["date"])
df.columns = ["ds", "y"]

model = Prophet(
    growth="linear", seasonality_mode="multiplicative", yearly_seasonality=4
)

model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig = model.plot(forecast)
add_changepoints_to_plot(fig.gca(), model, forecast, cp_linestyle="")
plt.show()
