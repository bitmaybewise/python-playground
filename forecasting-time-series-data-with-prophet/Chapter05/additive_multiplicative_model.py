import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv("../data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.columns = ["ds", "y"]

model_a = Prophet(seasonality_mode="additive", yearly_seasonality=4)
model_a.fit(df)
forecast_a = model_a.predict()
fig_a = model_a.plot(forecast_a)
fig_a.savefig("additive_model.png")
# plt.show()

model_m = Prophet(seasonality_mode="multiplicative", yearly_seasonality=4)
model_m.fit(df)
forecast_m = model_m.predict()
fig_m = model_m.plot(forecast_m)
fig_m.savefig("multiplicative_model.png")
# plt.show()
