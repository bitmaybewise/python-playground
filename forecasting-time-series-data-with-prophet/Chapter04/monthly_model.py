import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

df = pd.read_csv("../data/AirPassengers.csv")
df["Month"] = pd.to_datetime(df["Month"])
df.columns = ["ds", "y"]
model = Prophet(seasonality_mode="multiplicative")
model.fit(df)
future = model.make_future_dataframe(
    periods=12 * 5, freq="MS"  # MS = month start
)  # 12 entries per year, one each month, over 5 years
forecast = model.predict(future)
model.plot(forecast)
plt.show()
