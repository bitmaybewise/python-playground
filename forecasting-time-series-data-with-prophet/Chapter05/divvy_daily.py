import pandas as pd
from prophet import Prophet

df = pd.read_csv("../data/divvy_daily.csv")
df = df[["date", "rides"]]
df["date"] = pd.to_datetime(df["date"])
df.columns = ["ds", "y"]

model = Prophet(seasonality_mode="multiplicative")
model.fit(df)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
fig = model.plot(forecast)
fig.savefig("divvy_daily.png")
fig2 = model.plot_components(forecast)
fig2.savefig("divvy_daily_components.png")
