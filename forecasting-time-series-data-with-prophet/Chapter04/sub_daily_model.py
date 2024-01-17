import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

data = pd.read_csv("../data/divvy_hourly.csv")
df = pd.DataFrame({"ds": pd.to_datetime(data["date"]), "y": data["rides"]})
model = Prophet(seasonality_mode="multiplicative")
model.fit(df)
future = model.make_future_dataframe(periods=365 * 24, freq="h")
forecast = model.predict(future)
fig = model.plot(forecast)
fig.savefig("sub_daily_model_plot.png")
fig2 = model.plot_components(forecast)
fig2.savefig("sub_daily_model_components_plot.png")
# plt.show()
