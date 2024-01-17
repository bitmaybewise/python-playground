import pandas as pd
from prophet import Prophet

df = pd.read_csv(
    "../data/sunspots.csv",
    usecols=[
        "Date",
        "Monthly Mean Total Sunspot Number",
    ],
)
df["Date"] = pd.to_datetime(df["Date"])
df.columns = ["ds", "y"]
model = Prophet(seasonality_mode="multiplicative", yearly_seasonality=False)
model.add_seasonality(
    name="11-year cycle",
    period=11 * 365.25,  # always counted in days
    fourier_order=5,
)
model.fit(df)
future = model.make_future_dataframe(periods=240, freq="M")
forecast = model.predict(future)
fig2 = model.plot_components(forecast)
fig2.savefig("eleven_year_cycle_components.png")
