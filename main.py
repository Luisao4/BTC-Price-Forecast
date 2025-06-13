import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from prophet import Prophet
import numpy as np

# Load data
csv_file = 'data.csv'  # Adjust path if needed
try:
    df = pd.read_csv(csv_file)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# Convert Unix timestamp to datetime
df['ds'] = pd.to_datetime(df['time'], unit='s', errors='coerce')

# Check for invalid timestamps
if df['ds'].isna().any():
    print(f"Found {df['ds'].isna().sum()} invalid timestamps:")
    print(df[df['ds'].isna()][['time', 'ds']])
    df = df.dropna(subset=['ds'])

# Verify data start date
print(f"Data starts at: {df['ds'].min()}")
print(f"Data ends at: {df['ds'].max()}")

# Prepare BTC OHLC for candlestick plot
btc_df = df[['ds', 'open', 'high', 'low', 'close']].copy()
btc_df = btc_df.rename(columns={'ds': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
btc_df.set_index('Date', inplace=True)

# Prepare ticker data for Prophet
ticker_df = df[['ds', 'Lag 0']].copy()
ticker_df = ticker_df.rename(columns={'Lag 0': 'y'})
ticker_df['y'] = pd.to_numeric(ticker_df['y'], errors='coerce').interpolate(method='linear')

# Initialize Prophet model
model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0
)

# Add H2 seasonality (24h and 12h cycles)
model.add_seasonality(name='daily_24h', period=24/2, fourier_order=10)
model.add_seasonality(name='half_daily_12h', period=12/2, fourier_order=5)

# Fit Prophet model
try:
    model.fit(ticker_df[['ds', 'y']])
except Exception as e:
    print(f"Error fitting Prophet model: {e}")
    exit(1)

# Create future dataframe (forecast 1000 hours = 500 H2 bars)
future = model.make_future_dataframe(periods=500, freq='2h')
forecast = model.predict(future)

# --- NEW ADDITION FOR VISUAL CONTINUITY ---
# Get the last actual historical y value and its corresponding date
last_actual_y = ticker_df['y'].iloc[-1]
last_historical_date = ticker_df['ds'].max()

# Force the forecast's 'yhat' at the last historical date to match the actual historical 'y'
# This ensures visual continuity between the historical and forecast lines
if last_historical_date in forecast['ds'].values:
    forecast.loc[forecast['ds'] == last_historical_date, 'yhat'] = last_actual_y
# --- END NEW ADDITION ---

# Create figure with two subplots
fig = plt.figure(figsize=(14, 10))

# Top pane: BTC candlestick chart
ax1 = plt.subplot2grid((5, 1), (0, 0), rowspan=3)
mpf.plot(
    btc_df,
    type='candle',
    ax=ax1,
    volume=False,
    style='yahoo',
    datetime_format='%Y-%m-%d',
    xrotation=45,
    warn_too_much_data=10000  # Increased for large dataset
)
ax1.set_ylabel('BTC Price (USD)')
ax1.set_title('BTC Candlestick (H2)')

# Bottom pane: Ticker and forecast (unscaled)
ax2 = plt.subplot2grid((5, 1), (3, 0), rowspan=2)

# Plot historical data (blue line)
ax2.plot(ticker_df['ds'], ticker_df['y'], color='blue', label='Ticker (Historical)', linewidth=2)

# Plot forecast data starting from the last historical date
forecast_future = forecast[forecast['ds'] >= last_historical_date]
ax2.plot(forecast_future['ds'], forecast_future['yhat'], color='purple', label='Ticker (Forecast)', linewidth=2)

ax2.set_ylabel('Ticker Value (Scaled M2)')
ax2.set_xlabel('Date')
ax2.legend(loc='upper left')
ax2.tick_params(axis='x', rotation=45)

# Layout and save
plt.tight_layout()
plt.savefig('btc_ticker_plot.png')
plt.show()

# Save forecast (note: the saved forecast for the first point will now reflect the forced value)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('ticker_forecast.csv', index=False)
print("Forecast saved to 'ticker_forecast.csv'")
print("Plot saved to 'btc_ticker_plot.png'")