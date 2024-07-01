import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from matplotlib.ticker import FuncFormatter



# Define the ticker symbol and date range
ticker_symbol = "Goog"
start_date = "2022-01-01"
end_date = "2023-11-07"
# Download historical data
historical_data = yf.download(ticker_symbol, start=start_date, end=end_date)

# Print the historical data
print(historical_data)


# Create DataFrames for Ticker prices
#msft_df = pd.DataFrame({'Date': pd.date_range(start='2018-01-01', periods=len(msft_price), freq='D'), 'MSFT Price': msft_price})
ticker_df = pd.DataFrame({'Date': historical_data.index, ticker_symbol+' Price': historical_data['Adj Close']})


# Calculate and plot autocorrelation for MSFT and BTC
def plot_autocorrelation(data, title):
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(data['Date'], data[title+' Price'], label='Price')
    plt.title(f'{title} Price')
    plt.grid(True)

    # Rotate the x-axis labels by 60 degrees
    plt.xticks(rotation=60)
    plt.xlabel('Date')
    plt.ylabel(ticker_symbol+' Price')


    plt.subplot(1, 2, 2)
    #lags31 = np.arange(1, 31)
    #autocorr31 = [data[title+' Price'].autocorr(lag) for lag in lags31]
    #plt.bar(lags31, autocorr31, alpha=0.5)
    #plt.plot(lags31, autocorr31, 'r-', label='31 day lag')

    lags93 = np.arange(1, 93)
    autocorr93 = [data[title+' Price'].autocorr(lag) for lag in lags93]
    plt.bar(lags93, autocorr93, alpha=0.5)
    plt.plot(lags93, autocorr93, 'g-', label='93 day lag')

    plt.title(f'Autocorrelation for {title}')
    plt.xlabel("Lag [days]")
    plt.ylabel("Autocorrelation")
    plt.grid(True)
    #plt.legend()

    plt.tight_layout()

plot_autocorrelation(ticker_df, ticker_symbol)

# Example of Moving Average and Oscillator
def moving_average(data, title, window):
    return data[title+' Price'].rolling(window=window).mean()

def oscillator(data, title, short_window, long_window):
    short_ma = moving_average(data, title, short_window)
    long_ma = moving_average(data, title, long_window)
    oscillator = short_ma - long_ma
    return oscillator

ticker_df[ticker_symbol+' MA 10'] = moving_average(ticker_df, ticker_symbol, 10)
ticker_df[ticker_symbol+' MA 50'] = moving_average(ticker_df, ticker_symbol, 50)
ticker_df[ticker_symbol+' Oscillator'] = oscillator(ticker_df, ticker_symbol, 10, 50)


# Plot Moving Averages and Oscillators for ticker
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(ticker_df[ticker_symbol+' Price'], label=ticker_symbol+' Price', alpha=0.5)
plt.plot(ticker_df[ticker_symbol+' MA 10'], label='MA 10', linestyle='--')
plt.plot(ticker_df[ticker_symbol+' MA 50'], label='MA 50', linestyle='--')
plt.title(ticker_symbol+' Price with Moving Averages')


# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=60)
plt.xlabel('Date')
plt.ylabel(ticker_symbol+' Price')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ticker_df[ticker_symbol+' Oscillator'], label='Oscillator', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title(ticker_symbol+' Oscillator')
plt.grid(True)

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=60)
plt.xlabel('Date')
plt.ylabel(ticker_symbol+' Price')

"""
# Plot Moving Averages and Oscillators for BTC
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(btc_df['BTC Price'], label='BTC Price', alpha=0.5)
plt.plot(btc_df['BTC MA 10'], label='MA 10', linestyle='--')
plt.plot(btc_df['BTC MA 50'], label='MA 50', linestyle='--')
plt.title('BTC Price with Moving Averages')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(btc_df['BTC Oscillator'], label='Oscillator', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('BTC Oscillator')
plt.grid(True)
"""
plt.tight_layout()
plt.show()
plt.clf()


# Random _______________________________________________________________________

ticker_symbol = 'MSFT'
# Generate sample historical price data for MSFT and BTC
np.random.seed(0)
ticker_price = np.cumsum(np.random.randn(252 * 5))  # 5 years of daily data
#btc_price = np.cumsum(np.random.randn(252 * 5))  # 5 years of daily data
ticker_df = pd.DataFrame({'Date': pd.date_range(start='2018-01-01', periods=len(ticker_price), freq='D'), ticker_symbol+' Price':ticker_price})

plot_autocorrelation(ticker_df, ticker_symbol)
ticker_df[ticker_symbol+' MA 10'] = moving_average(ticker_df, ticker_symbol, 10)
ticker_df[ticker_symbol+' MA 50'] = moving_average(ticker_df, ticker_symbol, 50)
ticker_df[ticker_symbol+' Oscillator'] = oscillator(ticker_df, ticker_symbol, 10, 50)

# Plot Moving Averages and Oscillators for MSFT
plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
plt.plot(ticker_df[ticker_symbol+' Price'], label=ticker_symbol+' Price', alpha=0.5)
plt.plot(ticker_df[ticker_symbol+' MA 10'], label='MA 10', linestyle='--')
plt.plot(ticker_df[ticker_symbol+' MA 50'], label='MA 50', linestyle='--')
plt.title(ticker_symbol+' Price with Moving Averages')


# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=60)
plt.xlabel('Date')
plt.ylabel(ticker_symbol+' Price')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ticker_df[ticker_symbol+' Oscillator'], label='Oscillator', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title(ticker_symbol+' Oscillator')
plt.grid(True)

# Rotate the x-axis labels by 60 degrees
plt.xticks(rotation=60)
plt.xlabel('Date')
plt.ylabel(ticker_symbol+' Price')

plt.tight_layout()
plt.show()
plt.clf()


import sys
sys.exit()
# Razlika med np.corellate in pd.autocorr

ticker_pd = pd.DataFrame({ticker_symbol+' Price': historical_data['Adj Close']})
ticker_np = np.array(historical_data['Adj Close'])
#print(ticker_np)

lags93 = np.arange(1, 93)
autocorr_pd = np.array([ticker_pd[ticker_symbol+' Price'].autocorr(lag) for lag in lags93])
#autocorr_np = np.correlate(ticker_np, ticker_np[:len(ticker_np)-91])
corr = ticker_np[::-1]
autocorr_np = np.array([np.correlate(ticker_np[len(ticker_np)-92+lag], ticker_np, mode='full') for lag in lags93])
print(autocorr_np)
print(len(autocorr_np))
difference = np.abs(autocorr_np - autocorr_np)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.bar(lags93, autocorr_pd, color='r', alpha=0.5)
plt.plot(lags93, autocorr_pd, 'r-', alpha=0.5, lw=0.7, label='pd.autocorr')

#plt.bar(lags93, autocorr_np, color='g', alpha=0.5)
#plt.plot(lags93, autocorr_np, 'g-', alpha=0.5, lw=0.7, label='np.correlate')

plt.title(f'Autocorrelation')
plt.xlabel("Lag [days]")
plt.ylabel("Autocorrelation")
plt.grid(True)


plt.subplot(1, 2, 2)

plt.plot(lags93, difference)

plt.tight_layout()
plt.show()