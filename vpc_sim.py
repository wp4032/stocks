import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def vpc_screen(stocks, period="3mo", window=20):
    results = []

    for ticker in stocks:
        try:
            print(f"Fetching data for {ticker}...")
            # Download OHLCV data
            df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                continue

            volumes = np.array(df["Volume"].iloc[-window:].values).flatten()
            closes = np.array(df["Close"].iloc[-window:].values).flatten()
            
            avg_volume = volumes.sum() / window
            vpc = volumes / avg_volume
            latest_vpc = vpc[-1]

            # Get latest close, resistance/support (simple: 20-day high/low)
            latest = closes[-1]
            resistance = closes.max()  # yesterday's 20d high
            support = closes.min()

            # Conditions for breakout or breakdown
            breakout = latest > resistance and latest_vpc > 1.5
            breakdown = latest < support and latest_vpc > 1.5

            results.append({
                "Ticker": ticker,
                "Close": latest,
                "Volume": volumes[-1],
                "VPC": latest_vpc,
                "Breakout?": breakout,
                "Breakdown?": breakdown
            })

        except Exception as e:
            print(f"Error with {ticker}: {e}")

    return pd.DataFrame(results)

# Example usage:
my_stocks = pd.read_excel('all_tickers_2025.xlsx')['Ticker'].tolist()
df = vpc_screen(my_stocks)

# Save to Excel
current_date = datetime.now().strftime("%Y%m%d")
output_path = f'vpc_analysis_{current_date}.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine
writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
df.to_excel(writer, index=False, sheet_name='Sheet1')

# Access the XlsxWriter workbook and worksheet objects
workbook = writer.book
worksheet = writer.sheets['Sheet1']

# Define formats
float_format = workbook.add_format({'num_format': '0.00'})
int_format = workbook.add_format({'num_format': '0'})

# Apply formats to columns
worksheet.set_column('B:B', None, float_format)  # Close
worksheet.set_column('C:C', None, int_format)    # Volume
worksheet.set_column('D:D', None, float_format)  # VPC

# Save the file
writer._save()
print(f"Data saved to {output_path}")
