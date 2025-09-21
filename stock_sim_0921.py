import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Function to calculate CAGR for a given period in years
def calculate_cagr(ticker, years):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365 + years // 4)  # Approximate, accounting for leap years
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if data.empty:
            return np.nan
        start_price = data['Close'].iloc[2].to_list()[0]
        end_price = data['Close'].iloc[-1].to_list()[0]

        actual_years = (end_date - start_date).days / 365.25
        if start_price <= 0 or actual_years <= 0:
            return np.nan
        cagr = (end_price / start_price) ** (1 / actual_years) - 1
        return cagr
    except Exception as e:
        print(f"Error calculating CAGR for {ticker}: {e}")
        return np.nan

# Function to fetch data for a single ticker
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = {}
    
    # CAGR calculations
    data['CAGR_10Y'] = calculate_cagr(ticker, 10)
    data['CAGR_5Y'] = calculate_cagr(ticker, 5)
    data['CAGR_3Y'] = calculate_cagr(ticker, 3)
    data['CAGR_1Y'] = calculate_cagr(ticker, 1)
    
    # Beta
    try:
        data['Beta'] = stock.info.get('beta', np.nan)
    except Exception:
        data['Beta'] = np.nan
    
    # Margins from info (TTM)
    try:
        data['Gross_Margin'] = stock.info.get('grossMargins', np.nan)
        data['Operating_Margin'] = stock.info.get('operatingMargins', np.nan)
        data['Net_Margin'] = stock.info.get('profitMargins', np.nan)
    except Exception:
        data['Gross_Margin'] = np.nan
        data['Operating_Margin'] = np.nan
        data['Net_Margin'] = np.nan
    
    # Financials for ROCE and Debt/Equity
    try:
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        
        # Latest available data (assuming annual, TTM if available)
        if not income_stmt.empty and not balance_sheet.empty:
            # EBIT (use Operating Income as proxy if EBIT not directly available)
            ebit = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else np.nan
            
            # Capital Employed: Total Assets - Current Liabilities
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else np.nan
            current_liab = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else np.nan
            capital_employed = total_assets - current_liab if not np.isnan(total_assets) and not np.isnan(current_liab) else np.nan
            
            # ROCE
            data['ROCE'] = ebit / capital_employed if not np.isnan(ebit) and capital_employed != 0 else np.nan
            
            # Debt/Equity
            total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else np.nan
            total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else np.nan
            data['Debt_Equity'] = total_debt / total_equity if not np.isnan(total_debt) and total_equity != 0 else np.nan
        else:
            data['ROCE'] = np.nan
            data['Debt_Equity'] = np.nan
    except Exception:
        data['ROCE'] = np.nan
        data['Debt_Equity'] = np.nan
    
    return data

# Main script
def main():
    # Load tickers from Excel file
    file_path = 'all_tickers_2025.xlsx'  # Update path if needed
    df_tickers = pd.read_excel(file_path)
    tickers = df_tickers['Ticker'].tolist()  # Assume column name is 'Ticker'; adjust if different (e.g., 'Symbol')
    
    # Fetch data for all tickers
    results = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = fetch_stock_data(ticker)
        data['Ticker'] = ticker
        results.append(data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[['Ticker', 'CAGR_10Y', 'CAGR_5Y', 'CAGR_3Y', 'CAGR_1Y', 'ROCE', 'Gross_Margin', 'Operating_Margin', 'Net_Margin', 'Debt_Equity', 'Beta']]
    
    # Format percentages
    for col in ['CAGR_10Y', 'CAGR_5Y', 'CAGR_3Y', 'CAGR_1Y', 'ROCE', 'Gross_Margin', 'Operating_Margin', 'Net_Margin']:
        df[col] = df[col].apply(lambda x: f"{x * 100:.2f}%" if not np.isnan(x) else 'N/A')
    for col in ['Debt_Equity', 'Beta']:
        df[col] = df[col].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else 'N/A')
    
    # Save to CSV
    from datetime import datetime

    current_date = datetime.now().strftime("%Y%m%d")
    output_path = f'stock_analysis_comparison_{current_date}.xlsx'
    df.to_excel(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()