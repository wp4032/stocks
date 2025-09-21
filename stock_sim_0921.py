import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

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

# Function to calculate average ROCE over a given number of years
def calculate_avg_roce(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        if income_stmt.empty or balance_sheet.empty:
            return np.nan
        
        roce_values = []
        available_years = min(years, len(income_stmt.columns))
        for year in range(available_years):
            ebit = income_stmt.iloc[:, year].get('Operating Income', np.nan)
            total_assets = balance_sheet.iloc[:, year].get('Total Assets', np.nan)
            current_liab = balance_sheet.iloc[:, year].get('Current Liabilities', np.nan)
            capital_employed = total_assets - current_liab if not np.isnan(total_assets) and not np.isnan(current_liab) else np.nan
            roce = ebit / capital_employed if not np.isnan(ebit) and capital_employed != 0 else np.nan
            roce_values.append(roce)
        
        valid_roce = [x for x in roce_values if not np.isnan(x)]
        return np.mean(valid_roce) if valid_roce else np.nan
    except Exception as e:
        print(f"Error calculating ROCE for {ticker}: {e}")
        return np.nan

# Function to calculate average COGS Margin over a given number of years
def calculate_avg_cogs_margin(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        if income_stmt.empty:
            return np.nan
        
        cogs_margin_values = []
        available_years = min(years, len(income_stmt.columns))
        for year in range(available_years):
            revenue = income_stmt.iloc[:, year].get('Total Revenue', np.nan)
            cogs = income_stmt.iloc[:, year].get('Cost Of Revenue', np.nan)
            cogs_margin = cogs / revenue if not np.isnan(cogs) and not np.isnan(revenue) and revenue != 0 else np.nan
            cogs_margin_values.append(cogs_margin)
        
        valid_cogs_margin = [x for x in cogs_margin_values if not np.isnan(x)]
        return np.mean(valid_cogs_margin) if valid_cogs_margin else np.nan
    except Exception as e:
        print(f"Error calculating COGS Margin for {ticker}: {e}")
        return np.nan

# Function to calculate average Debt/Equity over a given number of years
def calculate_avg_debt_equity(ticker, years):
    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        if balance_sheet.empty:
            return np.nan
        
        de_values = []
        available_years = min(years, len(balance_sheet.columns))
        for year in range(available_years):
            total_debt = balance_sheet.iloc[:, year].get('Total Debt', np.nan)
            total_equity = balance_sheet.iloc[:, year].get('Stockholders Equity', np.nan)
            de = total_debt / total_equity if not np.isnan(total_debt) and not np.isnan(total_equity) and total_equity != 0 else np.nan
            de_values.append(de)
        
        valid_de = [x for x in de_values if not np.isnan(x)]
        return np.mean(valid_de) if valid_de else np.nan
    except Exception as e:
        print(f"Error calculating Debt/Equity for {ticker}: {e}")
        return np.nan

# Function to calculate Beta over a given number of years
def calculate_beta(ticker, years):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + years // 4)
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
        market_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=True)
        if stock_data.empty or market_data.empty:
            return np.nan
        
        # Calculate daily returns
        stock_returns = stock_data['Close'].pct_change().dropna()
        market_returns = market_data['Close'].pct_change().dropna()
        
        # Align dates
        aligned_data = pd.concat([stock_returns, market_returns], axis=1, join='inner')
        aligned_data.columns = ['stock', 'market']
        
        # Perform linear regression to get Beta (slope)
        if len(aligned_data) < 2:
            return np.nan
        slope, _, _, _, _ = stats.linregress(aligned_data['market'], aligned_data['stock'])
        return slope
    except Exception as e:
        print(f"Error calculating Beta for {ticker}: {e}")
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
    data['CAGR_Compound'] = (
        data['CAGR_5Y'] * 0.2 + 
        data['CAGR_3Y'] * 0.3 + 
        data['CAGR_1Y'] * 0.5
    ) if not any(np.isnan([data['CAGR_5Y'], data['CAGR_3Y'], data['CAGR_1Y']])) else np.nan
    
    # ROCE calculations (TTM and averages)
    try:
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        if not income_stmt.empty and not balance_sheet.empty:
            ebit = income_stmt.loc['Operating Income'].iloc[0] if 'Operating Income' in income_stmt.index else np.nan
            total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else np.nan
            current_liab = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else np.nan
            capital_employed = total_assets - current_liab if not np.isnan(total_assets) and not np.isnan(current_liab) else np.nan
            data['ROCE'] = ebit / capital_employed if not np.isnan(ebit) and capital_employed != 0 else np.nan
        else:
            data['ROCE'] = np.nan
    except Exception:
        data['ROCE'] = np.nan
    
    # Average ROCE
    data['ROCE_5Y'] = calculate_avg_roce(ticker, 5)
    data['ROCE_3Y'] = calculate_avg_roce(ticker, 3)
    data['ROCE_1Y'] = calculate_avg_roce(ticker, 1)
    data['ROCE_Compound'] = (
        data['ROCE_5Y'] * 0.2 + 
        data['ROCE_3Y'] * 0.3 + 
        data['ROCE_1Y'] * 0.5
    ) if not any(np.isnan([data['ROCE_5Y'], data['ROCE_3Y'], data['ROCE_1Y']])) else np.nan
    
    # Beta (TTM and averages)
    try:
        data['Beta'] = stock.info.get('beta', np.nan)
    except Exception:
        data['Beta'] = np.nan
    data['Beta_5Y'] = calculate_beta(ticker, 5)
    data['Beta_3Y'] = calculate_beta(ticker, 3)
    data['Beta_1Y'] = calculate_beta(ticker, 1)
    data['Beta_Compound'] = (
        data['Beta_5Y'] * 0.2 + 
        data['Beta_3Y'] * 0.3 + 
        data['Beta_1Y'] * 0.5
    ) if not any(np.isnan([data['Beta_5Y'], data['Beta_3Y'], data['Beta_1Y']])) else np.nan
    
    # Margins from info (TTM)
    try:
        data['Gross_Margin'] = stock.info.get('grossMargins', np.nan)
        data['Operating_Margin'] = stock.info.get('operatingMargins', np.nan)
        data['Net_Margin'] = stock.info.get('profitMargins', np.nan)
    except Exception:
        data['Gross_Margin'] = np.nan
        data['Operating_Margin'] = np.nan
        data['Net_Margin'] = np.nan
    
    # COGS Margin (averages)
    data['COGS_Margin_5Y'] = calculate_avg_cogs_margin(ticker, 5)
    data['COGS_Margin_3Y'] = calculate_avg_cogs_margin(ticker, 3)
    data['COGS_Margin_1Y'] = calculate_avg_cogs_margin(ticker, 1)
    data['COGS_Margin_Compound'] = (
        data['COGS_Margin_5Y'] * 0.2 + 
        data['COGS_Margin_3Y'] * 0.3 + 
        data['COGS_Margin_1Y'] * 0.5
    ) if not any(np.isnan([data['COGS_Margin_5Y'], data['COGS_Margin_3Y'], data['COGS_Margin_1Y']])) else np.nan
    
    # Debt/Equity (TTM and averages)
    try:
        total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else np.nan
        total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else np.nan
        data['Debt_Equity'] = total_debt / total_equity if not np.isnan(total_debt) and total_equity != 0 else np.nan
    except Exception:
        data['Debt_Equity'] = np.nan
    data['Debt_Equity_5Y'] = calculate_avg_debt_equity(ticker, 5)
    data['Debt_Equity_3Y'] = calculate_avg_debt_equity(ticker, 3)
    data['Debt_Equity_1Y'] = calculate_avg_debt_equity(ticker, 1)
    data['Debt_Equity_Compound'] = (
        data['Debt_Equity_5Y'] * 0.2 + 
        data['Debt_Equity_3Y'] * 0.3 + 
        data['Debt_Equity_1Y'] * 0.5
    ) if not any(np.isnan([data['Debt_Equity_5Y'], data['Debt_Equity_3Y'], data['Debt_Equity_1Y']])) else np.nan
    
    return data

# Main script
def main():
    # Load tickers from Excel file
    file_path = 'all_tickers_2025.xlsx'
    df_tickers = pd.read_excel(file_path)
    tickers = df_tickers['Ticker'].tolist()
    
    # Fetch data for all tickers
    results = []
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = fetch_stock_data(ticker)
        data['Ticker'] = ticker
        results.append(data)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    df = df[[
        'Ticker', 'CAGR_10Y', 'CAGR_5Y', 'CAGR_3Y', 'CAGR_1Y', 'CAGR_Compound',
        'ROCE', 'ROCE_5Y', 'ROCE_3Y', 'ROCE_1Y', 'ROCE_Compound',
        'Gross_Margin', 'Operating_Margin', 'Net_Margin',
        'COGS_Margin_5Y', 'COGS_Margin_3Y', 'COGS_Margin_1Y', 'COGS_Margin_Compound',
        'Debt_Equity', 'Debt_Equity_5Y', 'Debt_Equity_3Y', 'Debt_Equity_1Y', 'Debt_Equity_Compound',
        'Beta', 'Beta_5Y', 'Beta_3Y', 'Beta_1Y', 'Beta_Compound'
    ]]
    
    # Replace np.nan with None for Excel compatibility
    df = df.fillna(np.nan).replace(np.nan, None)
    df.sort_values(by='CAGR_Compound', ascending=False, inplace=True)
    
    # Save to Excel
    current_date = datetime.now().strftime("%Y%m%d")
    output_path = f'stock_analysis_comparison_{current_date}.xlsx'
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    # Access the XlsxWriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    
    # Define formats
    percentage_format = workbook.add_format({'num_format': '0.00%'})
    float_format = workbook.add_format({'num_format': '0.00'})
    
    # Apply formats to columns
    percentage_columns = [
        'CAGR_10Y', 'CAGR_5Y', 'CAGR_3Y', 'CAGR_1Y', 'CAGR_Compound',
        'ROCE', 'ROCE_5Y', 'ROCE_3Y', 'ROCE_1Y', 'ROCE_Compound',
        'Gross_Margin', 'Operating_Margin', 'Net_Margin',
        'COGS_Margin_5Y', 'COGS_Margin_3Y', 'COGS_Margin_1Y', 'COGS_Margin_Compound'
    ]
    float_columns = [
        'Debt_Equity', 'Debt_Equity_5Y', 'Debt_Equity_3Y', 'Debt_Equity_1Y', 'Debt_Equity_Compound',
        'Beta', 'Beta_5Y', 'Beta_3Y', 'Beta_1Y', 'Beta_Compound'
    ]
    
    for col in percentage_columns:
        col_idx = df.columns.get_loc(col)
        worksheet.set_column(col_idx, col_idx, None, percentage_format)
    
    for col in float_columns:
        col_idx = df.columns.get_loc(col)

    
    # Save the file
    writer._save()
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()