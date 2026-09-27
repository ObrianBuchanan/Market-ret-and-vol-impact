import pandas as pd

tickers = ['AAPL', 'MSFT', 'JNJ', 'GOOGL', 'WMT', 'JPM', 'KO', 'NVDA', 'XOM']

# base market & vol series
spx = pd.read_csv('SPX_20yr_data.csv', parse_dates=['datetime'], index_col='datetime')
vix = pd.read_csv('VIX_20yr_data.csv', parse_dates=['datetime'], index_col='datetime')

cols = {
    'spx_ret': spx['return'],
    'dvix': vix['close'].diff()
}

# pull return series for each basket name
for t in tickers:
    df = pd.read_csv(f'{t}_20yr_data.csv', parse_dates=['datetime'], index_col='datetime')
    cols[f'{t}_ret'] = df['return']

# align to common dates
panel = pd.DataFrame(cols).dropna()

print(panel.info())
print(f"\nDate range: {panel.index.min().date()} to {panel.index.max().date()}")
print(panel.head())

panel.to_csv('panel.csv')