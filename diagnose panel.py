import numpy as np
import pandas as pd

tickers = ['AAPL', 'MSFT', 'JNJ', 'GOOGL', 'WMT', 'JPM', 'KO', 'NVDA', 'XOM']

print("--- Raw Files ---")
for s in ['SPX', 'VIX'] + tickers:
    df = pd.read_csv(f"{s}_20yr_data.csv", parse_dates=['datetime'])
    start = df['datetime'].min().strftime('%Y-%m-%d')
    end = df['datetime'].max().strftime('%Y-%m-%d')
    print(f"{s:<6} rows: {len(df):<5} | range: {start} -> {end}")

panel = pd.read_csv('panel.csv', parse_dates=['datetime'], index_col='datetime')
print(f"\n--- Aligned Panel ---\nShape: {panel.shape}")
print(f"Date range: {panel.index.min().date()} to {panel.index.max().date()}")
print(f"Trading days per year:\n{panel.groupby(panel.index.year).size().to_dict()}")

ret_cols = ['spx_ret'] + [f"{t}_ret" for t in tickers]
summary = pd.DataFrame({
    'Ann Mean (%)': (panel[ret_cols].mean() * 252 * 100).round(2),
    'Ann Vol (%)': (panel[ret_cols].std() * np.sqrt(252) * 100).round(2)
})
print(f"\n--- Return Stats ---\n{summary}")

# 4. Outlier moves
print("\n--- Top 3 Outlier Days (abs return) ---")
for col in ret_cols:
    top3 = panel[col].abs().nlargest(3).index
    formatted = [f"{idx.strftime('%Y-%m-%d')}: {panel.loc[idx, col]:+.2%}" for idx in top3]
    print(f"{col:<10} " + " | ".join(formatted))

print("\n--- SPX Cross-Correlations (Lead/Lag) ---")
spx = panel['spx_ret']

for t in tickers:
    col = f"{t}_ret"
    lag_neg1 = panel[col].corr(spx.shift(1))
    lag_0 = panel[col].corr(spx)
    lag_pos1 = panel[col].corr(spx.shift(-1))
    
    warn = " [CHECK TIMING]" if max(abs(lag_neg1), abs(lag_pos1)) > abs(lag_0) else ""
    print(f"{col:<10} t-1: {lag_neg1:+.3f} | t: {lag_0:+.3f} | t+1: {lag_pos1:+.3f}{warn}")