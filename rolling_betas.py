import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

WINDOW = 252

tickers = ['AAPL', 'MSFT', 'JNJ', 'GOOGL', 'WMT', 'JPM', 'KO', 'NVDA', 'XOM']

panel = pd.read_csv('panel.csv', parse_dates=['datetime'], index_col='datetime')
X = sm.add_constant(panel[['spx_ret', 'dvix_orth']])

betas = pd.DataFrame(index=panel.index)
ses = pd.DataFrame(index=panel.index)
full_betas = {}

for t in tickers:
    y = panel[f'{t}_ret']
    res = RollingOLS(y, X, window=WINDOW).fit(cov_type='HCCM')
    betas[t] = res.params['dvix_orth']
    ses[t] = res.bse['dvix_orth']
    full_betas[t] = sm.OLS(y, X).fit().params['dvix_orth']

betas.to_csv('rolling_vol_betas.csv')

# Sort plots by full-sample vol beta (most negative first)
tickers_sorted = sorted(full_betas, key=full_betas.get)

crises = [
    ('2008-09-01', '2009-06-30'),
    ('2020-02-15', '2020-06-30'),
    ('2022-01-01', '2022-10-31')
]

fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)

for ax, t in zip(axes.flat, tickers_sorted):
    b = betas[t].dropna()
    se = ses[t].loc[b.index]

    for start, end in crises:
        ax.axvspan(start, end, color='#eeeeee', zorder=0)

    ax.fill_between(b.index, b - 2 * se, b + 2 * se, color='#2a6fdb', alpha=0.15)
    ax.plot(b, color='#2a6fdb', lw=1.3, label='Rolling 1Y β')
    ax.axhline(full_betas[t], color='#555555', ls='--', lw=1, label='Full-sample β')
    ax.axhline(0, color='black', lw=0.6)

    ax.set_title(f"{t} (full: {full_betas[t]:+.4f})", fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

for ax in axes[:, 0]:
    ax.set_ylabel('Vol Beta (return per pt ΔVIX)')

axes[0, 0].legend(loc='lower left', frameon=False, fontsize=8)

fig.suptitle('Rolling 1-Year Sensitivity to Pure Volatility Shocks', fontsize=12)
plt.tight_layout()
plt.savefig('rolling_vol_betas.png', dpi=150)
plt.show()