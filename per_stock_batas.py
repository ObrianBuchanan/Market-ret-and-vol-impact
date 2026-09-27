import pandas as pd
import statsmodels.api as sm

panel = pd.read_csv('panel.csv', parse_dates=['datetime'], index_col='datetime')

if 'dvix_orth' not in panel.columns:
    mkt = sm.add_constant(panel['spx_ret'])
    panel['dvix_orth'] = sm.OLS(panel['dvix'], mkt).fit().resid

stocks = [c.replace('_ret', '') for c in panel.columns if c.endswith('_ret') and c != 'spx_ret']
factors = sm.add_constant(panel[['spx_ret', 'dvix_orth']])

rows = []
for s in stocks:
    df = pd.concat([panel[f'{s}_ret'], factors], axis=1).dropna()
    
    y = df[f'{s}_ret']
    X = df[['const', 'spx_ret', 'dvix_orth']]
    
    res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 5})
    
    rows.append({
        'stock': s,
        'alpha': res.params['const'],
        'mkt_beta': res.params['spx_ret'],
        'vol_beta': res.params['dvix_orth'],
        'vol_tstat': res.tvalues['dvix_orth'],
        'r2': res.rsquared
    })

results = pd.DataFrame(rows).sort_values('vol_beta').reset_index(drop=True)

print(results.round(4).to_string(index=False))

sig_neg = ((results['vol_beta'] < 0) & (results['vol_tstat'] < -2)).sum()
print(f"\nH1: {(results['vol_beta'] < 0).sum()}/{len(results)} stocks have negative vol_beta; "
      f"{sig_neg}/{len(results)} are significantly negative (t < -2)")

results.to_csv('h1_betas.csv', index=False)