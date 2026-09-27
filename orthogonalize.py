# Step two of the analysis pipeline: strip the market move out of dVIX.

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

panel = pd.read_csv("panel.csv", parse_dates=['datetime'])
panel = panel.set_index('datetime')

y = panel['dvix']
X = sm.add_constant(panel['spx_ret'])
model = sm.OLS(y, X).fit()
panel['dvix_orth'] = model.resid

print("corr(dvix_orth, spx_ret):", panel['dvix_orth'].corr(panel['spx_ret']))
print("slope on spx_ret:", model.params['spx_ret'])
print("R-squared:", model.rsquared)

panel.to_csv('panel.csv')
print("Saved panel.csv with dvix_orth")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(panel['spx_ret'], panel['dvix'], s=8, alpha=0.4)
x_line = pd.Series(sorted(panel['spx_ret']))
axes[0].plot(
    x_line,
    model.params['const'] + model.params['spx_ret'] * x_line,
    color='red',
    lw=1.5
)
axes[0].set_xlabel('SPX return')
axes[0].set_ylabel('dVIX')
axes[0].set_title(f"Leverage effect (slope = {model.params['spx_ret']:.2f})")

axes[1].plot(panel.index, panel['dvix_orth'], lw=0.6)
axes[1].axhline(0, color='red', lw=0.8)
axes[1].set_title('dvix_orth over time')

plt.tight_layout()
plt.show()