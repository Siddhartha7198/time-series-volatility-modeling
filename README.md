# Time-Series Volatility Modeling for Multi-Asset Portfolio Risk

## Executive Summary

Estimating and forecasting volatility is fundamental in financial risk management. This project develops a robust, implementable pipeline for **1-day Value-at-Risk (VaR)** and **Expected Shortfall (ES)** estimation for a multi-asset portfolio using time-series volatility models. We employ univariate **GJR-GARCH** with Student-t innovations for conditional volatility, **EWMA correlation** estimation for co-movements, and **multivariate Student-t simulation** for tail-aware distribution generation. The methodology is rigorously backtested with Kupiec and Christoffersen tests, and a stress-testing overlay demonstrates sensitivity to extreme market conditions. The code is object-oriented, organized for reproducibility, and designed for both academic understanding and practical application.

---

## 1. Motivation and Practical Relevance

Financial institutions must estimate future risk to allocate capital, set limits, price derivatives, and meet regulatory requirements. Traditional static risk metrics fail during periods of regime change or high correlation. Time-varying volatility and correlation modeling, combined with simulation, provides richer risk forecasts that account for:
- **volatility clustering**,  
- **fat tails**, and  
- **dynamic correlation**,  

which are critical under stressed market conditions.

This project implements such a pipeline for a **realistic multi-asset portfolio**, producing risk measures that are directly useful for risk managers, portfolio managers, and quant researchers.

---

## 2. Model Overview

The risk estimation framework comprises:
1. **Univariate volatility models**:
   - GJR-GARCH(1,1) with Student-t innovations (asymmetric volatility response)
2. **Correlation estimation**:
   - EWMA (Exponentially Weighted Moving Average) correlation of standardized residuals
3. **Simulation engine**:
   - Multivariate Student-t simulation via scale mixture of normals
4. **Risk measures**:
   - **Value-at-Risk (VaR)** and **Expected Shortfall (ES)**
5. **Backtesting**:
   - Kupiec’s Unconditional Coverage
   - Christoffersen’s Independence & Conditional Coverage
6. **Stress testing overlay**:
   - Volatility multipliers and correlation shifts

---

## 3. Mathematical Framework

### 3.1 Returns

Asset log-returns:
$$
r_{i,t} = \log(P_{i,t}) - \log(P_{i,t-1})
$$

Portfolio return:
$$
r_{p,t} = \mathbf{w}^\top \mathbf{r}_t
$$

### 3.2 GJR-GARCH Conditional Variance

For each asset:
$$
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \gamma \mathbf{1}_{\{\epsilon_{t-1}<0\}}\epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

with Student-t innovations.

### 3.3 EWMA Correlation

Given standardized residuals \(z_t\):
$$
S_t = \frac{\sum_{k=0}^{L-1} \lambda^k z_{t-k} z_{t-k}^\top}{\sum_{k=0}^{L-1} \lambda^k},\quad
R_t = D_t^{-1}S_tD_t^{-1}
$$

### 3.4 Multivariate Student-t Simulation

Using a scale mixture:
$$
\mathbf{r} = \frac{\mathbf{y}}{\sqrt{u/\nu}},\quad
\mathbf{y}\sim \mathcal{N}(0,\Sigma),\quad
u\sim \chi^2_\nu
$$

### 3.5 Risk Measures

Value-at-Risk:
$$
\text{VaR}_{\alpha} = -Q_{1-\alpha}\left(r_p\right)
$$

Expected Shortfall:
$$
\text{ES}_{\alpha} = -\mathbb{E}[r_p\mid r_p < -\text{VaR}_{\alpha}]
$$

---

## 4. Data Requirements

- **Price data**: daily adjusted close prices for selected tickers  
- **Time horizon**: at least **3 years** of data recommended for stable estimation  
- **Multiple assets**: a minimum of two assets to evaluate correlation effects

Example tickers used: ["SPY", "TLT", "GLD", "LQD"]

Data is pulled using `yfinance`, cleaned, and aligned across assets to ensure consistency.

---

## 5. Implementation Architecture

Each module encapsulates a logical component:
- **data_loader**: price acquisition/cleaning  
- **returns**: return computation  
- **volatility_models**: GARCH family wrappers  
- **correlation_models**: EWMA correlation  
- **var_engine**: VaR/ES simulation engine  
- **backtest**: rolling backtesting pipeline  
- **var_backtests**: statistical tests  
- **diagnostics**: model diagnostics  
- **stress_testing**: stress scenarios

---

## 6. Empirical Analysis

1. Load and clean price data  
2. Compute daily log-returns  
3. Fit GJR-GARCH models on rolling windows  
4. Estimate conditional correlation using EWMA  
5. Forecast one-day ahead variances and correlations  
6. Simulate multivariate Student-t portfolio returns  
7. Compute VaR and ES forecasts  
8. Evaluate performance using backtesting tests

The process is fully automated in `analysis.ipynb`, with plots and tables for interpretation.

---

## 7. Assumptions

- Daily returns are stationary within estimation windows  
- GJR-GARCH captures conditional heteroskedasticity  
- Student-t captures tail heaviness but assumes symmetric tails  
- EWMA correlation is a parsimonious dynamic correlation estimate  
- Simulation assumes elliptical distribution (no skew)

Assumptions are discussed and evaluated in the notebook.

---

## 8. Computational Considerations

- Simulation is vectorized; typical Monte Carlo runs of **20,000 draws** are performant on modern hardware  
- For larger dimensions, Cholesky decomposition is used with jitter stabilization  
- Diagnostics on standardized residuals ensure model adequacy before risk aggregation

---

## 9. Results and Interpretation

The `analysis.ipynb` notebook produces:

- **Time series plots** of portfolio returns vs. VaR  
- **Breach indicators** over the backtest period  
- **Kupiec & Christoffersen** test statistics  
- **Expected Shortfall** estimates alongside VaR  
- **Stress test comparison**, demonstrating how risk metrics widen under stressed volatility and correlation scenarios

Key findings typically reveal:
- Fat tails and dynamic volatility significantly affect tail risk forecasts  
- Simple EWMA correlation underestimates tail co-movements relative to crisis scenarios  
- Stress testing produces materially higher VaR/ES, underscoring risk sensitivity

---

## 10. Conclusion

This project builds a comprehensive, extensible pipeline for portfolio risk estimation using time-series volatility and correlation modeling. The combination of **GJR-GARCH volatility**, **EWMA correlation**, and **multivariate Student-t simulation** provides a practical and statistically grounded method to estimate risk measures relevant for both internal risk management and regulatory frameworks.

The code structure supports reproducibility and further extension, including DCC models, skewed distributions, and intraday realized volatility.

---

## Reproducibility

To reproduce the results:

1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Open the notebook: notebooks/analysis.ipynb
Figures and outputs will be generated in figures/
