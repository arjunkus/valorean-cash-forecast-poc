# 💰 Cash Forecasting Intelligence System

## Enterprise-Grade Multi-Horizon Cash Flow Forecasting with AI/ML

A comprehensive cash flow forecasting solution that combines traditional statistical methods with deep learning to provide accurate predictions across multiple time horizons. Built for SAP FQM integration with multi-company, multi-currency support.

---

## 🎯 Key Features

### Forecasting Models
| Horizon | Model | Best For |
|---------|-------|----------|
| **RT+7** | ARIMA | Real-time + 7 days, captures recent momentum |
| **T+30** | Prophet | 30-day forecast, handles seasonality & holidays |
| **T+90** | LSTM | 90-day forecast, captures complex patterns |
| **NT+365** | Ensemble | Annual forecast, combines Prophet + LSTM |

### Analytics
- **Daily MAPE Analysis** - Granular accuracy by day-of-week (not weekly averages!)
- **Trend Decomposition** - STL decomposition with seasonality breakdown
- **SHAP Analysis** - Feature importance and model explainability
- **Outlier Detection** - Multi-method ensemble (Z-score, IQR, Isolation Forest)

### Business Intelligence
- Actionable recommendations with priority rankings
- Liquidity threshold monitoring
- Forecast accuracy alerts
- Trend warnings and investment opportunities

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.10+
pip install -r requirements.txt
```

### Launch Dashboard
```bash
cd cash_forecast
streamlit run dashboard_enhanced.py
```

Or use the launcher script:
```bash
python run_dashboard.py
```

Access at: **http://localhost:8501**

---

## 📁 Project Structure

```
cash_forecast/
├── config.py              # Configuration, thresholds, business rules
├── data_simulator.py      # SAP FQM data simulation
├── models.py              # ARIMA, Prophet, LSTM, Ensemble forecasters
├── analysis.py            # MAPE, Trend, SHAP, Outlier analysis
├── recommendations.py     # Actionable insights engine
├── dashboard_enhanced.py  # Interactive Streamlit dashboard
├── run_dashboard.py       # Dashboard launcher
├── requirements.txt       # Python dependencies
└── __init__.py           # Package exports
```

---

## 📊 Dashboard Features

### Tab 1: Overview
- Cash position with forecast overlay
- Inflow/outflow waterfall chart
- Category distribution (pie charts)
- Key metrics (cash position, days of cash, net flows)

### Tab 2: Forecasts
- Multi-horizon comparison view
- Individual horizon details with confidence intervals
- Forecast data tables with export

### Tab 3: MAPE Analysis
- **Daily MAPE** breakdown by day of week
- Gauge charts with industry thresholds
- Heatmap visualization
- Best/worst day identification

### Tab 4: Trends & SHAP
- Trend direction and slope analysis
- Change point detection
- SHAP feature importance bar chart
- Full decomposition (trend + seasonality + residuals)

### Tab 5: Outliers
- Timeline visualization with outliers highlighted
- Statistical bands (mean ± 2σ)
- Outlier table with scores
- Pattern insights

### Tab 6: Recommendations
- Priority-ranked action items
- Severity filtering (Critical/Warning/Info/Success)
- Category filtering
- Full context and impact analysis

### Tab 7: Export
- CSV downloads for all data
- Forecast exports
- Recommendation reports
- MAPE analysis exports

---

## ⚙️ Configuration

### MAPE Thresholds (Industry Standards)
```python
MAPE_THRESHOLDS = {
    "RT+7": {"excellent": 3%, "good": 5%, "acceptable": 8%, "poor": 15%},
    "T+30": {"excellent": 5%, "good": 10%, "acceptable": 15%, "poor": 25%},
    "T+90": {"excellent": 8%, "good": 15%, "acceptable": 20%, "poor": 30%},
    "NT+365": {"excellent": 12%, "good": 20%, "acceptable": 30%, "poor": 40%},
}
```

### Liquidity Thresholds
```python
minimum_cash_days = 30   # Alert threshold
target_cash_days = 60    # Target buffer
excess_cash_days = 90    # Investment opportunity
```

### Company Codes (Multi-Entity)
- 1000: US Operations (USD)
- 2000: EU Operations (EUR)
- 3000: UK Operations (GBP)
- 4000: APAC Operations (JPY)
- 5000: LATAM Operations (BRL)

All currencies automatically converted to USD.

---

## 🔧 SAP FQM Integration

The data simulator mimics SAP FQM_FLOW table structure:

```python
FQM_FLOW columns:
- transaction_id
- posting_date, value_date
- company_code, company_name
- bank_account_id, bank_name
- flow_type (INFLOW/OUTFLOW)
- category_id, category_name
- amount_local, currency, exchange_rate
- amount_usd, region
```

### Cash Flow Categories
**Inflows:** AR, Investment Income, Loan Proceeds, Intercompany In, Other
**Outflows:** AP, Payroll, Tax, Debt Service, CapEx, Intercompany Out, Other

---

## 📈 Model Details

### ARIMA (RT+7)
- Order: (5, 1, 2) with weekly seasonality
- Best for: Short-term momentum, immediate cash needs
- Typical MAPE: 3-8%

### Prophet (T+30)
- Weekly, monthly, quarterly, yearly seasonality
- Changepoint detection for trend shifts
- Best for: Medium-term with strong seasonality
- Typical MAPE: 5-15%

### LSTM (T+90)
- 2-layer architecture, 50 units each
- 30-day sequence length
- Best for: Complex non-linear patterns
- Typical MAPE: 8-20%

### Ensemble (NT+365)
- 60% Prophet + 40% LSTM weighted average
- Confidence intervals from both models
- Best for: Long-term strategic planning
- Typical MAPE: 12-30%

---

## 🛠️ API Usage

```python
from cash_forecast import (
    generate_sample_data,
    CashFlowForecaster,
    CashFlowAnalyzer,
    generate_recommendations
)

# Generate data
data = generate_sample_data(periods=730)
daily_cash = data['daily_cash_position']

# Train models
forecaster = CashFlowForecaster()
forecaster.fit(daily_cash)

# Generate forecasts
forecasts = forecaster.predict()  # All horizons
forecast_7d = forecaster.predict("RT+7")  # Specific horizon

# Run analysis
analyzer = CashFlowAnalyzer()
results = analyzer.full_analysis(daily_cash)

# Get recommendations
recommendations, summary = generate_recommendations(
    daily_cash, forecasts, results
)
```

---

## 📋 Requirements

```
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
streamlit>=1.29.0
statsmodels>=0.14.0
prophet>=1.1.4
tensorflow>=2.15.0
scikit-learn>=1.3.0
shap>=0.44.0
scipy>=1.11.0
joblib>=1.3.0
```

---

## 🔮 Roadmap

- [ ] Real SAP FQM connector
- [ ] Real-time data streaming
- [ ] Automated model retraining
- [ ] Email/Slack alerts
- [ ] Multi-tenant deployment
- [ ] API endpoints for integration

---

## 📄 License

Enterprise Treasury Management Solution - Internal Use

---

## 🙋 Support

For questions or customization requests, please contact the Treasury Analytics team.
