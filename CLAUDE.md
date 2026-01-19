# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Valorean Cash Forecast POC is a Python/Streamlit treasury cash forecasting system for multi-horizon cash position management. It uses Prophet (primary), ARIMA, and LSTM models to forecast cash positions at T+1, T+7, T+30, T+90, and T+365 horizons.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard (opens on localhost:8501)
streamlit run dashboard_v6.py
```

## Architecture

```
dashboard_v6.py          → Main Streamlit UI (entry point)
    ↓
models_prophet_v6.py     → Forecasting engine (ProphetCashForecaster, ForecastAnalyzer)
daily_position.py        → T+1 position tracking (DailyPositionManager)
analysis_v3.py           → Anomaly detection (OutlierDetector) & explainability (SHAPAnalyzer)
    ↓
config.py                → All configuration (horizons, thresholds, business rules)
data_simulator_v3.py     → Synthetic data generation for testing
```

### Key Classes

- **ProphetCashForecaster** (`models_prophet_v6.py`): Main forecasting engine with cyclic event tracking
- **DailyPositionManager** (`daily_position.py`): Manages opening/closing positions, intraday transactions, SAP scheduled payments
- **CyclicalEventTracker** (`models_prophet_v6.py`): Detects recurring cash events (payroll, debt, taxes)
- **OutlierDetector** (`analysis_v3.py`): Z-score based anomaly detection by category
- **SHAPAnalyzer** (`analysis_v3.py`): Feature importance using SHAP values

### Configuration

All system parameters are in `config.py`:
- `TIME_HORIZONS`: Forecast periods and their associated models
- `MAPE_THRESHOLDS`: Accuracy evaluation criteria per horizon
- `BUSINESS_RULES`: Operating balance targets, sweep/funding thresholds, payment priorities
- `CASH_FLOW_CATEGORIES`: Inflow/outflow category definitions with schedules
- Model parameter dataclasses: `ProphetParams`, `ARIMAParams`, `LSTMParams`

## Data Flow

1. `data_simulator_v3.py` generates 2-year synthetic cash flow data with realistic patterns (biweekly payroll, day-of-week seasonality, month-end effects)
2. `models_prophet_v6.py` trains Prophet models with cyclic event detection
3. `daily_position.py` calculates daily positions from forecasts and transactions
4. `analysis_v3.py` runs outlier detection and SHAP analysis
5. `dashboard_v6.py` renders all visualizations in a tabbed interface
