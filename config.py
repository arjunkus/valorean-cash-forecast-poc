from dataclasses import dataclass

TIME_HORIZONS = {
    "T+1": {"days": 1, "model": "Prophet", "description": "Next Day Forecast"},
    "T+7": {"days": 7, "model": "ARIMA", "description": "Weekly Forecast"},
    "T+30": {"days": 30, "model": "Prophet", "description": "Monthly Forecast"},
    "T+90": {"days": 90, "model": "Prophet", "description": "Quarterly Forecast"},
    "T+365": {"days": 365, "model": "Ensemble", "description": "Annual Forecast"},
    "RT+7": {"days": 7, "model": "ARIMA", "description": "7-Day Forecast"},
    # "NT+365": {"days": 365, "model": "Ensemble", "description": "1-Year Forecast"}  # Disabled,
}

MAPE_THRESHOLDS = {
    "T+1": {"excellent": 3.0, "good": 5.0, "acceptable": 10.0, "poor": 20.0},
    "T+7": {"excellent": 5.0, "good": 10.0, "acceptable": 15.0, "poor": 25.0},
    "T+30": {"excellent": 8.0, "good": 15.0, "acceptable": 20.0, "poor": 30.0},
    "T+90": {"excellent": 12.0, "good": 20.0, "acceptable": 25.0, "poor": 35.0},
    "T+365": {"excellent": 15.0, "good": 25.0, "acceptable": 35.0, "poor": 50.0},
    "RT+7": {"excellent": 5.0, "good": 10.0, "acceptable": 15.0, "poor": 25.0},
    "NT+365": {"excellent": 15.0, "good": 25.0, "acceptable": 35.0, "poor": 50.0},
}

OUTLIER_THRESHOLDS = {"z_score": 3.0, "iqr_multiplier": 1.5, "isolation_forest_contamination": 0.1, "min_samples": 30, "window_size": 30}
BUSINESS_RULES = {"cash_management": {"min_operating_balance": 5000000, "target_operating_balance": 15000000, "max_operating_balance": 25000000, "sweep_threshold": 20000000, "funding_threshold": 8000000}, "alerts": {"critical_threshold": 0.10, "warning_threshold": 0.25, "forecast_accuracy_threshold": 15.0, "large_transaction_threshold": 1000000}, "payment_priorities": [{"priority": 1, "category": "PAYROLL", "description": "Employee payroll"}, {"priority": 2, "category": "TAX", "description": "Tax payments"}, {"priority": 3, "category": "DEBT", "description": "Debt service"}, {"priority": 4, "category": "AP", "description": "Accounts payable"}, {"priority": 5, "category": "CAPEX", "description": "Capital expenditures"}], "concentration_limits": {"single_bank_limit": 0.40, "single_currency_limit": 0.60, "single_entity_limit": 0.50}, "investment_guidelines": {"min_investment_amount": 1000000, "max_maturity_days": 90, "allowed_instruments": ["Money Market", "Commercial Paper", "Treasury Bills"]}}

@dataclass
class LiquidityThresholds:
    minimum_cash_days: int = 30
    target_cash_days: int = 60
    excess_cash_days: int = 90
    critical_threshold_pct: float = 0.10
    warning_threshold_pct: float = 0.25

LIQUIDITY = LiquidityThresholds()

COMPANY_CODES = [{"code": "1000", "name": "US Operations", "currency": "USD", "region": "NA"}, {"code": "2000", "name": "EU Operations", "currency": "EUR", "region": "EU"}, {"code": "3000", "name": "UK Operations", "currency": "GBP", "region": "EU"}, {"code": "4000", "name": "APAC Operations", "currency": "JPY", "region": "APAC"}, {"code": "5000", "name": "LATAM Operations", "currency": "BRL", "region": "LATAM"}]
BANK_ACCOUNTS = [{"account_id": "BA001", "company_code": "1000", "bank_name": "JPMorgan Chase", "account_type": "Operating"}, {"account_id": "BA002", "company_code": "1000", "bank_name": "Bank of America", "account_type": "Payroll"}, {"account_id": "BA003", "company_code": "2000", "bank_name": "Deutsche Bank", "account_type": "Operating"}, {"account_id": "BA004", "company_code": "3000", "bank_name": "Barclays", "account_type": "Operating"}, {"account_id": "BA005", "company_code": "4000", "bank_name": "Mitsubishi UFJ", "account_type": "Operating"}, {"account_id": "BA006", "company_code": "5000", "bank_name": "Itau Unibanco", "account_type": "Operating"}]
EXCHANGE_RATES = {"USD": 1.0, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "BRL": 0.20}

@dataclass
class ProphetParams:
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = 'additive'
    interval_width: float = 0.95

PROPHET_PARAMS = ProphetParams()

@dataclass
class ARIMAParams:
    order: tuple = (2, 1, 2)
    seasonal_order: tuple = (1, 1, 1, 7)
    trend: str = 'ct'
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True

ARIMA_PARAMS = ARIMAParams()

@dataclass
class LSTMParams:
    units: int = 64
    n_units: int = 64
    dropout: float = 0.2
    recurrent_dropout: float = 0.2
    epochs: int = 50
    batch_size: int = 32
    lookback: int = 14
    sequence_length: int = 30
    learning_rate: float = 0.001
    validation_split: float = 0.2

LSTM_PARAMS = LSTMParams()

CASH_FLOW_CATEGORIES = {"inflows": [{"category_id": "IN001", "category": "AR", "name": "Accounts Receivable", "category_name": "Accounts Receivable", "schedule": "Daily (Banking Days)", "typical_pct": 0.60, "volatility": 0.15}, {"category_id": "IN002", "category": "INV_INC", "name": "Investment Income", "category_name": "Investment Income", "schedule": "Monthly (1st)", "typical_pct": 0.10, "volatility": 0.05}, {"category_id": "IN003", "category": "IC_IN", "name": "Intercompany In", "category_name": "Intercompany In", "schedule": "Weekly (Monday)", "typical_pct": 0.30, "volatility": 0.10}], "outflows": [{"category_id": "OUT001", "category": "PAYROLL", "name": "Payroll", "category_name": "Payroll", "schedule": "Bi-weekly (15th & Month End)", "typical_pct": 0.35, "volatility": 0.02}, {"category_id": "OUT002", "category": "AP", "name": "Accounts Payable", "category_name": "Accounts Payable", "schedule": "Weekly (Friday)", "typical_pct": 0.40, "volatility": 0.20}, {"category_id": "OUT003", "category": "TAX", "name": "Tax Payments", "category_name": "Tax Payments", "schedule": "Quarterly (15th)", "typical_pct": 0.10, "volatility": 0.05}, {"category_id": "OUT004", "category": "CAPEX", "name": "Capital Expenditures", "category_name": "Capital Expenditures", "schedule": "Every 4 Months (20th)", "typical_pct": 0.05, "volatility": 0.30}, {"category_id": "OUT005", "category": "DEBT", "name": "Debt Service", "category_name": "Debt Service", "schedule": "Monthly (1st)", "typical_pct": 0.05, "volatility": 0.01}, {"category_id": "OUT006", "category": "IC_OUT", "name": "Intercompany Out", "category_name": "Intercompany Out", "schedule": "Weekly (Wednesday)", "typical_pct": 0.05, "volatility": 0.10}]}

DASHBOARD_CONFIG = {"title": "Valorean Cash Forecasting POC", "refresh_interval": 300, "chart_height": 400, "color_scheme": {"actual": "#1f77b4", "forecast": "# ff7f0e", "inflow": "#2ca02c", "outflow": "#d62728", "confidence": "rgba(255, 127, 14, 0.2)"}}
