"""
Cash Forecasting Configuration
==============================
Updated with T+1 horizon for daily forecasting.
"""

from dataclasses import dataclass
from typing import Dict, List

# =============================================================================
# TIME HORIZONS (Now includes T+1)
# =============================================================================
TIME_HORIZONS = {
    "T+1": {"days": 1, "model": "Prophet", "description": "Next Day Forecast"},
    "T+7": {"days": 7, "model": "Prophet", "description": "7-Day Forecast"},
    "T+30": {"days": 30, "model": "Prophet", "description": "30-Day Forecast"},
    "T+90": {"days": 90, "model": "Prophet", "description": "90-Day Forecast"},
}

# =============================================================================
# MAPE THRESHOLDS (Industry Standards)
# =============================================================================
MAPE_THRESHOLDS = {
    "T+1": {"excellent": 3.0, "good": 5.0, "acceptable": 10.0, "poor": 20.0},
    "T+7": {"excellent": 5.0, "good": 10.0, "acceptable": 15.0, "poor": 25.0},
    "T+30": {"excellent": 8.0, "good": 15.0, "acceptable": 20.0, "poor": 30.0},
    "T+90": {"excellent": 12.0, "good": 20.0, "acceptable": 25.0, "poor": 35.0},
}

# =============================================================================
# LIQUIDITY THRESHOLDS
# =============================================================================
@dataclass
class LiquidityThresholds:
    """Cash buffer thresholds based on industry standards."""
    minimum_cash_days: int = 30
    target_cash_days: int = 60
    excess_cash_days: int = 90
    critical_threshold_pct: float = 0.10
    warning_threshold_pct: float = 0.25

LIQUIDITY = LiquidityThresholds()

# =============================================================================
# COMPANY CODES (Multi-Entity)
# =============================================================================
COMPANY_CODES = [
    {"code": "1000", "name": "US Operations", "currency": "USD", "region": "NA"},
    {"code": "2000", "name": "EU Operations", "currency": "EUR", "region": "EU"},
    {"code": "3000", "name": "UK Operations", "currency": "GBP", "region": "EU"},
    {"code": "4000", "name": "APAC Operations", "currency": "JPY", "region": "APAC"},
    {"code": "5000", "name": "LATAM Operations", "currency": "BRL", "region": "LATAM"},
]

BANK_ACCOUNTS = [
    {"account_id": "BA001", "company_code": "1000", "bank_name": "JPMorgan Chase", "account_type": "Operating"},
    {"account_id": "BA002", "company_code": "1000", "bank_name": "Bank of America", "account_type": "Payroll"},
    {"account_id": "BA003", "company_code": "2000", "bank_name": "Deutsche Bank", "account_type": "Operating"},
    {"account_id": "BA004", "company_code": "3000", "bank_name": "Barclays", "account_type": "Operating"},
    {"account_id": "BA005", "company_code": "4000", "bank_name": "Mitsubishi UFJ", "account_type": "Operating"},
    {"account_id": "BA006", "company_code": "5000", "bank_name": "Itau Unibanco", "account_type": "Operating"},
]

# =============================================================================
# EXCHANGE RATES (to USD)
# =============================================================================
EXCHANGE_RATES = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "JPY": 0.0067,
    "BRL": 0.20,
}

# =============================================================================
# PROPHET MODEL PARAMETERS
# =============================================================================
@dataclass
class ProphetParams:
    """Prophet model parameters optimized for cash forecasting."""
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = 'additive'
    interval_width: float = 0.95

PROPHET_PARAMS = ProphetParams()

# =============================================================================
# CASH FLOW CATEGORIES
# =============================================================================
CASH_FLOW_CATEGORIES = {
    "inflows": [
        {"category": "AR", "name": "Accounts Receivable", "schedule": "Daily (Banking Days)"},
        {"category": "INV_INC", "name": "Investment Income", "schedule": "Monthly (1st)"},
        {"category": "IC_IN", "name": "Intercompany In", "schedule": "Weekly (Monday)"},
    ],
    "outflows": [
        {"category": "PAYROLL", "name": "Payroll", "schedule": "Bi-weekly (15th & Month End)"},
        {"category": "AP", "name": "Accounts Payable", "schedule": "Weekly (Friday)"},
        {"category": "TAX", "name": "Tax Payments", "schedule": "Quarterly (15th)"},
        {"category": "CAPEX", "name": "Capital Expenditures", "schedule": "Every 4 Months (20th)"},
        {"category": "DEBT", "name": "Debt Service", "schedule": "Monthly (1st)"},
        {"category": "IC_OUT", "name": "Intercompany Out", "schedule": "Weekly (Wednesday)"},
    ],
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================
DASHBOARD_CONFIG = {
    "title": "Cash Forecasting Intelligence Dashboard",
    "refresh_interval": 300,
    "chart_height": 400,
    "color_scheme": {
        "actual": "#1f77b4",
        "forecast": "#ff7f0e",
        "inflow": "#2ca02c",
        "outflow": "#d62728",
        "confidence": "rgba(255, 127, 14, 0.2)",
    },
}
