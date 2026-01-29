"""
Cash Forecasting Intelligence System
=====================================
A comprehensive multi-horizon cash flow forecasting solution with 
ARIMA, Prophet, LSTM, and Ensemble models.

Modules:
- config: Configuration and business rules
- data_simulator: SAP FQM data simulation
- models: Forecasting models (ARIMA, Prophet, LSTM, Ensemble)
- analysis: MAPE, Trend, SHAP, and Outlier analysis
- recommendations: Actionable insights engine
- dashboard: Interactive Streamlit dashboard
"""

__version__ = "1.0.0"
__author__ = "Cash Forecasting Team"

from .config import (
    TIME_HORIZONS,
    MAPE_THRESHOLDS,
    COMPANY_CODES,
    CASH_FLOW_CATEGORIES,
    EXCHANGE_RATES,
)

from .data_simulator import (
    SAPFQMSimulator,
    generate_sample_data,
)

from .models import (
    ARIMAForecaster,
    ProphetForecaster,
    LSTMForecaster,
    EnsembleForecaster,
    CashFlowForecaster,
    run_backtest,
)

from .analysis import (
    MAPEAnalyzer,
    TrendAnalyzer,
    OutlierDetector,
    SHAPAnalyzer,
    CashFlowAnalyzer,
)

from .recommendations import (
    RecommendationEngine,
    generate_recommendations,
    Recommendation,
    Severity,
    Category,
)

__all__ = [
    # Config
    "TIME_HORIZONS",
    "MAPE_THRESHOLDS",
    "COMPANY_CODES",
    "CASH_FLOW_CATEGORIES",
    "EXCHANGE_RATES",
    # Data
    "SAPFQMSimulator",
    "generate_sample_data",
    # Models
    "ARIMAForecaster",
    "ProphetForecaster",
    "LSTMForecaster",
    "EnsembleForecaster",
    "CashFlowForecaster",
    "run_backtest",
    # Analysis
    "MAPEAnalyzer",
    "TrendAnalyzer",
    "OutlierDetector",
    "SHAPAnalyzer",
    "CashFlowAnalyzer",
    # Recommendations
    "RecommendationEngine",
    "generate_recommendations",
    "Recommendation",
    "Severity",
    "Category",
]
