"""
Cash Forecast Analysis v2
=========================
SHAP Analysis + Outlier Detection for Prophet-based forecasting.

Components:
1. SHAP Analysis - Explains what's driving the forecast
2. Outlier Detection - Identifies anomalous cash flows
3. Trend Decomposition - Breaks down seasonality components
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from scipy.stats import zscore
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# SHAP-LIKE ANALYSIS FOR PROPHET
# =============================================================================
class ProphetExplainer:
    """
    SHAP-like explainability for Prophet forecasts.
    
    Prophet doesn't directly support SHAP, but we can decompose predictions
    into interpretable components:
    - Trend: Long-term direction
    - Weekly Seasonality: Day-of-week effects
    - Monthly Seasonality: Day-of-month effects
    - Yearly Seasonality: Time-of-year effects
    - Holidays: Holiday effects (if configured)
    
    This gives treasury managers clear insight into WHY the forecast
    is what it is for any given day.
    """
    
    def __init__(self, model, model_name: str = ""):
        """
        Initialize explainer with a fitted Prophet model.
        
        Args:
            model: Fitted Prophet model
            model_name: Name for reporting (e.g., "Inflow", "Outflow")
        """
        self.model = model
        self.model_name = model_name
        self.components = None
    
    def explain(self, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """
        Decompose forecast into explainable components.
        
        Returns DataFrame with columns for each component's contribution.
        """
        if self.model is None:
            raise ValueError("No model provided")
        
        # Get Prophet's component breakdown
        # Prophet stores these in the forecast DataFrame
        components = ['trend']
        
        # Check which seasonalities exist
        if 'weekly' in forecast_df.columns:
            components.append('weekly')
        if 'monthly' in forecast_df.columns:
            components.append('monthly')
        if 'yearly' in forecast_df.columns:
            components.append('yearly')
        if 'biweekly' in forecast_df.columns:
            components.append('biweekly')
        if 'quarterly' in forecast_df.columns:
            components.append('quarterly')
        if 'month_end' in forecast_df.columns:
            components.append('month_end')
        if 'holidays' in forecast_df.columns:
            components.append('holidays')
        
        # Build explanation DataFrame
        explanation = pd.DataFrame({
            'date': forecast_df['ds'],
            'prediction': forecast_df['yhat'],
        })
        
        for comp in components:
            if comp in forecast_df.columns:
                explanation[f'{comp}_contribution'] = forecast_df[comp]
        
        # Calculate percentage contribution of each component
        total_variation = explanation['prediction'].std()
        if total_variation > 0:
            for comp in components:
                col = f'{comp}_contribution'
                if col in explanation.columns:
                    comp_std = explanation[col].std()
                    explanation[f'{comp}_importance'] = (comp_std / total_variation) * 100
        
        self.components = explanation
        return explanation
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance summary.
        
        Returns DataFrame with each component's relative importance.
        """
        if self.components is None:
            raise ValueError("Call explain() first")
        
        importance_cols = [c for c in self.components.columns if c.endswith('_importance')]
        
        if not importance_cols:
            return pd.DataFrame({'component': ['trend'], 'importance': [100.0]})
        
        importances = []
        for col in importance_cols:
            comp_name = col.replace('_importance', '')
            imp_value = self.components[col].iloc[0]  # Same for all rows
            importances.append({
                'component': comp_name,
                'importance': imp_value
            })
        
        df = pd.DataFrame(importances).sort_values('importance', ascending=False)
        
        # Normalize to 100%
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance'] / total) * 100
        else:
            df['importance_pct'] = 0
        
        return df
    
    def explain_single_day(self, forecast_df: pd.DataFrame, date: datetime) -> Dict:
        """
        Explain forecast for a single day.
        
        Returns dict with each component's contribution to that day's forecast.
        """
        if self.components is None:
            self.explain(forecast_df)
        
        date = pd.Timestamp(date)
        day_data = self.components[self.components['date'] == date]
        
        if len(day_data) == 0:
            return {"error": f"Date {date} not found in forecast"}
        
        day_data = day_data.iloc[0]
        
        result = {
            'date': date,
            'prediction': day_data['prediction'],
            'components': {}
        }
        
        contribution_cols = [c for c in self.components.columns if c.endswith('_contribution')]
        for col in contribution_cols:
            comp_name = col.replace('_contribution', '')
            result['components'][comp_name] = day_data[col]
        
        return result


class SHAPAnalyzer:
    """
    Comprehensive SHAP-like analysis for cash forecasting.
    Analyzes both inflow and outflow models.
    """
    
    def __init__(self, forecaster):
        """
        Initialize with a fitted ProphetCashForecaster.
        
        Args:
            forecaster: Fitted ProphetCashForecaster instance
        """
        self.forecaster = forecaster
        self.inflow_explainer = None
        self.outflow_explainer = None
        self.inflow_importance = None
        self.outflow_importance = None
    
    def analyze(self, future_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Run full SHAP analysis on forecasts.
        
        Args:
            future_df: DataFrame with 'ds' column for dates to explain.
                      If None, uses last 30 days of training data.
        
        Returns:
            Dict with inflow and outflow explanations
        """
        if future_df is None:
            # Use recent training data for explanation
            training_dates = self.forecaster.training_data['date'].tail(30)
            future_df = pd.DataFrame({'ds': training_dates})
        
        # Get Prophet predictions with components
        inflow_pred = self.forecaster.inflow_model.predict(future_df)
        outflow_pred = self.forecaster.outflow_model.predict(future_df)
        
        # Create explainers
        self.inflow_explainer = ProphetExplainer(
            self.forecaster.inflow_model, "Inflow"
        )
        self.outflow_explainer = ProphetExplainer(
            self.forecaster.outflow_model, "Outflow"
        )
        
        # Generate explanations
        inflow_explanation = self.inflow_explainer.explain(inflow_pred)
        outflow_explanation = self.outflow_explainer.explain(outflow_pred)
        
        # Get feature importance
        self.inflow_importance = self.inflow_explainer.get_feature_importance()
        self.outflow_importance = self.outflow_explainer.get_feature_importance()
        
        return {
            'inflow': {
                'explanation': inflow_explanation,
                'importance': self.inflow_importance,
            },
            'outflow': {
                'explanation': outflow_explanation,
                'importance': self.outflow_importance,
            }
        }
    
    def print_report(self):
        """Print SHAP analysis report."""
        print("\n" + "="*70)
        print("SHAP-LIKE FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        print("(What's driving the cash flow forecasts?)")
        
        print(f"\n{'─'*70}")
        print("INFLOW MODEL - Feature Importance")
        print(f"{'─'*70}")
        
        if self.inflow_importance is not None:
            for _, row in self.inflow_importance.iterrows():
                bar_len = int(row['importance_pct'] / 5)
                bar = '█' * bar_len
                print(f"  {row['component']:<15} {row['importance_pct']:>6.1f}%  {bar}")
        
        print(f"\n{'─'*70}")
        print("OUTFLOW MODEL - Feature Importance")
        print(f"{'─'*70}")
        
        if self.outflow_importance is not None:
            for _, row in self.outflow_importance.iterrows():
                bar_len = int(row['importance_pct'] / 5)
                bar = '█' * bar_len
                print(f"  {row['component']:<15} {row['importance_pct']:>6.1f}%  {bar}")
        
        print(f"\n{'─'*70}")
        print("INTERPRETATION")
        print(f"{'─'*70}")
        print("""
  • TREND: Long-term cash flow direction
  • WEEKLY: Day-of-week patterns (e.g., Friday AP runs)
  • MONTHLY: Day-of-month patterns (e.g., 15th payroll)
  • BIWEEKLY: Bi-weekly patterns (e.g., payroll cycles)
  • QUARTERLY: Quarterly patterns (e.g., tax payments)
  • YEARLY: Annual seasonality
        """)


# =============================================================================
# OUTLIER DETECTION
# =============================================================================
class OutlierDetector:
    """
    Detect outliers in cash flow data using multiple methods.
    
    Methods:
    1. Z-Score: Standard deviation based
    2. IQR: Interquartile range based
    3. Modified Z-Score: Robust to extreme outliers
    4. Forecast Deviation: Compares actuals to forecast
    """
    
    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        """
        Initialize outlier detector.
        
        Args:
            z_threshold: Z-score threshold for outlier detection
            iqr_multiplier: IQR multiplier for outlier detection
        """
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.outliers = None
        self.summary = None
    
    def detect(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Detect outliers in the specified columns.
        
        Args:
            df: DataFrame with cash flow data
            columns: Columns to check for outliers. 
                    Default: ['inflow', 'outflow', 'net_cash_flow']
        
        Returns:
            DataFrame with outlier flags and scores
        """
        if columns is None:
            columns = ['inflow', 'outflow', 'net_cash_flow']
        
        # Filter to columns that exist
        columns = [c for c in columns if c in df.columns]
        
        result = df.copy()
        
        for col in columns:
            data = df[col].values
            
            # Method 1: Z-Score
            z_scores = np.abs(zscore(data, nan_policy='omit'))
            result[f'{col}_zscore'] = z_scores
            result[f'{col}_outlier_zscore'] = z_scores > self.z_threshold
            
            # Method 2: IQR
            Q1 = np.nanpercentile(data, 25)
            Q3 = np.nanpercentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_multiplier * IQR
            upper_bound = Q3 + self.iqr_multiplier * IQR
            result[f'{col}_outlier_iqr'] = (data < lower_bound) | (data > upper_bound)
            result[f'{col}_iqr_lower'] = lower_bound
            result[f'{col}_iqr_upper'] = upper_bound
            
            # Method 3: Modified Z-Score (more robust)
            median = np.nanmedian(data)
            mad = np.nanmedian(np.abs(data - median))
            if mad > 0:
                modified_z = 0.6745 * (data - median) / mad
            else:
                modified_z = np.zeros_like(data)
            result[f'{col}_modified_zscore'] = np.abs(modified_z)
            result[f'{col}_outlier_modified_z'] = np.abs(modified_z) > self.z_threshold
            
            # Combined outlier flag (any method)
            result[f'{col}_is_outlier'] = (
                result[f'{col}_outlier_zscore'] | 
                result[f'{col}_outlier_iqr'] |
                result[f'{col}_outlier_modified_z']
            )
        
        # Overall outlier flag
        outlier_cols = [c for c in result.columns if c.endswith('_is_outlier')]
        result['is_outlier'] = result[outlier_cols].any(axis=1)
        
        self.outliers = result
        return result
    
    def detect_forecast_deviations(
        self, 
        actuals: pd.DataFrame, 
        forecasts: pd.DataFrame,
        threshold_pct: float = 20.0
    ) -> pd.DataFrame:
        """
        Detect days where forecast significantly deviates from actuals.
        
        Args:
            actuals: DataFrame with actual cash flows
            forecasts: DataFrame with forecasted cash flows
            threshold_pct: Percentage deviation threshold
        
        Returns:
            DataFrame with deviation analysis
        """
        # Merge actuals and forecasts
        merged = actuals.merge(
            forecasts[['date', 'forecast_inflow', 'forecast_outflow', 'forecast_net']],
            on='date',
            how='inner'
        )
        
        if len(merged) == 0:
            return pd.DataFrame()
        
        # Calculate deviations
        merged['inflow_deviation'] = merged['forecast_inflow'] - merged['inflow']
        merged['inflow_deviation_pct'] = np.where(
            merged['inflow'] != 0,
            (merged['inflow_deviation'] / merged['inflow']) * 100,
            0
        )
        
        merged['outflow_deviation'] = merged['forecast_outflow'] - merged['outflow']
        merged['outflow_deviation_pct'] = np.where(
            merged['outflow'] != 0,
            (merged['outflow_deviation'] / merged['outflow']) * 100,
            0
        )
        
        merged['net_deviation'] = merged['forecast_net'] - merged['net_cash_flow']
        merged['net_deviation_pct'] = np.where(
            merged['net_cash_flow'] != 0,
            (merged['net_deviation'] / merged['net_cash_flow']) * 100,
            0
        )
        
        # Flag significant deviations
        merged['inflow_significant_deviation'] = np.abs(merged['inflow_deviation_pct']) > threshold_pct
        merged['outflow_significant_deviation'] = np.abs(merged['outflow_deviation_pct']) > threshold_pct
        merged['net_significant_deviation'] = np.abs(merged['net_deviation_pct']) > threshold_pct
        
        merged['has_significant_deviation'] = (
            merged['inflow_significant_deviation'] |
            merged['outflow_significant_deviation'] |
            merged['net_significant_deviation']
        )
        
        return merged
    
    def get_outlier_summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """Get summary of detected outliers."""
        if df is None:
            df = self.outliers
        
        if df is None:
            return pd.DataFrame()
        
        outlier_rows = df[df['is_outlier']]
        
        if len(outlier_rows) == 0:
            return pd.DataFrame({'message': ['No outliers detected']})
        
        # Build summary
        summary_data = []
        for _, row in outlier_rows.iterrows():
            reasons = []
            
            for col in ['inflow', 'outflow', 'net_cash_flow']:
                if f'{col}_is_outlier' in row and row[f'{col}_is_outlier']:
                    z = row.get(f'{col}_zscore', 0)
                    value = row.get(col, 0)
                    reasons.append(f"{col}: ${value:,.0f} (z={z:.1f})")
            
            summary_data.append({
                'date': row['date'],
                'day_name': row['date'].strftime('%A') if hasattr(row['date'], 'strftime') else '',
                'reasons': '; '.join(reasons),
                'inflow': row.get('inflow', 0),
                'outflow': row.get('outflow', 0),
                'net': row.get('net_cash_flow', 0),
            })
        
        self.summary = pd.DataFrame(summary_data)
        return self.summary
    
    def print_report(self, df: pd.DataFrame = None):
        """Print outlier detection report."""
        if df is None:
            df = self.outliers
        
        if df is None:
            print("No outlier analysis available. Call detect() first.")
            return
        
        print("\n" + "="*70)
        print("OUTLIER DETECTION REPORT")
        print("="*70)
        
        # Count outliers by type
        total_days = len(df)
        outlier_days = df['is_outlier'].sum()
        
        print(f"\n  Total Days Analyzed: {total_days}")
        print(f"  Days with Outliers:  {outlier_days} ({outlier_days/total_days*100:.1f}%)")
        
        # Breakdown by column
        print(f"\n{'─'*70}")
        print("OUTLIERS BY CASH FLOW TYPE")
        print(f"{'─'*70}")
        
        for col in ['inflow', 'outflow', 'net_cash_flow']:
            outlier_col = f'{col}_is_outlier'
            if outlier_col in df.columns:
                count = df[outlier_col].sum()
                pct = count / total_days * 100
                print(f"  {col:<15}: {count:>4} days ({pct:>5.1f}%)")
        
        # Show outlier details
        summary = self.get_outlier_summary(df)
        
        if len(summary) > 0 and 'message' not in summary.columns:
            print(f"\n{'─'*70}")
            print("OUTLIER DETAILS (Top 10)")
            print(f"{'─'*70}")
            
            for _, row in summary.head(10).iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
                print(f"\n  {date_str} ({row['day_name']})")
                print(f"    Inflow:  ${row['inflow']:>12,.0f}")
                print(f"    Outflow: ${row['outflow']:>12,.0f}")
                print(f"    Net:     ${row['net']:>12,.0f}")
                print(f"    Reason:  {row['reasons']}")
        
        print(f"\n{'─'*70}")
        print("INTERPRETATION")
        print(f"{'─'*70}")
        print("""
  Outliers may indicate:
  • Unexpected large receipts or payments
  • Data entry errors
  • One-time transactions (M&A, special dividends)
  • Seasonal spikes (tax payments, bonus payouts)
  
  Action: Review outlier days with treasury team to:
  • Validate data accuracy
  • Determine if patterns should be modeled
  • Identify need for special event handling
        """)


# =============================================================================
# TREND DECOMPOSITION
# =============================================================================
class TrendDecomposer:
    """
    Decompose cash flow time series into trend, seasonality, and residual.
    Uses STL (Seasonal and Trend decomposition using Loess).
    """
    
    def __init__(self):
        self.decomposition = None
    
    def decompose(
        self, 
        df: pd.DataFrame, 
        column: str = 'net_cash_flow',
        period: int = 5  # Weekly for banking days
    ) -> Dict[str, pd.Series]:
        """
        Decompose time series into components.
        
        Args:
            df: DataFrame with cash flow data
            column: Column to decompose
            period: Seasonal period (5 = weekly for banking days)
        
        Returns:
            Dict with trend, seasonal, and residual components
        """
        from statsmodels.tsa.seasonal import STL
        
        # Prepare data
        data = df.set_index('date')[column].copy()
        
        # Handle missing values
        data = data.interpolate()
        
        # STL decomposition
        stl = STL(data, period=period, robust=True)
        result = stl.fit()
        
        self.decomposition = {
            'observed': result.observed,
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
        }
        
        return self.decomposition
    
    def print_report(self):
        """Print decomposition summary."""
        if self.decomposition is None:
            print("No decomposition available. Call decompose() first.")
            return
        
        print("\n" + "="*70)
        print("TREND DECOMPOSITION ANALYSIS")
        print("="*70)
        
        trend = self.decomposition['trend']
        seasonal = self.decomposition['seasonal']
        residual = self.decomposition['residual']
        
        print(f"\n  TREND COMPONENT:")
        print(f"    Start:   ${trend.iloc[0]:>12,.0f}")
        print(f"    End:     ${trend.iloc[-1]:>12,.0f}")
        print(f"    Change:  ${trend.iloc[-1] - trend.iloc[0]:>12,.0f}")
        
        trend_direction = "📈 Upward" if trend.iloc[-1] > trend.iloc[0] else "📉 Downward"
        print(f"    Direction: {trend_direction}")
        
        print(f"\n  SEASONAL COMPONENT:")
        print(f"    Max:     ${seasonal.max():>12,.0f}")
        print(f"    Min:     ${seasonal.min():>12,.0f}")
        print(f"    Range:   ${seasonal.max() - seasonal.min():>12,.0f}")
        
        print(f"\n  RESIDUAL (Unexplained):")
        print(f"    Std Dev: ${residual.std():>12,.0f}")
        print(f"    Max:     ${residual.max():>12,.0f}")
        print(f"    Min:     ${residual.min():>12,.0f}")
        
        # Calculate variance explained
        total_var = self.decomposition['observed'].var()
        trend_var = trend.var()
        seasonal_var = seasonal.var()
        residual_var = residual.var()
        
        print(f"\n  VARIANCE EXPLAINED:")
        print(f"    Trend:    {trend_var/total_var*100:>6.1f}%")
        print(f"    Seasonal: {seasonal_var/total_var*100:>6.1f}%")
        print(f"    Residual: {residual_var/total_var*100:>6.1f}%")


# =============================================================================
# MAIN ANALYSIS RUNNER
# =============================================================================
def run_full_analysis(
    forecaster,
    historical_df: pd.DataFrame,
    forecasts: Dict[str, pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Run complete analysis suite.
    
    Args:
        forecaster: Fitted ProphetCashForecaster
        historical_df: Historical cash flow data
        forecasts: Dict of forecast DataFrames (optional)
    
    Returns:
        Dict with all analysis results
    """
    results = {}
    
    # 1. SHAP Analysis
    print("\n[1/3] Running SHAP Analysis...")
    shap_analyzer = SHAPAnalyzer(forecaster)
    results['shap'] = shap_analyzer.analyze()
    shap_analyzer.print_report()
    
    # 2. Outlier Detection
    print("\n[2/3] Running Outlier Detection...")
    outlier_detector = OutlierDetector(z_threshold=3.0)
    outliers = outlier_detector.detect(historical_df)
    results['outliers'] = outliers
    results['outlier_summary'] = outlier_detector.get_outlier_summary()
    outlier_detector.print_report()
    
    # 3. Trend Decomposition
    print("\n[3/3] Running Trend Decomposition...")
    decomposer = TrendDecomposer()
    
    # Filter to banking days for decomposition
    from models_prophet_v2 import USBankingCalendar
    holidays = USBankingCalendar.get_us_holidays(
        historical_df['date'].min().year,
        historical_df['date'].max().year + 1
    )
    banking_df = historical_df[historical_df['date'].apply(
        lambda x: USBankingCalendar.is_banking_day(x, holidays)
    )].copy()
    
    if len(banking_df) > 10:
        results['decomposition'] = decomposer.decompose(banking_df)
        decomposer.print_report()
    
    # Store analyzers for later use
    results['shap_analyzer'] = shap_analyzer
    results['outlier_detector'] = outlier_detector
    results['decomposer'] = decomposer
    
    return results


if __name__ == "__main__":
    from data_simulator_realistic import generate_sample_data
    from models_prophet_v2 import ProphetCashForecaster, run_backtest
    
    print("="*70)
    print("CASH FORECAST ANALYSIS - SHAP & OUTLIER DETECTION")
    print("="*70)
    
    # Generate data
    print("\nGenerating test data...")
    data = generate_sample_data(periods=730)
    daily_cash = data['daily_cash_position']
    
    # Train model
    print("\nTraining model...")
    train_df = daily_cash.iloc[:-90].copy()
    test_df = daily_cash.iloc[-90:].copy()
    
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df)
    
    # Run full analysis
    print("\n" + "="*70)
    print("RUNNING FULL ANALYSIS SUITE")
    print("="*70)
    
    results = run_full_analysis(forecaster, train_df)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
