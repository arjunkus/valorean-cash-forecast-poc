"""
Analysis Module v3 - Forward-Looking Forecast Alerts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta


class ForecastAlertDetector:
    """
    Forward-looking alert detection for actionable treasury insights.

    Analyzes FORECASTED cash flows to alert users about:
    - Unusual inflows/outflows in upcoming days
    - Liquidity risk (projected balance below threshold)
    - Large scheduled payments requiring attention
    """

    # Alert thresholds
    DEVIATION_HIGH = 0.40    # 40% deviation = High priority
    DEVIATION_MEDIUM = 0.25  # 25% deviation = Medium priority
    LIQUIDITY_BUFFER = 5_000_000  # $5M minimum balance threshold

    CATEGORY_NAMES = {
        'AR': 'Accounts Receivable',
        'AP': 'Accounts Payable',
        'PAYROLL': 'Payroll',
        'IC_IN': 'Intercompany In',
        'IC_OUT': 'Intercompany Out',
        'TAX': 'Tax Payment',
        'DEBT': 'Debt Service',
        'INV_INC': 'Investment Income',
        'CAPEX': 'Capital Expenditure',
    }

    def __init__(self):
        self.alerts_df = None
        self.historical_stats = {}
        self.results = None

    def detect(self, forecast_df, historical_df, category_df=None):
        """
        Detect alerts in FORECASTED data by comparing to historical patterns.

        Args:
            forecast_df: DataFrame with forecasted values (T+1 to T+30)
            historical_df: DataFrame with historical actuals (for computing baselines)
            category_df: Optional category breakdown for detailed alerts
        """
        alerts_list = []

        # Compute historical baselines from actuals
        self._compute_historical_stats(historical_df, category_df)

        # Analyze each forecasted day
        for _, row in forecast_df.iterrows():
            day_alerts = self._analyze_forecast_day(row)
            alerts_list.extend(day_alerts)

        # Check for liquidity risk
        liquidity_alerts = self._check_liquidity_risk(forecast_df)
        alerts_list.extend(liquidity_alerts)

        if alerts_list:
            self.alerts_df = pd.DataFrame(alerts_list)
            self.alerts_df = self.alerts_df.sort_values(['severity_order', 'date']).reset_index(drop=True)
            self.alerts_df = self.alerts_df.drop(columns=['severity_order'])
        else:
            self.alerts_df = pd.DataFrame(columns=[
                'date', 'day_name', 'severity', 'alert_type',
                'description', 'recommended_action', 'forecast_value', 'expected_value'
            ])

        self.results = forecast_df
        return self.alerts_df

    def _compute_historical_stats(self, historical_df, category_df=None):
        """Compute historical averages by day of week for comparison."""
        df = historical_df.copy()
        if 'is_banking_day' in df.columns:
            df = df[df['is_banking_day']].copy()

        df['day_of_week'] = df['date'].dt.dayofweek

        # Overall stats
        self.historical_stats['inflow'] = {
            'mean': df['inflow'].mean(),
            'std': df['inflow'].std(),
            'by_dow': df.groupby('day_of_week')['inflow'].mean().to_dict()
        }
        self.historical_stats['outflow'] = {
            'mean': df['outflow'].mean(),
            'std': df['outflow'].std(),
            'by_dow': df.groupby('day_of_week')['outflow'].mean().to_dict()
        }
        self.historical_stats['net_flow'] = {
            'mean': df['net_cash_flow'].mean(),
            'std': df['net_cash_flow'].std(),
        }

        # Category stats if available
        if category_df is not None:
            cat_df = category_df.copy()
            if 'is_banking_day' in cat_df.columns:
                cat_df = cat_df[cat_df['is_banking_day']].copy()

            for cat in ['AR', 'AP', 'PAYROLL', 'TAX', 'DEBT', 'CAPEX']:
                if cat in cat_df.columns:
                    non_zero = cat_df[cat_df[cat] > 0][cat]
                    if len(non_zero) > 0:
                        self.historical_stats[cat] = {
                            'mean': non_zero.mean(),
                            'std': non_zero.std(),
                        }

    def _analyze_forecast_day(self, row):
        """Analyze a single forecasted day for anomalies."""
        alerts = []
        date = row['date']
        day_name = date.day_name() if hasattr(date, 'day_name') else pd.Timestamp(date).day_name()
        dow = date.dayofweek if hasattr(date, 'dayofweek') else pd.Timestamp(date).dayofweek

        # Check inflow forecast
        if 'forecast_inflow' in row:
            forecast_val = row['forecast_inflow']
            expected = self.historical_stats['inflow']['by_dow'].get(dow, self.historical_stats['inflow']['mean'])

            if expected > 0:
                pct_diff = (forecast_val - expected) / expected

                if abs(pct_diff) >= self.DEVIATION_HIGH:
                    severity = 'High'
                    severity_order = 1
                elif abs(pct_diff) >= self.DEVIATION_MEDIUM:
                    severity = 'Medium'
                    severity_order = 2
                else:
                    severity = None

                if severity:
                    if pct_diff < 0:
                        alert_type = 'Low Inflow Forecast'
                        description = f"Forecasted collections ${forecast_val:,.0f} is {abs(pct_diff)*100:.0f}% below typical {day_name} (${expected:,.0f})"
                        action = "Verify expected customer payments. Follow up on outstanding invoices."
                    else:
                        alert_type = 'High Inflow Forecast'
                        description = f"Forecasted collections ${forecast_val:,.0f} is {abs(pct_diff)*100:.0f}% above typical {day_name} (${expected:,.0f})"
                        action = "Confirm large incoming payments. Update investment strategy if recurring."

                    alerts.append({
                        'date': date,
                        'day_name': day_name,
                        'severity': severity,
                        'severity_order': severity_order,
                        'alert_type': alert_type,
                        'description': description,
                        'recommended_action': action,
                        'forecast_value': forecast_val,
                        'expected_value': expected,
                    })

        # Check outflow forecast
        if 'forecast_outflow' in row:
            forecast_val = row['forecast_outflow']
            expected = self.historical_stats['outflow']['by_dow'].get(dow, self.historical_stats['outflow']['mean'])

            if expected > 0:
                pct_diff = (forecast_val - expected) / expected

                if abs(pct_diff) >= self.DEVIATION_HIGH:
                    severity = 'High'
                    severity_order = 1
                elif abs(pct_diff) >= self.DEVIATION_MEDIUM:
                    severity = 'Medium'
                    severity_order = 2
                else:
                    severity = None

                if severity:
                    if pct_diff > 0:
                        alert_type = 'High Outflow Forecast'
                        description = f"Forecasted payments ${forecast_val:,.0f} is {abs(pct_diff)*100:.0f}% above typical {day_name} (${expected:,.0f})"
                        action = "Review scheduled payments. Ensure sufficient liquidity."
                    else:
                        alert_type = 'Low Outflow Forecast'
                        description = f"Forecasted payments ${forecast_val:,.0f} is {abs(pct_diff)*100:.0f}% below typical {day_name} (${expected:,.0f})"
                        action = "Verify all invoices are scheduled. Check for delayed payments."

                    alerts.append({
                        'date': date,
                        'day_name': day_name,
                        'severity': severity,
                        'severity_order': severity_order,
                        'alert_type': alert_type,
                        'description': description,
                        'recommended_action': action,
                        'forecast_value': forecast_val,
                        'expected_value': expected,
                    })

        # Check category-specific forecasts
        for cat in ['PAYROLL', 'TAX', 'DEBT', 'CAPEX']:
            if cat in row and row[cat] > 0 and cat in self.historical_stats:
                forecast_val = row[cat]
                expected = self.historical_stats[cat]['mean']

                if expected > 0:
                    pct_diff = (forecast_val - expected) / expected

                    if pct_diff >= self.DEVIATION_HIGH:
                        cat_name = self.CATEGORY_NAMES.get(cat, cat)
                        alerts.append({
                            'date': date,
                            'day_name': day_name,
                            'severity': 'High',
                            'severity_order': 1,
                            'alert_type': f'Large {cat_name} Payment',
                            'description': f"{cat_name} ${forecast_val:,.0f} is {pct_diff*100:.0f}% above average (${expected:,.0f})",
                            'recommended_action': f"Verify {cat_name.lower()} amount. Ensure sufficient balance.",
                            'forecast_value': forecast_val,
                            'expected_value': expected,
                        })

        return alerts

    def _check_liquidity_risk(self, forecast_df):
        """Check for days where projected balance falls below threshold."""
        alerts = []

        if 'closing_balance' not in forecast_df.columns:
            return alerts

        for _, row in forecast_df.iterrows():
            balance = row['closing_balance']
            date = row['date']
            day_name = date.day_name() if hasattr(date, 'day_name') else pd.Timestamp(date).day_name()

            if balance < self.LIQUIDITY_BUFFER:
                shortage = self.LIQUIDITY_BUFFER - balance

                if balance < 0:
                    severity = 'High'
                    severity_order = 0  # Highest priority
                    alert_type = 'NEGATIVE BALANCE'
                    action = "IMMEDIATE ACTION: Arrange funding or defer payments."
                elif balance < self.LIQUIDITY_BUFFER * 0.5:
                    severity = 'High'
                    severity_order = 1
                    alert_type = 'Critical Liquidity Risk'
                    action = "Expedite collections or arrange credit facility."
                else:
                    severity = 'Medium'
                    severity_order = 2
                    alert_type = 'Liquidity Warning'
                    action = "Monitor closely. Consider deferring non-critical payments."

                alerts.append({
                    'date': date,
                    'day_name': day_name,
                    'severity': severity,
                    'severity_order': severity_order,
                    'alert_type': alert_type,
                    'description': f"Projected balance ${balance:,.0f} is ${shortage:,.0f} below ${self.LIQUIDITY_BUFFER/1e6:.0f}M buffer",
                    'recommended_action': action,
                    'forecast_value': balance,
                    'expected_value': self.LIQUIDITY_BUFFER,
                })

        return alerts

    def get_alerts(self):
        """Get all detected alerts."""
        return self.alerts_df

    def get_outliers(self):
        """Alias for backward compatibility."""
        return self.alerts_df

    def get_outlier_summary(self):
        """Get summary of alerts."""
        if self.alerts_df is None or len(self.alerts_df) == 0:
            return {
                'total_days': 0,
                'outlier_count': 0,
                'by_severity': {},
                'by_type': {},
            }

        return {
            'total_days': len(self.alerts_df['date'].unique()),
            'outlier_count': len(self.alerts_df),
            'by_severity': self.alerts_df['severity'].value_counts().to_dict(),
            'by_type': self.alerts_df['alert_type'].value_counts().to_dict(),
        }


# Backward compatibility alias
OutlierDetector = ForecastAlertDetector


class SHAPAnalyzer:
    """SHAP analysis for forecast interpretability."""

    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.results = {}

    def analyze(self):
        if not self.forecaster.is_fitted:
            return {}

        results = {}
        if self.forecaster.inflow_model is not None:
            results['inflow'] = self._analyze_model(self.forecaster.inflow_model, 'inflow')
        if self.forecaster.outflow_model is not None:
            results['outflow'] = self._analyze_model(self.forecaster.outflow_model, 'outflow')

        self.results = results
        return results

    def _analyze_model(self, model, name):
        import pandas as pd
        training_dates = self.forecaster.training_data['date'].tail(30)
        future_df = pd.DataFrame({'ds': training_dates})
        forecast = model.predict(future_df)

        skip_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                    'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                    'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']
        component_cols = [col for col in forecast.columns if col not in skip_cols]

        importance = {}
        for col in component_cols:
            if col in forecast.columns:
                importance[col] = abs(forecast[col]).mean()

        total = sum(importance.values()) or 1
        importance_pct = {k: v/total * 100 for k, v in importance.items()}

        importance_df = pd.DataFrame([
            {'component': k, 'importance': v, 'importance_pct': importance_pct[k]}
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])

        return {'importance': importance_df, 'raw_importance': importance}
