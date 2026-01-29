"""
Prophet Cash Forecasting v4
===========================
CAPEX excluded from model training - will be user input.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

from config import TIME_HORIZONS, MAPE_THRESHOLDS


class USBankingCalendar:
    @staticmethod
    def get_us_holidays(start_year: int, end_year: int) -> List[datetime]:
        holidays = []
        for year in range(start_year, end_year + 1):
            holidays.append(datetime(year, 1, 1))
            holidays.append(datetime(year, 7, 4))
            holidays.append(datetime(year, 11, 11))
            holidays.append(datetime(year, 12, 25))
            nov1 = datetime(year, 11, 1)
            days_until_thu = (3 - nov1.weekday()) % 7
            thanksgiving = nov1 + timedelta(days=days_until_thu + 21)
            holidays.append(thanksgiving)
        return holidays
    
    @staticmethod
    def is_banking_day(date: pd.Timestamp, holidays: List[datetime]) -> bool:
        if date.weekday() >= 5:
            return False
        if date.date() in [h.date() for h in holidays]:
            return False
        return True
    
    @staticmethod
    def get_banking_days(start_date: datetime, num_days: int) -> pd.DatetimeIndex:
        holidays = USBankingCalendar.get_us_holidays(start_date.year, start_date.year + 2)
        banking_days = []
        current = start_date + timedelta(days=1)
        while len(banking_days) < num_days:
            if USBankingCalendar.is_banking_day(pd.Timestamp(current), holidays):
                banking_days.append(current)
            current += timedelta(days=1)
        return pd.DatetimeIndex(banking_days)


class CategoryProportions:
    """Calculate and apply category proportions by day-of-week. CAPEX EXCLUDED."""
    
    # Categories to include in model (CAPEX excluded)
    OUTFLOW_CATEGORIES = ['PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT']  # No CAPEX
    INFLOW_CATEGORIES = ['AR', 'INV_INC', 'IC_IN']
    
    def __init__(self):
        self.inflow_by_dow = {}
        self.outflow_by_dow = {}
    
    def fit(self, category_df: pd.DataFrame):
        df = category_df[category_df['is_banking_day']].copy()
        
        for dow in range(5):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) > 0:
                # Inflows (unchanged)
                inflow_total = dow_df[self.INFLOW_CATEGORIES].sum().sum()
                self.inflow_by_dow[dow] = {
                    cat: dow_df[cat].sum() / inflow_total if inflow_total > 0 else 0
                    for cat in self.INFLOW_CATEGORIES
                }
                
                # Outflows (CAPEX excluded)
                outflow_total = dow_df[self.OUTFLOW_CATEGORIES].sum().sum()
                self.outflow_by_dow[dow] = {
                    cat: dow_df[cat].sum() / outflow_total if outflow_total > 0 else 0
                    for cat in self.OUTFLOW_CATEGORIES
                }
        
        return self
    
    def allocate_inflow(self, total: float, dow: int) -> Dict[str, float]:
        props = self.inflow_by_dow.get(dow, {cat: 1/len(self.INFLOW_CATEGORIES) for cat in self.INFLOW_CATEGORIES})
        return {cat: total * props.get(cat, 0) for cat in self.INFLOW_CATEGORIES}
    
    def allocate_outflow(self, total: float, dow: int) -> Dict[str, float]:
        props = self.outflow_by_dow.get(dow, {cat: 1/len(self.OUTFLOW_CATEGORIES) for cat in self.OUTFLOW_CATEGORIES})
        return {cat: total * props.get(cat, 0) for cat in self.OUTFLOW_CATEGORIES}


class ProphetCashForecaster:
    """
    Prophet forecaster with CAPEX excluded from training.
    CAPEX will be added as user input to forecasts.
    """
    
    # Categories excluded from model (user input instead)
    EXCLUDED_OUTFLOWS = ['CAPEX']
    
    def __init__(self):
        self.inflow_model = None
        self.outflow_model = None
        self.is_fitted = False
        self.training_data = None
        self.last_actual_date = None
        self.last_actual_closing_balance = None
        self.category_props = CategoryProportions()
        self.holidays = []
    
    def fit(self, df: pd.DataFrame, category_df: pd.DataFrame = None) -> 'ProphetCashForecaster':
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        min_year = df['date'].min().year
        max_year = df['date'].max().year + 2
        self.holidays = USBankingCalendar.get_us_holidays(min_year, max_year)
        
        df['is_banking_day'] = df['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, self.holidays)
        )
        banking_df = df[df['is_banking_day']].copy()
        
        self.training_data = df
        self.last_actual_date = df['date'].iloc[-1]
        self.last_actual_closing_balance = df['closing_balance'].iloc[-1]
        
        # Calculate outflow excluding CAPEX for training
        if category_df is not None:
            self.category_props.fit(category_df)
            
            # Create adjusted outflow column (excluding CAPEX)
            cat_df = category_df.copy()
            cat_df['outflow_ex_capex'] = (
                cat_df['PAYROLL'] + cat_df['AP'] + cat_df['TAX'] + 
                cat_df['DEBT'] + cat_df['IC_OUT']
            )
            
            # Merge back to banking_df
            banking_df = banking_df.merge(
                cat_df[['date', 'outflow_ex_capex']], 
                on='date', 
                how='left'
            )
            outflow_col = 'outflow_ex_capex'
            
            # Calculate original CAPEX total for reporting
            self.total_historical_capex = cat_df['CAPEX'].sum()
            self.avg_monthly_capex = self.total_historical_capex / (len(cat_df) / 30)
        else:
            outflow_col = 'outflow'
            self.total_historical_capex = 0
            self.avg_monthly_capex = 0
        
        print(f"\nTraining on {len(banking_df)} banking days (CAPEX EXCLUDED)...")
        print(f"T0: {self.last_actual_date.strftime('%Y-%m-%d')}, Balance: ${self.last_actual_closing_balance:,.0f}")
        print(f"Historical CAPEX excluded: ${self.total_historical_capex:,.0f} total, ${self.avg_monthly_capex:,.0f}/month avg")
        
        inflow_df = pd.DataFrame({'ds': banking_df['date'], 'y': banking_df['inflow']})
        outflow_df = pd.DataFrame({'ds': banking_df['date'], 'y': banking_df[outflow_col]})
        
        self.inflow_model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0
        )
        self.inflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.inflow_model.fit(inflow_df)
        
        self.outflow_model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
            changepoint_prior_scale=0.05, seasonality_prior_scale=10.0
        )
        self.outflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.outflow_model.add_seasonality(name='biweekly', period=15.22, fourier_order=5)
        self.outflow_model.fit(outflow_df)
        
        self.is_fitted = True
        print("✅ Models trained (CAPEX excluded from outflows)")
        return self
    
    def predict(self, horizon: str = None, capex_schedule: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts.
        
        Args:
            horizon: Specific horizon or None for all
            capex_schedule: Optional dict of {date_str: amount} for planned CAPEX
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        horizons = {horizon: TIME_HORIZONS[horizon]} if horizon else TIME_HORIZONS
        results = {}
        
        for hz_name, hz_config in horizons.items():
            num_days = hz_config['days']
            future_dates = USBankingCalendar.get_banking_days(
                self.last_actual_date.to_pydatetime(), num_days
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            inflow_pred = self.inflow_model.predict(future_df)
            outflow_pred = self.outflow_model.predict(future_df)
            
            forecast_inflows = np.maximum(inflow_pred['yhat'].values, 0)
            forecast_outflows = np.maximum(outflow_pred['yhat'].values, 0)  # This is ex-CAPEX
            
            rows = []
            current_balance = self.last_actual_closing_balance
            
            for i in range(num_days):
                date = future_dates[i]
                date_str = date.strftime('%Y-%m-%d')
                dow = date.dayofweek
                
                inflow_total = forecast_inflows[i]
                outflow_ex_capex = forecast_outflows[i]
                
                # Get CAPEX from schedule if provided
                capex = 0
                if capex_schedule and date_str in capex_schedule:
                    capex = capex_schedule[date_str]
                
                outflow_total = outflow_ex_capex + capex
                
                inflow_cats = self.category_props.allocate_inflow(inflow_total, dow)
                outflow_cats = self.category_props.allocate_outflow(outflow_ex_capex, dow)
                
                opening = current_balance
                net = inflow_total - outflow_total
                closing = opening + net
                
                row = {
                    'date': date,
                    'horizon_day': i + 1,
                    'day_of_week': dow,
                    'day_name': date.day_name(),
                    'opening_balance': opening,
                    # Inflows
                    'AR': inflow_cats.get('AR', 0),
                    'INV_INC': inflow_cats.get('INV_INC', 0),
                    'IC_IN': inflow_cats.get('IC_IN', 0),
                    'forecast_inflow': inflow_total,
                    # Outflows (CAPEX separate)
                    'PAYROLL': outflow_cats.get('PAYROLL', 0),
                    'AP': outflow_cats.get('AP', 0),
                    'TAX': outflow_cats.get('TAX', 0),
                    'DEBT': outflow_cats.get('DEBT', 0),
                    'IC_OUT': outflow_cats.get('IC_OUT', 0),
                    'CAPEX': capex,  # User input or 0
                    'forecast_outflow_ex_capex': outflow_ex_capex,
                    'forecast_outflow': outflow_total,
                    'forecast_net': net,
                    'closing_balance': closing,
                    'horizon': hz_name,
                }
                rows.append(row)
                current_balance = closing
            
            results[hz_name] = pd.DataFrame(rows)
        
        return results


class ForecastAnalyzer:
    """Analyze forecast accuracy - now comparing ex-CAPEX outflows."""
    
    def __init__(self):
        self.results = {}
    
    def analyze(self, forecasts: Dict[str, pd.DataFrame], actuals: pd.DataFrame, 
                category_actuals: pd.DataFrame = None) -> Dict[str, Any]:
        
        holidays = USBankingCalendar.get_us_holidays(
            actuals['date'].min().year, actuals['date'].max().year + 1
        )
        actuals['is_banking_day'] = actuals['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, holidays)
        )
        actuals_banking = actuals[actuals['is_banking_day']].copy()
        
        # Calculate actual outflow ex-CAPEX if category data available
        if category_actuals is not None:
            cat_banking = category_actuals[category_actuals['is_banking_day']].copy()
            cat_banking['outflow_ex_capex'] = (
                cat_banking['PAYROLL'] + cat_banking['AP'] + cat_banking['TAX'] + 
                cat_banking['DEBT'] + cat_banking['IC_OUT']
            )
            actuals_banking = actuals_banking.merge(
                cat_banking[['date', 'outflow_ex_capex', 'CAPEX']],
                on='date', how='left'
            )
        
        results = {}
        
        for horizon, forecast_df in forecasts.items():
            forecast_renamed = forecast_df.rename(columns={
                'closing_balance': 'forecast_closing_balance',
                'opening_balance': 'forecast_opening_balance'
            })
            
            merged = forecast_renamed.merge(
                actuals_banking[['date', 'inflow', 'outflow', 'net_cash_flow', 'closing_balance'] + 
                               (['outflow_ex_capex', 'CAPEX'] if 'outflow_ex_capex' in actuals_banking.columns else [])],
                on='date', how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            inflow_mape = self._mape(merged['inflow'].values, merged['forecast_inflow'].values)
            
            # Compare outflow ex-CAPEX if available
            if 'outflow_ex_capex' in merged.columns:
                outflow_mape = self._mape(merged['outflow_ex_capex'].values, merged['forecast_outflow_ex_capex'].values)
                outflow_label = "Outflow (ex-CAPEX)"
            else:
                outflow_mape = self._mape(merged['outflow'].values, merged['forecast_outflow'].values)
                outflow_label = "Outflow"
            
            balance_mape = self._mape(merged['closing_balance'].values, merged['forecast_closing_balance'].values)
            
            # MAPE by horizon day
            merged['balance_pct_error'] = np.abs(
                (merged['forecast_closing_balance'] - merged['closing_balance']) / merged['closing_balance']
            ) * 100
            merged['inflow_pct_error'] = np.abs(
                (merged['forecast_inflow'] - merged['inflow']) / merged['inflow'].replace(0, np.nan)
            ) * 100
            
            if 'outflow_ex_capex' in merged.columns:
                merged['outflow_pct_error'] = np.abs(
                    (merged['forecast_outflow_ex_capex'] - merged['outflow_ex_capex']) / merged['outflow_ex_capex'].replace(0, np.nan)
                ) * 100
            else:
                merged['outflow_pct_error'] = np.abs(
                    (merged['forecast_outflow'] - merged['outflow']) / merged['outflow'].replace(0, np.nan)
                ) * 100
            
            mape_by_day = merged.groupby('horizon_day').agg({
                'balance_pct_error': 'mean',
                'inflow_pct_error': 'mean',
                'outflow_pct_error': 'mean'
            }).reset_index()
            mape_by_day.columns = ['horizon_day', 'balance_mape', 'inflow_mape', 'outflow_mape']
            
            thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
            if balance_mape <= thresholds["excellent"]:
                rating = "Excellent"
            elif balance_mape <= thresholds["good"]:
                rating = "Good"
            elif balance_mape <= thresholds["acceptable"]:
                rating = "Acceptable"
            else:
                rating = "Poor"
            
            results[horizon] = {
                'inflow_mape': inflow_mape,
                'outflow_mape': outflow_mape,
                'outflow_label': outflow_label,
                'balance_mape': balance_mape,
                'rating': rating,
                'samples': len(merged),
                'mape_by_horizon_day': mape_by_day,
            }
        
        self.results = results
        return results
    
    def _mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def run_backtest(df: pd.DataFrame, test_size: int = 90, category_df: pd.DataFrame = None):
    """Run backtest with CAPEX excluded."""
    
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    train_cat_df = category_df.iloc[:-test_size].copy() if category_df is not None else None
    test_cat_df = category_df.iloc[-test_size:].copy() if category_df is not None else None
    
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df, train_cat_df)
    forecasts = forecaster.predict()
    
    analyzer = ForecastAnalyzer()
    results = analyzer.analyze(forecasts, test_df, test_cat_df)
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    from data_simulator_v2 import generate_category_data
    
    print("="*60)
    print("Testing Prophet v4 (CAPEX EXCLUDED)")
    print("="*60)
    
    data = generate_category_data(periods=365)
    
    # Run with CAPEX excluded
    results, forecaster, forecasts = run_backtest(
        data['daily_cash_position'], test_size=30, category_df=data['category_details']
    )
    
    print("\n" + "="*60)
    print("ACCURACY RESULTS (CAPEX EXCLUDED)")
    print("="*60)
    
    for horizon, metrics in results.items():
        print(f"\n{horizon}:")
        print(f"  Inflow MAPE:  {metrics['inflow_mape']:.2f}%")
        print(f"  {metrics['outflow_label']} MAPE: {metrics['outflow_mape']:.2f}%")
        print(f"  Balance MAPE: {metrics['balance_mape']:.2f}%")
        print(f"  Rating: {metrics['rating']}")
