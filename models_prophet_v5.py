"""
Prophet Cash Forecasting v5
===========================
Cyclical events (Payroll, Debt, Tax) use LAST ACTUAL value, not trend prediction.
- Payroll: 15th and last business day → use last payroll amount
- Debt: 1st of month → use last debt amount
- Tax: Quarterly 15th → use last tax amount
- AP/AR/IC: Use Prophet trend prediction (truly pattern-based)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from prophet import Prophet
import warnings
import calendar

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
    
    @staticmethod
    def get_last_business_day(year: int, month: int, holidays: List[datetime]) -> datetime:
        """Get last business day of a month."""
        last_day = calendar.monthrange(year, month)[1]
        date = datetime(year, month, last_day)
        while not USBankingCalendar.is_banking_day(pd.Timestamp(date), holidays):
            date -= timedelta(days=1)
        return date
    
    @staticmethod
    def is_payroll_day(date: pd.Timestamp, holidays: List[datetime]) -> bool:
        """Check if date is a payroll day (15th or last business day)."""
        day = date.day
        
        # 15th of month (or next business day if 15th is not banking day)
        if day == 15:
            return True
        
        # Check if this is the last business day of month
        last_bd = USBankingCalendar.get_last_business_day(date.year, date.month, holidays)
        if date.date() == last_bd.date():
            return True
        
        return False
    
    @staticmethod
    def is_debt_day(date: pd.Timestamp) -> bool:
        """Check if date is debt service day (1st of month)."""
        return date.day == 1
    
    @staticmethod
    def is_tax_day(date: pd.Timestamp) -> bool:
        """Check if date is quarterly tax day (15th of Apr, Jun, Sep, Dec)."""
        return date.day == 15 and date.month in [4, 6, 9, 12]


class CyclicalEventTracker:
    """
    Tracks cyclical events and provides last actual values for forecasting.
    Uses "last actual" approach instead of trend prediction for scheduled payments.
    """
    
    def __init__(self):
        self.last_payroll = 0
        self.last_debt = 0
        self.last_tax = 0
        self.last_inv_income = 0
        self.payroll_dates = []
        self.debt_dates = []
        self.tax_dates = []
    
    def fit(self, category_df: pd.DataFrame, holidays: List[datetime]):
        """Extract last actual values for cyclical events."""
        
        df = category_df.copy()
        df['is_banking'] = df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
        banking_df = df[df['is_banking']].copy()
        
        # Last payroll amount (non-zero)
        payroll_days = banking_df[banking_df['PAYROLL'] > 0]
        if len(payroll_days) > 0:
            self.last_payroll = payroll_days['PAYROLL'].iloc[-1]
            self.payroll_dates = payroll_days['date'].tolist()
        
        # Last debt service amount (non-zero)
        debt_days = banking_df[banking_df['DEBT'] > 0]
        if len(debt_days) > 0:
            self.last_debt = debt_days['DEBT'].iloc[-1]
            self.debt_dates = debt_days['date'].tolist()
        
        # Last tax payment amount (non-zero)
        tax_days = banking_df[banking_df['TAX'] > 0]
        if len(tax_days) > 0:
            self.last_tax = tax_days['TAX'].iloc[-1]
            self.tax_dates = tax_days['date'].tolist()
        
        # Last investment income (non-zero)
        inv_days = banking_df[banking_df['INV_INC'] > 0]
        if len(inv_days) > 0:
            self.last_inv_income = inv_days['INV_INC'].iloc[-1]
        
        return self
    
    def get_scheduled_amount(self, date: pd.Timestamp, event_type: str, holidays: List[datetime]) -> float:
        """Get scheduled amount for a cyclical event on a specific date."""
        
        if event_type == 'PAYROLL':
            if USBankingCalendar.is_payroll_day(date, holidays):
                return self.last_payroll
            return 0
        
        elif event_type == 'DEBT':
            if USBankingCalendar.is_debt_day(date):
                return self.last_debt
            return 0
        
        elif event_type == 'TAX':
            if USBankingCalendar.is_tax_day(date):
                return self.last_tax
            return 0
        
        elif event_type == 'INV_INC':
            if date.day == 1:  # Investment income on 1st
                return self.last_inv_income
            return 0
        
        return 0
    
    def summary(self) -> Dict[str, Any]:
        """Return summary of cyclical event values."""
        return {
            'last_payroll': self.last_payroll,
            'last_debt': self.last_debt,
            'last_tax': self.last_tax,
            'last_inv_income': self.last_inv_income,
            'payroll_count': len(self.payroll_dates),
            'debt_count': len(self.debt_dates),
            'tax_count': len(self.tax_dates),
        }


class PatternBasedProportions:
    """
    Calculate proportions for truly pattern-based flows:
    - AR: Daily with day-of-week patterns
    - AP: Friday pattern
    - IC_IN: Monday pattern
    - IC_OUT: Wednesday pattern
    """
    
    def __init__(self):
        self.ar_by_dow = {}
        self.ap_by_dow = {}
        self.ic_in_by_dow = {}
        self.ic_out_by_dow = {}
        self.avg_ar = 0
        self.avg_ap = 0
        self.avg_ic_in = 0
        self.avg_ic_out = 0
    
    def fit(self, category_df: pd.DataFrame):
        df = category_df[category_df['is_banking_day']].copy()
        
        # Overall averages
        self.avg_ar = df['AR'].mean()
        self.avg_ap = df[df['AP'] > 0]['AP'].mean() if (df['AP'] > 0).any() else 0
        self.avg_ic_in = df[df['IC_IN'] > 0]['IC_IN'].mean() if (df['IC_IN'] > 0).any() else 0
        self.avg_ic_out = df[df['IC_OUT'] > 0]['IC_OUT'].mean() if (df['IC_OUT'] > 0).any() else 0
        
        # Day-of-week patterns
        for dow in range(5):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) > 0:
                self.ar_by_dow[dow] = dow_df['AR'].mean()
                self.ap_by_dow[dow] = dow_df['AP'].mean()
                self.ic_in_by_dow[dow] = dow_df['IC_IN'].mean()
                self.ic_out_by_dow[dow] = dow_df['IC_OUT'].mean()
        
        return self
    
    def get_ar(self, dow: int) -> float:
        return self.ar_by_dow.get(dow, self.avg_ar)
    
    def get_ap(self, dow: int) -> float:
        # AP is Friday (dow=4)
        return self.ap_by_dow.get(dow, 0)
    
    def get_ic_in(self, dow: int) -> float:
        # IC_IN is Monday (dow=0)
        return self.ic_in_by_dow.get(dow, 0)
    
    def get_ic_out(self, dow: int) -> float:
        # IC_OUT is Wednesday (dow=2)
        return self.ic_out_by_dow.get(dow, 0)


class ProphetCashForecaster:
    """
    Hybrid forecaster:
    - Prophet for pattern-based flows (AR, AP, IC)
    - Last-actual for cyclical events (Payroll, Debt, Tax)
    - User input for discretionary (CAPEX)
    """
    
    CYCLICAL_EVENTS = ['PAYROLL', 'DEBT', 'TAX', 'INV_INC']
    PATTERN_BASED = ['AR', 'AP', 'IC_IN', 'IC_OUT']
    USER_INPUT = ['CAPEX']
    
    def __init__(self):
        self.inflow_model = None
        self.outflow_model = None
        self.is_fitted = False
        self.training_data = None
        self.last_actual_date = None
        self.last_actual_closing_balance = None
        self.cyclical_tracker = CyclicalEventTracker()
        self.pattern_props = PatternBasedProportions()
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
        
        if category_df is not None:
            category_df = category_df.copy()
            category_df['is_banking_day'] = category_df['date'].apply(
                lambda x: USBankingCalendar.is_banking_day(x, self.holidays)
            )
            category_df['day_of_week'] = category_df['date'].dt.dayofweek
            
            # Fit cyclical event tracker
            self.cyclical_tracker.fit(category_df, self.holidays)
            
            # Fit pattern-based proportions
            self.pattern_props.fit(category_df)
            
            # Calculate pattern-based inflows/outflows (excluding cyclical)
            cat_banking = category_df[category_df['is_banking_day']].copy()
            
            # Inflows: AR + IC_IN (INV_INC is cyclical)
            cat_banking['inflow_pattern'] = cat_banking['AR'] + cat_banking['IC_IN']
            
            # Outflows: AP + IC_OUT (Payroll, Debt, Tax are cyclical; CAPEX is user input)
            cat_banking['outflow_pattern'] = cat_banking['AP'] + cat_banking['IC_OUT']
            
            # Merge back to banking_df
            banking_df = banking_df.merge(
                cat_banking[['date', 'inflow_pattern', 'outflow_pattern']],
                on='date', how='left'
            )
            
            inflow_col = 'inflow_pattern'
            outflow_col = 'outflow_pattern'
        else:
            inflow_col = 'inflow'
            outflow_col = 'outflow'
        
        print(f"\n{'='*60}")
        print("TRAINING PROPHET v5 (Cyclical Events Use Last Actual)")
        print(f"{'='*60}")
        print(f"Training Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"Banking Days: {len(banking_df)}")
        print(f"T0 Balance: ${self.last_actual_closing_balance:,.0f}")
        
        # Print cyclical event summary
        cyclical_summary = self.cyclical_tracker.summary()
        print(f"\nCyclical Events (Last Actual Values):")
        print(f"  Payroll:    ${cyclical_summary['last_payroll']:>12,.0f} (bi-monthly)")
        print(f"  Debt:       ${cyclical_summary['last_debt']:>12,.0f} (monthly)")
        print(f"  Tax:        ${cyclical_summary['last_tax']:>12,.0f} (quarterly)")
        print(f"  Inv Income: ${cyclical_summary['last_inv_income']:>12,.0f} (monthly)")
        
        # Train Prophet on pattern-based flows only
        print(f"\nTraining Prophet on pattern-based flows (AR, AP, IC)...")
        
        inflow_df = pd.DataFrame({'ds': banking_df['date'], 'y': banking_df[inflow_col]})
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
        self.outflow_model.fit(outflow_df)
        
        self.is_fitted = True
        print("✅ Models trained")
        
        return self
    
    def predict(self, horizon: str = None, capex_schedule: Dict[str, float] = None) -> Dict[str, pd.DataFrame]:
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
            
            # Prophet predictions for pattern-based flows
            inflow_pattern_pred = self.inflow_model.predict(future_df)
            outflow_pattern_pred = self.outflow_model.predict(future_df)
            
            rows = []
            current_balance = self.last_actual_closing_balance
            
            for i in range(num_days):
                date = future_dates[i]
                date_str = date.strftime('%Y-%m-%d')
                dow = date.dayofweek
                
                # Pattern-based predictions (from Prophet)
                pattern_inflow = max(inflow_pattern_pred['yhat'].iloc[i], 0)
                pattern_outflow = max(outflow_pattern_pred['yhat'].iloc[i], 0)
                
                # Allocate pattern-based flows to categories
                ar = self.pattern_props.get_ar(dow)
                ic_in = self.pattern_props.get_ic_in(dow)
                ap = self.pattern_props.get_ap(dow)
                ic_out = self.pattern_props.get_ic_out(dow)
                
                # Scale to match Prophet prediction
                pattern_inflow_sum = ar + ic_in
                if pattern_inflow_sum > 0:
                    scale = pattern_inflow / pattern_inflow_sum
                    ar *= scale
                    ic_in *= scale
                
                pattern_outflow_sum = ap + ic_out
                if pattern_outflow_sum > 0:
                    scale = pattern_outflow / pattern_outflow_sum
                    ap *= scale
                    ic_out *= scale
                
                # Cyclical events (last actual values on scheduled dates)
                payroll = self.cyclical_tracker.get_scheduled_amount(date, 'PAYROLL', self.holidays)
                debt = self.cyclical_tracker.get_scheduled_amount(date, 'DEBT', self.holidays)
                tax = self.cyclical_tracker.get_scheduled_amount(date, 'TAX', self.holidays)
                inv_inc = self.cyclical_tracker.get_scheduled_amount(date, 'INV_INC', self.holidays)
                
                # User input (CAPEX)
                capex = 0
                if capex_schedule and date_str in capex_schedule:
                    capex = capex_schedule[date_str]
                
                # Totals
                total_inflow = ar + ic_in + inv_inc
                total_outflow_ex_capex = ap + ic_out + payroll + debt + tax
                total_outflow = total_outflow_ex_capex + capex
                
                opening = current_balance
                net = total_inflow - total_outflow
                closing = opening + net
                
                row = {
                    'date': date,
                    'horizon_day': i + 1,
                    'day_of_week': dow,
                    'day_name': date.day_name(),
                    'opening_balance': opening,
                    # Inflows
                    'AR': ar,
                    'INV_INC': inv_inc,
                    'IC_IN': ic_in,
                    'forecast_inflow': total_inflow,
                    # Outflows - Pattern based
                    'AP': ap,
                    'IC_OUT': ic_out,
                    # Outflows - Cyclical (last actual)
                    'PAYROLL': payroll,
                    'DEBT': debt,
                    'TAX': tax,
                    # Outflows - User input
                    'CAPEX': capex,
                    'forecast_outflow_ex_capex': total_outflow_ex_capex,
                    'forecast_outflow': total_outflow,
                    'forecast_net': net,
                    'closing_balance': closing,
                    'horizon': hz_name,
                }
                rows.append(row)
                current_balance = closing
            
            results[hz_name] = pd.DataFrame(rows)
        
        return results


class ForecastAnalyzer:
    """Analyze forecast accuracy."""
    
    def __init__(self):
        self.results = {}
    
    def analyze(self, forecasts: Dict[str, pd.DataFrame], actuals: pd.DataFrame, 
                category_actuals: pd.DataFrame = None) -> Dict[str, Any]:
        
        holidays = USBankingCalendar.get_us_holidays(
            actuals['date'].min().year, actuals['date'].max().year + 1
        )
        actuals = actuals.copy()
        actuals['is_banking_day'] = actuals['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, holidays)
        )
        actuals['day_of_week'] = actuals['date'].dt.dayofweek
        actuals_banking = actuals[actuals['is_banking_day']].copy()
        
        if category_actuals is not None:
            cat = category_actuals.copy()
            cat['is_banking_day'] = cat['date'].apply(
                lambda x: USBankingCalendar.is_banking_day(x, holidays)
            )
            cat_banking = cat[cat['is_banking_day']].copy()
            cat_banking['outflow_ex_capex'] = (
                cat_banking['PAYROLL'] + cat_banking['AP'] + cat_banking['TAX'] + 
                cat_banking['DEBT'] + cat_banking['IC_OUT']
            )
            actuals_banking = actuals_banking.merge(
                cat_banking[['date', 'outflow_ex_capex']],
                on='date', how='left'
            )
        
        results = {}
        
        for horizon, forecast_df in forecasts.items():
            merged = forecast_df.merge(
                actuals_banking[['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week'] +
                               (['outflow_ex_capex'] if 'outflow_ex_capex' in actuals_banking.columns else [])],
                on='date', how='inner',
                suffixes=('_forecast', '_actual')
            )
            
            if len(merged) == 0:
                continue
            
            # Handle column naming
            if 'closing_balance_forecast' in merged.columns:
                merged['forecast_balance'] = merged['closing_balance_forecast']
                merged['actual_balance'] = merged['closing_balance_actual']
            else:
                merged['forecast_balance'] = merged['closing_balance']
                merged['actual_balance'] = merged['closing_balance']
            
            if 'day_of_week_actual' in merged.columns:
                merged['day_of_week'] = merged['day_of_week_actual']
            
            # Calculate errors
            merged['inflow_pct_error'] = np.abs(
                (merged['forecast_inflow'] - merged['inflow']) / merged['inflow'].replace(0, np.nan)
            ) * 100
            
            if 'outflow_ex_capex' in merged.columns:
                merged['outflow_pct_error'] = np.abs(
                    (merged['forecast_outflow_ex_capex'] - merged['outflow_ex_capex']) / 
                    merged['outflow_ex_capex'].replace(0, np.nan)
                ) * 100
            else:
                merged['outflow_pct_error'] = np.abs(
                    (merged['forecast_outflow'] - merged['outflow']) / merged['outflow'].replace(0, np.nan)
                ) * 100
            
            merged['balance_pct_error'] = np.abs(
                (merged['forecast_balance'] - merged['actual_balance']) / 
                merged['actual_balance'].replace(0, np.nan)
            ) * 100
            
            merged = merged.fillna(0)
            
            # MAPE by horizon day
            mape_by_horizon = merged.groupby('horizon_day').agg({
                'inflow_pct_error': 'mean',
                'outflow_pct_error': 'mean',
                'balance_pct_error': 'mean'
            }).reset_index()
            mape_by_horizon.columns = ['horizon_day', 'inflow_mape', 'outflow_mape', 'balance_mape']
            
            # MAPE by day of week
            mape_by_dow = merged.groupby('day_of_week').agg({
                'inflow_pct_error': 'mean',
                'outflow_pct_error': 'mean',
                'balance_pct_error': 'mean'
            }).reset_index()
            mape_by_dow.columns = ['day_of_week', 'inflow_mape', 'outflow_mape', 'balance_mape']
            mape_by_dow['day_name'] = mape_by_dow['day_of_week'].apply(
                lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][int(x)] if x < 5 else f'Day {int(x)}'
            )
            
            # Overall metrics
            inflow_mape = merged['inflow_pct_error'].mean()
            outflow_mape = merged['outflow_pct_error'].mean()
            balance_mape = merged['balance_pct_error'].mean()
            
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
                'balance_mape': balance_mape,
                'outflow_label': 'Outflow (ex-CAPEX)',
                'rating': rating,
                'samples': len(merged),
                'mape_by_horizon_day': mape_by_horizon,
                'mape_by_dow': mape_by_dow,
                'daily_errors': merged,
            }
        
        self.results = results
        return results


def run_backtest(df: pd.DataFrame, test_size: int = 90, category_df: pd.DataFrame = None):
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    train_cat = category_df.iloc[:-test_size].copy() if category_df is not None else None
    test_cat = category_df.iloc[-test_size:].copy() if category_df is not None else None
    
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df, train_cat)
    forecasts = forecaster.predict()
    
    analyzer = ForecastAnalyzer()
    results = analyzer.analyze(forecasts, test_df, test_cat)
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    from data_simulator_v2 import generate_category_data
    
    print("="*70)
    print("Testing Prophet v5 (Cyclical Events = Last Actual)")
    print("="*70)
    
    data = generate_category_data(periods=400)
    results, forecaster, forecasts = run_backtest(
        data['daily_cash_position'], test_size=60, category_df=data['category_details']
    )
    
    print("\n" + "="*70)
    print("ACCURACY RESULTS")
    print("="*70)
    
    for horizon, metrics in results.items():
        print(f"\n{horizon}:")
        print(f"  Balance MAPE:  {metrics['balance_mape']:.2f}%  ({metrics['rating']})")
        print(f"  Inflow MAPE:   {metrics['inflow_mape']:.2f}%")
        print(f"  Outflow MAPE:  {metrics['outflow_mape']:.2f}%")
    
    print("\n" + "="*70)
    print("T+7 FORECAST SAMPLE")
    print("="*70)
    
    t7 = forecasts['T+7']
    print(t7[['date', 'day_name', 'AR', 'PAYROLL', 'AP', 'closing_balance']].to_string())
