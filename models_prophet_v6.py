"""Prophet Cash Forecasting v6 - Biweekly Payroll"""
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
    def get_us_holidays(start_year, end_year):
        holidays = []
        for year in range(start_year, end_year + 1):
            holidays.extend([datetime(year, 1, 1), datetime(year, 7, 4), datetime(year, 11, 11), datetime(year, 12, 25)])
            nov1 = datetime(year, 11, 1)
            holidays.append(nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21))
        return holidays
    
    @staticmethod
    def is_banking_day(date, holidays):
        return date.weekday() < 5 and date.date() not in [h.date() for h in holidays]
    
    @staticmethod
    def get_banking_days(start_date, num_days):
        holidays = USBankingCalendar.get_us_holidays(start_date.year, start_date.year + 2)
        banking_days, current = [], start_date + timedelta(days=1)
        while len(banking_days) < num_days:
            if USBankingCalendar.is_banking_day(pd.Timestamp(current), holidays):
                banking_days.append(current)
            current += timedelta(days=1)
        return pd.DatetimeIndex(banking_days)

class BiweeklyPayrollTracker:
    def __init__(self):
        self.last_payroll_date = None
        self.last_payroll_amount = 0
        self.payroll_day_of_week = None
        self.cycle_days = 14
    
    def fit(self, category_df, holidays):
        df = category_df.copy()
        df['is_banking'] = df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
        payroll_days = df[df['is_banking'] & (df['PAYROLL'] > 0)]
        if len(payroll_days) > 0:
            self.last_payroll_date = payroll_days['date'].iloc[-1]
            self.last_payroll_amount = payroll_days['PAYROLL'].iloc[-1]
            self.payroll_day_of_week = self.last_payroll_date.dayofweek
            if len(payroll_days) >= 2:
                self.cycle_days = round(payroll_days['date'].diff().dropna().dt.days.mean())
        return self
    
    def is_payroll_date(self, date):
        if self.last_payroll_date is None:
            return False
        days_since = (date - self.last_payroll_date).days
        return days_since > 0 and days_since % self.cycle_days == 0 and date.dayofweek == self.payroll_day_of_week
    
    def summary(self):
        day_name = ['Mon','Tue','Wed','Thu','Fri'][self.payroll_day_of_week] if self.payroll_day_of_week is not None else None
        return {'last_payroll_amount': self.last_payroll_amount or 0, 'payroll_day': day_name, 'cycle_days': self.cycle_days}

class CyclicalEventTracker:
    def __init__(self):
        self.payroll_tracker = BiweeklyPayrollTracker()
        self.last_debt = 0
        self.last_tax = 0
        self.last_inv_income = 0
    
    def fit(self, category_df, holidays):
        df = category_df.copy()
        df['is_banking'] = df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
        banking_df = df[df['is_banking']]
        self.payroll_tracker.fit(category_df, holidays)
        debt_days = banking_df[banking_df['DEBT'] > 0]
        if len(debt_days) > 0:
            self.last_debt = debt_days['DEBT'].iloc[-1]
        tax_days = banking_df[banking_df['TAX'] > 0]
        if len(tax_days) > 0:
            self.last_tax = tax_days['TAX'].iloc[-1]
        inv_days = banking_df[banking_df['INV_INC'] > 0]
        if len(inv_days) > 0:
            self.last_inv_income = inv_days['INV_INC'].iloc[-1]
        return self
    
    def get_payroll(self, date):
        if self.payroll_tracker.is_payroll_date(date):
            return self.payroll_tracker.last_payroll_amount
        return 0
    
    def get_debt(self, date):
        return self.last_debt if date.day == 1 else 0
    
    def get_tax(self, date):
        return self.last_tax if date.day == 15 and date.month in [4,6,9,12] else 0
    
    def get_inv_income(self, date):
        return self.last_inv_income if date.day == 1 else 0
    
    def summary(self):
        s = self.payroll_tracker.summary()
        s['last_debt'] = self.last_debt
        s['last_tax'] = self.last_tax
        s['last_inv_income'] = self.last_inv_income
        return s

class PatternBasedProportions:
    def __init__(self):
        self.ar_by_dow = {}
        self.ap_by_dow = {}
        self.ic_in_by_dow = {}
        self.ic_out_by_dow = {}
    
    def fit(self, category_df):
        df = category_df[category_df['is_banking_day']]
        for dow in range(5):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) > 0:
                self.ar_by_dow[dow] = dow_df['AR'].mean()
                self.ap_by_dow[dow] = dow_df['AP'].mean()
                self.ic_in_by_dow[dow] = dow_df['IC_IN'].mean()
                self.ic_out_by_dow[dow] = dow_df['IC_OUT'].mean()
        return self
    
    def get_ar(self, dow):
        return self.ar_by_dow.get(dow, 0)
    def get_ap(self, dow):
        return self.ap_by_dow.get(dow, 0)
    def get_ic_in(self, dow):
        return self.ic_in_by_dow.get(dow, 0)
    def get_ic_out(self, dow):
        return self.ic_out_by_dow.get(dow, 0)

class ProphetCashForecaster:
    def __init__(self):
        self.inflow_model = None
        self.outflow_model = None
        self.training_data = None
        self.last_actual_date = None
        self.last_actual_closing_balance = None
        self.is_fitted = False
        self.cyclical_tracker = CyclicalEventTracker()
        self.pattern_props = PatternBasedProportions()
        self.holidays = []
    
    def fit(self, df, category_df=None):
        df = df.copy().sort_values('date').reset_index(drop=True)
        self.holidays = USBankingCalendar.get_us_holidays(df['date'].min().year, df['date'].max().year + 2)
        df['is_banking_day'] = df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, self.holidays))
        banking_df = df[df['is_banking_day']].copy()
        self.training_data = df
        self.last_actual_date = df['date'].iloc[-1]
        self.last_actual_closing_balance = df['closing_balance'].iloc[-1]
        
        inflow_col, outflow_col = 'inflow', 'outflow'
        if category_df is not None:
            category_df = category_df.copy()
            category_df['is_banking_day'] = category_df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, self.holidays))
            category_df['day_of_week'] = category_df['date'].dt.dayofweek
            self.cyclical_tracker.fit(category_df, self.holidays)
            self.pattern_props.fit(category_df)
            cat_banking = category_df[category_df['is_banking_day']].copy()
            cat_banking['inflow_pattern'] = cat_banking['AR'] + cat_banking['IC_IN']
            cat_banking['outflow_pattern'] = cat_banking['AP'] + cat_banking['IC_OUT']
            banking_df = banking_df.merge(cat_banking[['date', 'inflow_pattern', 'outflow_pattern']], on='date', how='left')
            inflow_col, outflow_col = 'inflow_pattern', 'outflow_pattern'
        
        print(f"Training Prophet v6... T0: {self.last_actual_date.strftime('%Y-%m-%d')}, Balance: ${self.last_actual_closing_balance:,.0f}")
        s = self.cyclical_tracker.summary()
        print(f"Payroll: ${s['last_payroll_amount']:,.0f} every {s['cycle_days']} days on {s['payroll_day']}")
        
        self.inflow_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.inflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.inflow_model.fit(pd.DataFrame({'ds': banking_df['date'], 'y': banking_df[inflow_col]}))
        
        self.outflow_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        self.outflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.outflow_model.fit(pd.DataFrame({'ds': banking_df['date'], 'y': banking_df[outflow_col]}))
        
        self.is_fitted = True
        print("Models trained")
        return self
    
    def predict(self, horizon=None, capex_schedule=None):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        horizons = {horizon: TIME_HORIZONS[horizon]} if horizon else TIME_HORIZONS
        results = {}
        for hz_name, hz_config in horizons.items():
            num_days = hz_config['days']
            future_dates = USBankingCalendar.get_banking_days(self.last_actual_date.to_pydatetime(), num_days)
            future_df = pd.DataFrame({'ds': future_dates})
            inflow_pred = self.inflow_model.predict(future_df)
            outflow_pred = self.outflow_model.predict(future_df)
            rows = []
            current_balance = self.last_actual_closing_balance
            for i in range(num_days):
                date = future_dates[i]
                dow = date.dayofweek
                pattern_inflow = max(inflow_pred['yhat'].iloc[i], 0)
                pattern_outflow = max(outflow_pred['yhat'].iloc[i], 0)
                ar = self.pattern_props.get_ar(dow)
                ic_in = self.pattern_props.get_ic_in(dow)
                ap = self.pattern_props.get_ap(dow)
                ic_out = self.pattern_props.get_ic_out(dow)
                if ar + ic_in > 0:
                    scale = pattern_inflow / (ar + ic_in)
                    ar = ar * scale
                    ic_in = ic_in * scale
                if ap + ic_out > 0:
                    scale = pattern_outflow / (ap + ic_out)
                    ap = ap * scale
                    ic_out = ic_out * scale
                payroll = self.cyclical_tracker.get_payroll(date)
                debt = self.cyclical_tracker.get_debt(date)
                tax = self.cyclical_tracker.get_tax(date)
                inv_inc = self.cyclical_tracker.get_inv_income(date)
                capex = (capex_schedule or {}).get(date.strftime('%Y-%m-%d'), 0)
                total_inflow = ar + ic_in + inv_inc
                total_outflow_ex_capex = ap + ic_out + payroll + debt + tax
                total_outflow = total_outflow_ex_capex + capex
                opening = current_balance
                net = total_inflow - total_outflow
                closing = opening + net
                rows.append({
                    'date': date, 'horizon_day': i+1, 'day_of_week': dow, 'day_name': date.day_name(),
                    'opening_balance': opening, 'AR': ar, 'INV_INC': inv_inc, 'IC_IN': ic_in,
                    'forecast_inflow': total_inflow, 'AP': ap, 'IC_OUT': ic_out,
                    'PAYROLL': payroll, 'DEBT': debt, 'TAX': tax, 'CAPEX': capex,
                    'forecast_outflow_ex_capex': total_outflow_ex_capex,
                    'forecast_outflow': total_outflow, 'forecast_net': net,
                    'closing_balance': closing, 'horizon': hz_name, 'is_payroll_day': payroll > 0
                })
                current_balance = closing
            results[hz_name] = pd.DataFrame(rows)
        return results

class ForecastAnalyzer:
    def __init__(self):
        self.results = {}
    
    def analyze(self, forecasts, actuals, category_actuals=None):
        holidays = USBankingCalendar.get_us_holidays(actuals['date'].min().year, actuals['date'].max().year + 1)
        actuals = actuals.copy()
        actuals['is_banking_day'] = actuals['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
        actuals['day_of_week'] = actuals['date'].dt.dayofweek
        actuals_banking = actuals[actuals['is_banking_day']].copy()
        
        if category_actuals is not None:
            cat = category_actuals.copy()
            cat['is_banking_day'] = cat['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
            cat_banking = cat[cat['is_banking_day']].copy()
            cat_banking['outflow_ex_capex'] = cat_banking['PAYROLL'] + cat_banking['AP'] + cat_banking['TAX'] + cat_banking['DEBT'] + cat_banking['IC_OUT']
            actuals_banking = actuals_banking.merge(cat_banking[['date', 'outflow_ex_capex']], on='date', how='left')
        
        results = {}
        for horizon, forecast_df in forecasts.items():
            merge_cols = ['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week']
            if 'outflow_ex_capex' in actuals_banking.columns:
                merge_cols.append('outflow_ex_capex')
            merged = forecast_df.merge(actuals_banking[merge_cols], on='date', how='inner', suffixes=('_forecast', '_actual'))
            if len(merged) == 0:
                continue
            
            if 'closing_balance_forecast' in merged.columns:
                merged['forecast_balance'] = merged['closing_balance_forecast']
                merged['actual_balance'] = merged['closing_balance_actual']
            if 'day_of_week_actual' in merged.columns:
                merged['day_of_week'] = merged['day_of_week_actual']
            
            merged['inflow_pct_error'] = np.abs((merged['forecast_inflow'] - merged['inflow']) / merged['inflow'].replace(0, np.nan)) * 100
            if 'outflow_ex_capex' in merged.columns:
                merged['outflow_pct_error'] = np.abs((merged['forecast_outflow_ex_capex'] - merged['outflow_ex_capex']) / merged['outflow_ex_capex'].replace(0, np.nan)) * 100
            else:
                merged['outflow_pct_error'] = np.abs((merged['forecast_outflow'] - merged['outflow']) / merged['outflow'].replace(0, np.nan)) * 100
            merged['balance_pct_error'] = np.abs((merged['forecast_balance'] - merged['actual_balance']) / merged['actual_balance'].replace(0, np.nan)) * 100
            merged = merged.fillna(0)
            
            mape_by_horizon = merged.groupby('horizon_day').agg({
                'inflow_pct_error': 'mean', 'outflow_pct_error': 'mean', 'balance_pct_error': 'mean'
            }).reset_index()
            mape_by_horizon.columns = ['horizon_day', 'inflow_mape', 'outflow_mape', 'balance_mape']
            
            mape_by_dow = merged.groupby('day_of_week').agg({
                'inflow_pct_error': 'mean', 'outflow_pct_error': 'mean', 'balance_pct_error': 'mean'
            }).reset_index()
            mape_by_dow.columns = ['day_of_week', 'inflow_mape', 'outflow_mape', 'balance_mape']
            mape_by_dow['day_name'] = mape_by_dow['day_of_week'].apply(
                lambda x: ['Monday','Tuesday','Wednesday','Thursday','Friday'][int(x)] if x < 5 else f'Day {int(x)}'
            )
            
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
                'inflow_mape': merged['inflow_pct_error'].mean(),
                'outflow_mape': merged['outflow_pct_error'].mean(),
                'balance_mape': balance_mape,
                'outflow_label': 'Outflow (ex-CAPEX)',
                'rating': rating,
                'samples': len(merged),
                'mape_by_horizon_day': mape_by_horizon,
                'mape_by_dow': mape_by_dow,
                'daily_errors': merged
            }
        self.results = results
        return results

def run_backtest(df, test_size=90, category_df=None):
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
