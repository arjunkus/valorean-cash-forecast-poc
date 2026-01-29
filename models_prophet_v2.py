"""
Prophet Cash Forecasting v2
===========================
Improvements:
1. Banking days only (excludes weekends + US holidays)
2. MAPE tracked by day of horizon (Day 1, Day 2, ... Day N)
3. Shows forecast accuracy degradation over time

This aligns with how Treasury actually operates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

from config import TIME_HORIZONS, MAPE_THRESHOLDS, PROPHET_PARAMS


# =============================================================================
# US BANKING CALENDAR
# =============================================================================
class USBankingCalendar:
    """
    US Federal Reserve banking calendar.
    Excludes weekends and federal holidays.
    """
    
    @staticmethod
    def get_us_holidays(start_year: int, end_year: int) -> List[datetime]:
        """Generate US federal holidays for given year range."""
        holidays = []
        
        for year in range(start_year, end_year + 1):
            # Fixed holidays
            holidays.append(datetime(year, 1, 1))   # New Year's Day
            holidays.append(datetime(year, 7, 4))   # Independence Day
            holidays.append(datetime(year, 11, 11)) # Veterans Day
            holidays.append(datetime(year, 12, 25)) # Christmas Day
            
            # MLK Day - 3rd Monday of January
            holidays.append(USBankingCalendar._nth_weekday(year, 1, 0, 3))
            
            # Presidents Day - 3rd Monday of February
            holidays.append(USBankingCalendar._nth_weekday(year, 2, 0, 3))
            
            # Memorial Day - Last Monday of May
            holidays.append(USBankingCalendar._last_weekday(year, 5, 0))
            
            # Labor Day - 1st Monday of September
            holidays.append(USBankingCalendar._nth_weekday(year, 9, 0, 1))
            
            # Columbus Day - 2nd Monday of October
            holidays.append(USBankingCalendar._nth_weekday(year, 10, 0, 2))
            
            # Thanksgiving - 4th Thursday of November
            holidays.append(USBankingCalendar._nth_weekday(year, 11, 3, 4))
        
        return holidays
    
    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime:
        """Get nth occurrence of weekday in month (weekday: 0=Mon, 6=Sun)."""
        first_day = datetime(year, month, 1)
        first_weekday = first_day + timedelta(days=(weekday - first_day.weekday()) % 7)
        return first_weekday + timedelta(weeks=n-1)
    
    @staticmethod
    def _last_weekday(year: int, month: int, weekday: int) -> datetime:
        """Get last occurrence of weekday in month."""
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        days_since_weekday = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=days_since_weekday)
    
    @staticmethod
    def is_banking_day(date: pd.Timestamp, holidays: List[datetime]) -> bool:
        """Check if date is a US banking day."""
        # Weekend check
        if date.weekday() >= 5:
            return False
        # Holiday check
        if date.date() in [h.date() for h in holidays]:
            return False
        return True
    
    @staticmethod
    def get_banking_days(start_date: datetime, num_days: int) -> pd.DatetimeIndex:
        """Generate next N banking days from start date."""
        holidays = USBankingCalendar.get_us_holidays(
            start_date.year, 
            start_date.year + 2
        )
        
        banking_days = []
        current = start_date + timedelta(days=1)
        
        while len(banking_days) < num_days:
            if USBankingCalendar.is_banking_day(pd.Timestamp(current), holidays):
                banking_days.append(current)
            current += timedelta(days=1)
        
        return pd.DatetimeIndex(banking_days)


# =============================================================================
# PROPHET FORECASTER (Banking Days Only)
# =============================================================================
class ProphetCashForecaster:
    """
    Cash flow forecaster using Prophet.
    Forecasts only on US banking days.
    """
    
    def __init__(self):
        self.inflow_model = None
        self.outflow_model = None
        self.is_fitted = False
        self.training_data = None
        self.last_actual_date = None
        self.last_actual_closing_balance = None
        self.fit_diagnostics = {}
        self.holidays = []
    
    def fit(self, df: pd.DataFrame) -> 'ProphetCashForecaster':
        """Fit Prophet models on banking days only."""
        required_cols = ['date', 'inflow', 'outflow', 'opening_balance', 'closing_balance']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        df = df.copy().sort_values('date').reset_index(drop=True)
        
        # Generate holiday list
        min_year = df['date'].min().year
        max_year = df['date'].max().year + 2
        self.holidays = USBankingCalendar.get_us_holidays(min_year, max_year)
        
        # Filter to banking days only for training
        df['is_banking_day'] = df['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, self.holidays)
        )
        banking_df = df[df['is_banking_day']].copy()
        
        self.training_data = df  # Keep full data
        self.training_data_banking = banking_df  # Banking days only
        self.last_actual_date = df['date'].iloc[-1]
        self.last_actual_closing_balance = df['closing_balance'].iloc[-1]
        
        print("\n" + "="*60)
        print("TRAINING PROPHET MODELS (Banking Days Only)")
        print("="*60)
        print(f"Total Days: {len(df)}")
        print(f"Banking Days: {len(banking_df)} ({len(banking_df)/len(df)*100:.1f}%)")
        print(f"Training Period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        print(f"T0 Closing Balance: ${self.last_actual_closing_balance:,.0f}")
        
        # Prepare Prophet data (banking days only)
        inflow_df = pd.DataFrame({'ds': banking_df['date'], 'y': banking_df['inflow']})
        outflow_df = pd.DataFrame({'ds': banking_df['date'], 'y': banking_df['outflow']})
        
        # INFLOW MODEL
        print("\n[1/2] Training Inflow Model (Banking Days)...")
        self.inflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            interval_width=0.95,
        )
        self.inflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.inflow_model.add_seasonality(name='month_end', period=30.5, fourier_order=10)
        self.inflow_model.fit(inflow_df)
        
        self._quick_validation(self.inflow_model, inflow_df, 'Inflow')
        
        # OUTFLOW MODEL
        print("\n[2/2] Training Outflow Model (Banking Days)...")
        self.outflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            interval_width=0.95,
        )
        self.outflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.outflow_model.add_seasonality(name='biweekly', period=15.22, fourier_order=5)
        self.outflow_model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        self.outflow_model.fit(outflow_df)
        
        self._quick_validation(self.outflow_model, outflow_df, 'Outflow')
        
        self.is_fitted = True
        print("\n" + "="*60)
        print("✅ MODELS TRAINED SUCCESSFULLY")
        print("="*60)
        
        return self
    
    def _quick_validation(self, model: Prophet, df: pd.DataFrame, name: str) -> Dict:
        """Quick in-sample validation."""
        fitted = model.predict(df[['ds']])
        actual = df['y'].values
        predicted = fitted['yhat'].values
        
        mask = actual > 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = 0
        mae = np.mean(np.abs(actual - predicted))
        
        print(f"  {name} In-Sample: MAPE={mape:.1f}%, MAE=${mae:,.0f}")
        return {'mape': mape, 'mae': mae}
    
    def predict(self, horizon: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for banking days only.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon:
            horizons = {horizon: TIME_HORIZONS[horizon]}
        else:
            horizons = TIME_HORIZONS
        
        results = {}
        
        for hz_name, hz_config in horizons.items():
            num_banking_days = hz_config['days']
            
            # Get next N banking days
            future_dates = USBankingCalendar.get_banking_days(
                self.last_actual_date.to_pydatetime(),
                num_banking_days
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Prophet forecasts
            inflow_pred = self.inflow_model.predict(future_df)
            outflow_pred = self.outflow_model.predict(future_df)
            
            forecast_inflows = np.maximum(inflow_pred['yhat'].values, 0)
            forecast_outflows = np.maximum(outflow_pred['yhat'].values, 0)
            
            inflow_lower = np.maximum(inflow_pred['yhat_lower'].values, 0)
            inflow_upper = np.maximum(inflow_pred['yhat_upper'].values, 0)
            outflow_lower = np.maximum(outflow_pred['yhat_lower'].values, 0)
            outflow_upper = np.maximum(outflow_pred['yhat_upper'].values, 0)
            
            # TREASURY LOGIC: Calculate closing balances
            opening_balances = []
            closing_balances = []
            
            current_balance = self.last_actual_closing_balance
            
            for i in range(num_banking_days):
                opening_balances.append(current_balance)
                closing = current_balance + forecast_inflows[i] - forecast_outflows[i]
                closing_balances.append(closing)
                current_balance = closing
            
            # Build result with HORIZON DAY tracking
            result = pd.DataFrame({
                'date': future_dates,
                'horizon_day': list(range(1, num_banking_days + 1)),  # Day 1, Day 2, etc.
                'day_of_week': future_dates.dayofweek,
                'day_name': future_dates.day_name(),
                'day_of_month': future_dates.day,
                'opening_balance': opening_balances,
                'forecast_inflow': forecast_inflows,
                'forecast_outflow': forecast_outflows,
                'forecast_net': forecast_inflows - forecast_outflows,
                'closing_balance': closing_balances,
                'inflow_lower': inflow_lower,
                'inflow_upper': inflow_upper,
                'outflow_lower': outflow_lower,
                'outflow_upper': outflow_upper,
                'net_lower': inflow_lower - outflow_upper,
                'net_upper': inflow_upper - outflow_lower,
                'horizon': hz_name,
                'model': 'Prophet',
                # Compatibility columns
                'forecast': forecast_inflows - forecast_outflows,
                'lower_bound': inflow_lower - outflow_upper,
                'upper_bound': inflow_upper - outflow_lower,
            })
            
            results[hz_name] = result
        
        return results
    
    def get_forecast_summary(self, forecasts: Dict[str, pd.DataFrame] = None) -> pd.DataFrame:
        """Generate summary statistics for each horizon."""
        if forecasts is None:
            forecasts = self.predict()
        
        summaries = []
        for horizon, df in forecasts.items():
            summaries.append({
                'Horizon': horizon,
                'Banking Days': len(df),
                'Calendar Days': (df['date'].max() - df['date'].min()).days + 1,
                'T0 Balance': f"${self.last_actual_closing_balance:,.0f}",
                'Final Balance': f"${df['closing_balance'].iloc[-1]:,.0f}",
                'Total Inflows': f"${df['forecast_inflow'].sum():,.0f}",
                'Total Outflows': f"${df['forecast_outflow'].sum():,.0f}",
                'Net Change': f"${df['forecast_net'].sum():,.0f}",
            })
        return pd.DataFrame(summaries)


# =============================================================================
# FORECAST ANALYZER (By Horizon Day)
# =============================================================================
class ForecastAnalyzer:
    """
    Analyze forecast accuracy by HORIZON DAY (Day 1, Day 2, ... Day N).
    This shows how accuracy degrades over time - key for treasury managers.
    """
    
    def __init__(self):
        self.results = {}
    
    def analyze(self, forecasts: Dict[str, pd.DataFrame], actuals: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze forecast accuracy by horizon day.
        
        Key Insight: Day 1 forecasts should be more accurate than Day 30 forecasts.
        """
        # Filter actuals to banking days only
        holidays = USBankingCalendar.get_us_holidays(
            actuals['date'].min().year,
            actuals['date'].max().year + 1
        )
        actuals['is_banking_day'] = actuals['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, holidays)
        )
        actuals_banking = actuals[actuals['is_banking_day']].copy()
        
        results = {}
        
        for horizon, forecast_df in forecasts.items():
            # Rename columns for merge
            forecast_renamed = forecast_df.rename(columns={
                'closing_balance': 'forecast_closing_balance',
                'opening_balance': 'forecast_opening_balance'
            })
            
            # Merge on date
            merged = forecast_renamed.merge(
                actuals_banking[['date', 'inflow', 'outflow', 'net_cash_flow', 
                                'opening_balance', 'closing_balance']],
                on='date',
                how='inner'
            )
            
            if len(merged) == 0:
                continue
            
            # Calculate errors
            merged['inflow_error'] = merged['forecast_inflow'] - merged['inflow']
            merged['outflow_error'] = merged['forecast_outflow'] - merged['outflow']
            merged['net_error'] = merged['forecast_net'] - merged['net_cash_flow']
            merged['balance_error'] = merged['forecast_closing_balance'] - merged['closing_balance']
            
            # Percentage errors
            merged['inflow_pct_error'] = np.abs(merged['inflow_error'] / merged['inflow'].replace(0, np.nan)) * 100
            merged['outflow_pct_error'] = np.abs(merged['outflow_error'] / merged['outflow'].replace(0, np.nan)) * 100
            merged['net_pct_error'] = np.abs(merged['net_error'] / merged['net_cash_flow'].replace(0, np.nan)) * 100
            merged['balance_pct_error'] = np.abs(merged['balance_error'] / merged['closing_balance']) * 100
            
            # =========================================================
            # MAPE BY HORIZON DAY (Day 1, Day 2, Day 3, etc.)
            # =========================================================
            mape_by_horizon_day = merged.groupby('horizon_day').agg({
                'inflow_pct_error': 'mean',
                'outflow_pct_error': 'mean',
                'net_pct_error': 'mean',
                'balance_pct_error': 'mean',
                'balance_error': ['mean', 'std'],
            }).round(2)
            
            # Flatten column names
            mape_by_horizon_day.columns = [
                'inflow_mape', 'outflow_mape', 'net_mape', 'balance_mape',
                'balance_error_mean', 'balance_error_std'
            ]
            mape_by_horizon_day = mape_by_horizon_day.reset_index()
            
            # Overall metrics
            inflow_mape = merged['inflow_pct_error'].mean()
            outflow_mape = merged.loc[merged['outflow'] > 0, 'outflow_pct_error'].mean()
            net_mape = merged.loc[merged['net_cash_flow'] != 0, 'net_pct_error'].mean()
            balance_mape = merged['balance_pct_error'].mean()
            
            # Rating based on balance MAPE
            thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
            if balance_mape <= thresholds["excellent"]:
                rating = "Excellent"
            elif balance_mape <= thresholds["good"]:
                rating = "Good"
            elif balance_mape <= thresholds["acceptable"]:
                rating = "Acceptable"
            else:
                rating = "Poor"
            
            # Store results
            results[horizon] = {
                # Overall metrics
                'inflow_mape': inflow_mape,
                'outflow_mape': outflow_mape if not np.isnan(outflow_mape) else 0,
                'net_mape': net_mape if not np.isnan(net_mape) else 0,
                'balance_mape': balance_mape,
                'mape': balance_mape,
                'rating': rating,
                'samples': len(merged),
                
                # Absolute metrics
                'mae': np.mean(np.abs(merged['net_error'])),
                'balance_mae': np.mean(np.abs(merged['balance_error'])),
                
                # MAPE by horizon day (KEY METRIC)
                'mape_by_horizon_day': mape_by_horizon_day,
                
                # For dashboard compatibility
                'daily_analysis': mape_by_horizon_day.set_index('horizon_day')['balance_mape'].to_dict(),
                
                # Raw data
                'merged_data': merged,
            }
        
        # Dashboard compatibility
        results['daily_analysis'] = {
            hz: results[hz]['daily_analysis'] 
            for hz in results if hz != 'daily_analysis'
        }
        
        self.results = results
        return results
    
    def print_report(self):
        """Print detailed accuracy report by horizon day."""
        print("\n" + "="*70)
        print("FORECAST ACCURACY BY HORIZON DAY")
        print("="*70)
        print("(Shows how accuracy degrades as forecast horizon increases)")
        
        for horizon, metrics in self.results.items():
            if horizon == 'daily_analysis':
                continue
            
            print(f"\n{'─'*70}")
            print(f"HORIZON: {horizon} ({metrics['samples']} banking days analyzed)")
            print(f"{'─'*70}")
            
            print(f"\n  OVERALL METRICS:")
            print(f"    Inflow MAPE:    {metrics['inflow_mape']:6.1f}%")
            print(f"    Outflow MAPE:   {metrics['outflow_mape']:6.1f}%")
            print(f"    Net MAPE:       {metrics['net_mape']:6.1f}%")
            print(f"    Balance MAPE:   {metrics['balance_mape']:6.1f}%  ⭐ PRIMARY")
            print(f"    Rating:         {metrics['rating']}")
            
            print(f"\n  MAPE BY HORIZON DAY (Balance):")
            print(f"    {'Day':<5} {'MAPE':>8} {'Error Mean':>14} {'Error Std':>14}  Visual")
            print(f"    {'-'*5} {'-'*8} {'-'*14} {'-'*14}  {'-'*20}")
            
            mape_df = metrics['mape_by_horizon_day']
            
            for _, row in mape_df.iterrows():
                day = int(row['horizon_day'])
                mape = row['balance_mape']
                err_mean = row['balance_error_mean']
                err_std = row['balance_error_std']
                
                # Visual bar
                bar_len = int(min(mape, 10) * 2) if not np.isnan(mape) else 0
                bar = '█' * bar_len
                
                if not np.isnan(mape):
                    print(f"    {day:<5} {mape:>7.2f}% ${err_mean:>12,.0f} ${err_std:>12,.0f}  {bar}")
            
            # Show trend
            mape_values = mape_df['balance_mape'].dropna().values
            if len(mape_values) > 1:
                trend = "📈 Increasing" if mape_values[-1] > mape_values[0] else "📉 Decreasing"
                print(f"\n    Accuracy Trend: {trend} error over time")
                print(f"    Day 1 MAPE: {mape_values[0]:.2f}%")
                print(f"    Day {len(mape_values)} MAPE: {mape_values[-1]:.2f}%")


def run_backtest(df: pd.DataFrame, test_size: int = 90) -> Tuple[Dict, ProphetCashForecaster, Dict[str, pd.DataFrame]]:
    """Run backtest with banking days only."""
    print("\n" + "="*70)
    print("BACKTEST CONFIGURATION (Banking Days Only)")
    print("="*70)
    print(f"Total calendar days: {len(df)}")
    
    # Split data
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    print(f"Training: {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Testing:  {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"T0 Balance: ${train_df['closing_balance'].iloc[-1]:,.0f}")
    
    # Train
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df)
    
    # Predict
    forecasts = forecaster.predict()
    
    # Analyze
    analyzer = ForecastAnalyzer()
    results = analyzer.analyze(forecasts, test_df)
    analyzer.print_report()
    
    # Summary
    print("\n" + "="*70)
    print("FORECAST SUMMARY")
    print("="*70)
    summary = forecaster.get_forecast_summary(forecasts)
    print(summary.to_string(index=False))
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    from data_simulator_realistic import generate_sample_data
    
    print("="*70)
    print("PROPHET CASH FORECASTING v2")
    print("Banking Days Only + MAPE by Horizon Day")
    print("="*70)
    
    print("\nGenerating realistic cash flow data (2 years)...")
    data = generate_sample_data(periods=730)
    daily_cash = data['daily_cash_position']
    
    print(f"\nData Summary:")
    print(f"  Period: {daily_cash['date'].min().strftime('%Y-%m-%d')} to {daily_cash['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Total Days: {len(daily_cash)}")
    
    # Count banking days
    holidays = USBankingCalendar.get_us_holidays(
        daily_cash['date'].min().year,
        daily_cash['date'].max().year + 1
    )
    banking_days = sum(1 for d in daily_cash['date'] 
                       if USBankingCalendar.is_banking_day(d, holidays))
    print(f"  Banking Days: {banking_days} ({banking_days/len(daily_cash)*100:.1f}%)")
    print(f"  Starting Balance: ${daily_cash['opening_balance'].iloc[0]:,.0f}")
    print(f"  Ending Balance:   ${daily_cash['closing_balance'].iloc[-1]:,.0f}")
    
    # Run backtest
    results, forecaster, forecasts = run_backtest(daily_cash, test_size=90)
    
    # Show sample forecast
    print("\n" + "="*70)
    print("SAMPLE T+7 FORECAST (Banking Days Only)")
    print("="*70)
    
    t7 = forecasts['T+7'][['date', 'horizon_day', 'day_name', 'opening_balance', 
                           'forecast_inflow', 'forecast_outflow', 'forecast_net', 
                           'closing_balance']]
    
    print("\nTreasury Logic: Closing = Opening + Inflows - Outflows")
    print("Note: Weekends and US holidays excluded")
    print("-"*70)
    
    for _, row in t7.iterrows():
        print(f"Day {row['horizon_day']}: {row['date'].strftime('%Y-%m-%d')} ({row['day_name'][:3]}): "
              f"${row['opening_balance']:>12,.0f} + ${row['forecast_inflow']:>10,.0f} "
              f"- ${row['forecast_outflow']:>10,.0f} = ${row['closing_balance']:>12,.0f}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("""
    1. BANKING DAYS ONLY
       - No forecasts for weekends or US federal holidays
       - Aligns with actual treasury operations
    
    2. MAPE BY HORIZON DAY
       - Shows accuracy degradation: Day 1 vs Day 7 vs Day 30
       - Treasury managers can set confidence levels by horizon
       - Example: "Day 1-5 forecasts are 95% reliable, Day 20+ are 85%"
    
    3. BALANCE MAPE IS PRIMARY METRIC
       - What matters is cumulative cash position accuracy
       - Component MAPE (inflow/outflow) is secondary
    """)
