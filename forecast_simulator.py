"""
Rolling Forecast Simulator
==========================
Simulates daily treasury operations with:
1. T+1 forecast (next banking day)
2. Accuracy trail (recent forecast vs actual)
3. Forward forecasts (T+7, T+30, T+90)

This shows how the system would work in production with daily actual updates.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

from config import TIME_HORIZONS, MAPE_THRESHOLDS
from models_prophet_v2 import ProphetCashForecaster, USBankingCalendar


# Update TIME_HORIZONS to include T+1
TIME_HORIZONS_WITH_T1 = {
    "T+1": {"days": 1, "model": "Prophet", "description": "Next Day Forecast"},
    "T+7": {"days": 7, "model": "Prophet", "description": "7-Day Forecast"},
    "T+30": {"days": 30, "model": "Prophet", "description": "30-Day Forecast"},
    "T+90": {"days": 90, "model": "Prophet", "description": "90-Day Forecast"},
}


class ForecastAccuracyTrail:
    """
    Tracks rolling forecast accuracy.
    Compares what we forecasted N days ago vs what actually happened.
    """
    
    def __init__(self, trail_length: int = 14):
        """
        Args:
            trail_length: Number of days to keep in accuracy trail
        """
        self.trail_length = trail_length
        self.trail = []  # List of {date, forecast, actual, error, mape}
    
    def add_observation(
        self, 
        date: datetime, 
        forecast_inflow: float,
        forecast_outflow: float,
        forecast_net: float,
        forecast_balance: float,
        actual_inflow: float,
        actual_outflow: float,
        actual_net: float,
        actual_balance: float
    ):
        """Add a new actual vs forecast observation."""
        
        # Calculate errors
        inflow_error = forecast_inflow - actual_inflow
        outflow_error = forecast_outflow - actual_outflow
        net_error = forecast_net - actual_net
        balance_error = forecast_balance - actual_balance
        
        # Calculate MAPE
        inflow_mape = abs(inflow_error / actual_inflow) * 100 if actual_inflow != 0 else 0
        outflow_mape = abs(outflow_error / actual_outflow) * 100 if actual_outflow != 0 else 0
        net_mape = abs(net_error / actual_net) * 100 if actual_net != 0 else 0
        balance_mape = abs(balance_error / actual_balance) * 100 if actual_balance != 0 else 0
        
        observation = {
            'date': date,
            'forecast_inflow': forecast_inflow,
            'actual_inflow': actual_inflow,
            'inflow_error': inflow_error,
            'inflow_mape': inflow_mape,
            'forecast_outflow': forecast_outflow,
            'actual_outflow': actual_outflow,
            'outflow_error': outflow_error,
            'outflow_mape': outflow_mape,
            'forecast_net': forecast_net,
            'actual_net': actual_net,
            'net_error': net_error,
            'net_mape': net_mape,
            'forecast_balance': forecast_balance,
            'actual_balance': actual_balance,
            'balance_error': balance_error,
            'balance_mape': balance_mape,
        }
        
        self.trail.append(observation)
        
        # Keep only last N observations
        if len(self.trail) > self.trail_length:
            self.trail = self.trail[-self.trail_length:]
    
    def get_trail_df(self) -> pd.DataFrame:
        """Get accuracy trail as DataFrame."""
        if not self.trail:
            return pd.DataFrame()
        return pd.DataFrame(self.trail)
    
    def get_rolling_mape(self) -> Dict[str, float]:
        """Get rolling average MAPE over trail period."""
        if not self.trail:
            return {}
        
        df = self.get_trail_df()
        return {
            'inflow_mape': df['inflow_mape'].mean(),
            'outflow_mape': df['outflow_mape'].mean(),
            'net_mape': df['net_mape'].mean(),
            'balance_mape': df['balance_mape'].mean(),
            'days': len(df),
        }
    
    def get_accuracy_rating(self) -> str:
        """Get accuracy rating based on rolling balance MAPE."""
        metrics = self.get_rolling_mape()
        if not metrics:
            return "N/A"
        
        mape = metrics.get('balance_mape', 100)
        
        # Use T+1 thresholds (strictest)
        if mape <= 3.0:
            return "Excellent"
        elif mape <= 5.0:
            return "Good"
        elif mape <= 10.0:
            return "Acceptable"
        else:
            return "Poor"


class DailyForecastSimulator:
    """
    Simulates daily treasury forecast operations.
    
    Each day:
    1. Receive actual data for T0
    2. Compare to yesterday's T+1 forecast
    3. Update accuracy trail
    4. Generate new forecasts (T+1, T+7, T+30, T+90)
    """
    
    def __init__(
        self, 
        historical_df: pd.DataFrame,
        min_training_days: int = 365
    ):
        """
        Initialize simulator.
        
        Args:
            historical_df: Full historical data
            min_training_days: Minimum days needed for training
        """
        self.historical_df = historical_df.copy().sort_values('date').reset_index(drop=True)
        self.min_training_days = min_training_days
        
        # Get banking calendar
        min_year = historical_df['date'].min().year
        max_year = historical_df['date'].max().year + 2
        self.holidays = USBankingCalendar.get_us_holidays(min_year, max_year)
        
        # Filter to banking days
        self.historical_df['is_banking_day'] = self.historical_df['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, self.holidays)
        )
        
        # State
        self.current_day_index = min_training_days
        self.forecaster = None
        self.accuracy_trail = ForecastAccuracyTrail(trail_length=14)
        self.last_forecast = None
        self.simulation_log = []
    
    def get_current_date(self) -> datetime:
        """Get current simulation date (T0)."""
        return self.historical_df.iloc[self.current_day_index]['date']
    
    def get_training_data(self) -> pd.DataFrame:
        """Get all data up to and including current day."""
        return self.historical_df.iloc[:self.current_day_index + 1].copy()
    
    def get_actual_for_date(self, date: datetime) -> Dict:
        """Get actual cash flow data for a specific date."""
        row = self.historical_df[self.historical_df['date'] == date]
        if len(row) == 0:
            return None
        row = row.iloc[0]
        return {
            'date': row['date'],
            'inflow': row['inflow'],
            'outflow': row['outflow'],
            'net': row['net_cash_flow'],
            'opening_balance': row['opening_balance'],
            'closing_balance': row['closing_balance'],
        }
    
    def train_model(self):
        """Train/retrain the forecast model on current training data."""
        training_data = self.get_training_data()
        self.forecaster = ProphetCashForecaster()
        self.forecaster.fit(training_data)
    
    def generate_forecasts(self) -> Dict[str, pd.DataFrame]:
        """Generate forecasts for all horizons."""
        if self.forecaster is None:
            self.train_model()
        
        forecasts = {}
        
        for horizon, config in TIME_HORIZONS_WITH_T1.items():
            days = config['days']
            
            # Get next N banking days
            future_dates = USBankingCalendar.get_banking_days(
                self.get_current_date().to_pydatetime(),
                days
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Get Prophet predictions
            inflow_pred = self.forecaster.inflow_model.predict(future_df)
            outflow_pred = self.forecaster.outflow_model.predict(future_df)
            
            forecast_inflows = np.maximum(inflow_pred['yhat'].values, 0)
            forecast_outflows = np.maximum(outflow_pred['yhat'].values, 0)
            
            # Calculate balances using treasury logic
            opening_balances = []
            closing_balances = []
            current_balance = self.forecaster.last_actual_closing_balance
            
            for i in range(days):
                opening_balances.append(current_balance)
                closing = current_balance + forecast_inflows[i] - forecast_outflows[i]
                closing_balances.append(closing)
                current_balance = closing
            
            forecasts[horizon] = pd.DataFrame({
                'date': future_dates,
                'horizon_day': list(range(1, days + 1)),
                'day_name': future_dates.day_name(),
                'opening_balance': opening_balances,
                'forecast_inflow': forecast_inflows,
                'forecast_outflow': forecast_outflows,
                'forecast_net': forecast_inflows - forecast_outflows,
                'closing_balance': closing_balances,
                'horizon': horizon,
            })
        
        self.last_forecast = forecasts
        return forecasts
    
    def advance_day(self) -> Dict[str, Any]:
        """
        Advance simulation by one day.
        
        Returns:
            Dict with:
            - previous_date: Yesterday's date
            - current_date: Today's date (new T0)
            - actual: Today's actual data
            - forecast_vs_actual: Comparison of yesterday's T+1 forecast vs today's actual
            - new_forecasts: New forecasts from today
            - accuracy_trail: Updated accuracy trail
        """
        # Store previous state
        previous_date = self.get_current_date()
        previous_forecast = self.last_forecast
        
        # Advance to next day
        self.current_day_index += 1
        
        if self.current_day_index >= len(self.historical_df):
            return {"error": "No more data to simulate"}
        
        current_date = self.get_current_date()
        
        # Get today's actual
        actual = self.get_actual_for_date(current_date)
        
        # Compare yesterday's T+1 forecast to today's actual
        forecast_vs_actual = None
        if previous_forecast and 'T+1' in previous_forecast:
            t1_forecast = previous_forecast['T+1']
            if len(t1_forecast) > 0:
                t1_row = t1_forecast.iloc[0]
                
                forecast_vs_actual = {
                    'date': current_date,
                    'forecast_inflow': t1_row['forecast_inflow'],
                    'actual_inflow': actual['inflow'],
                    'forecast_outflow': t1_row['forecast_outflow'],
                    'actual_outflow': actual['outflow'],
                    'forecast_net': t1_row['forecast_net'],
                    'actual_net': actual['net'],
                    'forecast_balance': t1_row['closing_balance'],
                    'actual_balance': actual['closing_balance'],
                }
                
                # Add to accuracy trail
                self.accuracy_trail.add_observation(
                    date=current_date,
                    forecast_inflow=t1_row['forecast_inflow'],
                    forecast_outflow=t1_row['forecast_outflow'],
                    forecast_net=t1_row['forecast_net'],
                    forecast_balance=t1_row['closing_balance'],
                    actual_inflow=actual['inflow'],
                    actual_outflow=actual['outflow'],
                    actual_net=actual['net'],
                    actual_balance=actual['closing_balance'],
                )
        
        # Retrain model with new data (in production, might do this less frequently)
        self.train_model()
        
        # Generate new forecasts
        new_forecasts = self.generate_forecasts()
        
        # Build result
        result = {
            'previous_date': previous_date,
            'current_date': current_date,
            'actual': actual,
            'forecast_vs_actual': forecast_vs_actual,
            'new_forecasts': new_forecasts,
            'accuracy_trail': self.accuracy_trail.get_trail_df(),
            'rolling_mape': self.accuracy_trail.get_rolling_mape(),
            'accuracy_rating': self.accuracy_trail.get_accuracy_rating(),
        }
        
        # Log
        self.simulation_log.append(result)
        
        return result
    
    def run_simulation(self, num_days: int = 14, verbose: bool = True) -> List[Dict]:
        """
        Run simulation for N days.
        
        Args:
            num_days: Number of days to simulate
            verbose: Print progress
        
        Returns:
            List of daily results
        """
        if verbose:
            print("\n" + "="*70)
            print("DAILY FORECAST SIMULATION")
            print("="*70)
            print(f"Starting from: {self.get_current_date().strftime('%Y-%m-%d')}")
            print(f"Simulating: {num_days} days")
        
        # Initial training and forecast
        if verbose:
            print("\n[Initial] Training model and generating first forecast...")
        self.train_model()
        self.generate_forecasts()
        
        results = []
        
        for day in range(num_days):
            result = self.advance_day()
            
            if "error" in result:
                print(f"  Simulation stopped: {result['error']}")
                break
            
            results.append(result)
            
            if verbose:
                self._print_day_summary(day + 1, result)
        
        if verbose:
            self._print_final_summary(results)
        
        return results
    
    def _print_day_summary(self, day_num: int, result: Dict):
        """Print summary for a single simulation day."""
        print(f"\n{'─'*70}")
        print(f"DAY {day_num}: {result['current_date'].strftime('%Y-%m-%d')} ({result['current_date'].strftime('%A')})")
        print(f"{'─'*70}")
        
        # Actual
        actual = result['actual']
        print(f"\n  TODAY'S ACTUAL (T0):")
        print(f"    Inflow:   ${actual['inflow']:>12,.0f}")
        print(f"    Outflow:  ${actual['outflow']:>12,.0f}")
        print(f"    Net:      ${actual['net']:>12,.0f}")
        print(f"    Balance:  ${actual['closing_balance']:>12,.0f}")
        
        # Forecast vs Actual
        fva = result.get('forecast_vs_actual')
        if fva:
            print(f"\n  YESTERDAY'S T+1 FORECAST vs TODAY'S ACTUAL:")
            
            inflow_err = fva['forecast_inflow'] - fva['actual_inflow']
            outflow_err = fva['forecast_outflow'] - fva['actual_outflow']
            balance_err = fva['forecast_balance'] - fva['actual_balance']
            balance_mape = abs(balance_err / fva['actual_balance']) * 100
            
            print(f"    {'':20} {'Forecast':>12} {'Actual':>12} {'Error':>12}")
            print(f"    {'Inflow':20} ${fva['forecast_inflow']:>11,.0f} ${fva['actual_inflow']:>11,.0f} ${inflow_err:>+11,.0f}")
            print(f"    {'Outflow':20} ${fva['forecast_outflow']:>11,.0f} ${fva['actual_outflow']:>11,.0f} ${outflow_err:>+11,.0f}")
            print(f"    {'Closing Balance':20} ${fva['forecast_balance']:>11,.0f} ${fva['actual_balance']:>11,.0f} ${balance_err:>+11,.0f}")
            print(f"    Balance MAPE: {balance_mape:.2f}%")
        
        # Tomorrow's forecast (T+1)
        t1 = result['new_forecasts'].get('T+1')
        if t1 is not None and len(t1) > 0:
            t1_row = t1.iloc[0]
            print(f"\n  TOMORROW'S FORECAST (T+1): {t1_row['date'].strftime('%Y-%m-%d')} ({t1_row['day_name']})")
            print(f"    Inflow:   ${t1_row['forecast_inflow']:>12,.0f}")
            print(f"    Outflow:  ${t1_row['forecast_outflow']:>12,.0f}")
            print(f"    Net:      ${t1_row['forecast_net']:>12,.0f}")
            print(f"    Balance:  ${t1_row['closing_balance']:>12,.0f}")
        
        # Rolling accuracy
        rolling = result.get('rolling_mape', {})
        if rolling:
            print(f"\n  ROLLING ACCURACY (Last {rolling.get('days', 0)} days):")
            print(f"    Balance MAPE: {rolling.get('balance_mape', 0):.2f}%")
            print(f"    Rating: {result.get('accuracy_rating', 'N/A')}")
    
    def _print_final_summary(self, results: List[Dict]):
        """Print final simulation summary."""
        print("\n" + "="*70)
        print("SIMULATION SUMMARY")
        print("="*70)
        
        if not results:
            print("No results to summarize.")
            return
        
        # Get final accuracy trail
        final_trail = results[-1].get('accuracy_trail')
        if final_trail is not None and len(final_trail) > 0:
            print(f"\n  ACCURACY TRAIL ({len(final_trail)} days):")
            print(f"  {'Date':<12} {'Forecast':>12} {'Actual':>12} {'Error':>12} {'MAPE':>8}")
            print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")
            
            for _, row in final_trail.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d')
                print(f"  {date_str:<12} ${row['forecast_balance']:>11,.0f} ${row['actual_balance']:>11,.0f} ${row['balance_error']:>+11,.0f} {row['balance_mape']:>7.2f}%")
        
        # Final metrics
        final_rolling = results[-1].get('rolling_mape', {})
        print(f"\n  FINAL ROLLING MAPE:")
        print(f"    Inflow:   {final_rolling.get('inflow_mape', 0):>6.2f}%")
        print(f"    Outflow:  {final_rolling.get('outflow_mape', 0):>6.2f}%")
        print(f"    Net:      {final_rolling.get('net_mape', 0):>6.2f}%")
        print(f"    Balance:  {final_rolling.get('balance_mape', 0):>6.2f}%  ⭐")
        print(f"\n  OVERALL RATING: {results[-1].get('accuracy_rating', 'N/A')}")


def get_combined_view(
    actuals_df: pd.DataFrame,
    accuracy_trail: pd.DataFrame,
    forecasts: Dict[str, pd.DataFrame],
    horizon: str = 'T+7'
) -> pd.DataFrame:
    """
    Create combined view for charting:
    [Historical Actuals] + [Accuracy Trail with forecast overlay] + [Future Forecast]
    
    This is what treasury managers want to see!
    """
    result_parts = []
    
    # 1. Historical actuals (last 30 days before trail)
    if len(actuals_df) > 0:
        hist = actuals_df.tail(30)[['date', 'closing_balance']].copy()
        hist['type'] = 'actual'
        hist['forecast_balance'] = None
        result_parts.append(hist)
    
    # 2. Accuracy trail (actual + what we forecasted)
    if accuracy_trail is not None and len(accuracy_trail) > 0:
        trail = accuracy_trail[['date', 'actual_balance', 'forecast_balance']].copy()
        trail = trail.rename(columns={'actual_balance': 'closing_balance'})
        trail['type'] = 'trail'
        result_parts.append(trail)
    
    # 3. Future forecast
    if horizon in forecasts:
        fcast = forecasts[horizon][['date', 'closing_balance']].copy()
        fcast['type'] = 'forecast'
        fcast['forecast_balance'] = fcast['closing_balance']
        result_parts.append(fcast)
    
    if result_parts:
        return pd.concat(result_parts, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    from data_simulator_realistic import generate_sample_data
    
    print("="*70)
    print("ROLLING FORECAST SIMULATOR")
    print("="*70)
    
    # Generate data
    print("\nGenerating test data (2 years)...")
    data = generate_sample_data(periods=730)
    daily_cash = data['daily_cash_position']
    
    print(f"Data: {len(daily_cash)} days")
    print(f"Period: {daily_cash['date'].min().strftime('%Y-%m-%d')} to {daily_cash['date'].max().strftime('%Y-%m-%d')}")
    
    # Initialize simulator (start after 1 year of training data)
    simulator = DailyForecastSimulator(
        historical_df=daily_cash,
        min_training_days=365
    )
    
    # Run 14-day simulation
    results = simulator.run_simulation(num_days=14, verbose=True)
    
    # Show combined view concept
    if results:
        print("\n" + "="*70)
        print("COMBINED VIEW DATA STRUCTURE")
        print("="*70)
        print("""
    The dashboard will show:
    
    ┌──────────────────────────────────────────────────────────────────┐
    │  [Historical]  │  [Accuracy Trail]  │  [Future Forecast]         │
    │                │  Actual ━━━        │                            │
    │  Actual ━━━    │  Forecast ┅┅┅      │  Forecast ━━━              │
    │                │  (Shows recent     │  (T+7, T+30, T+90)         │
    │                │   accuracy)        │                            │
    └──────────────────────────────────────────────────────────────────┘
    
    This gives treasury managers:
    1. Historical context
    2. Recent forecast accuracy (confidence builder)
    3. Forward-looking forecasts
        """)
