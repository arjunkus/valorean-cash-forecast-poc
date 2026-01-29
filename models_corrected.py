"""
Corrected Cash Forecasting Models
==================================
Proper treasury logic:
  T1f = T0a + Forecast_Receipts - Forecast_Payments
  
Forecasts inflows and outflows separately, then calculates
closing balances iteratively.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from config import TIME_HORIZONS, MAPE_THRESHOLDS


class CashFlowForecaster:
    """
    Forecasts cash position using proper treasury logic:
    - Forecast receipts (inflows) separately
    - Forecast payments (outflows) separately  
    - Calculate closing balance: T1f = T0a + Receipts - Payments
    """
    
    def __init__(self):
        self.inflow_model = None
        self.outflow_model = None
        self.is_fitted = False
        self.training_data = None
        self.last_actual_balance = None
        
    def fit(self, df: pd.DataFrame, verbose: int = 0):
        """
        Fit Prophet models for inflows and outflows separately.
        """
        from prophet import Prophet
        
        self.training_data = df.copy().sort_values('date').reset_index(drop=True)
        self.last_actual_balance = df['cash_position'].iloc[-1]
        self.last_actual_date = df['date'].iloc[-1]
        
        # Prepare data for Prophet
        inflow_df = pd.DataFrame({
            'ds': df['date'],
            'y': df['inflow']
        })
        
        outflow_df = pd.DataFrame({
            'ds': df['date'],
            'y': df['outflow']
        })
        
        # Train Inflow Model
        print("Training Inflow Forecast Model...")
        self.inflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.inflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.inflow_model.fit(inflow_df)
        
        # Train Outflow Model
        print("Training Outflow Forecast Model...")
        self.outflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.outflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.outflow_model.fit(outflow_df)
        
        self.is_fitted = True
        print("Models trained successfully!")
        
        return self
    
    def predict(self, horizon: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for specified horizon or all horizons.
        
        Returns DataFrame with:
        - date
        - forecast_inflow (Tfr)
        - forecast_outflow (Tfp)
        - forecast_net (Tfr - Tfp)
        - opening_balance
        - closing_balance (T0a + Tfr - Tfp)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon:
            horizons = {horizon: TIME_HORIZONS[horizon]}
        else:
            horizons = TIME_HORIZONS
        
        results = {}
        
        for hz_name, hz_config in horizons.items():
            days = hz_config['days']
            
            # Create future dates
            future_dates = pd.date_range(
                start=self.last_actual_date + timedelta(days=1),
                periods=days,
                freq='D'
            )
            future_df = pd.DataFrame({'ds': future_dates})
            
            # Forecast inflows and outflows
            inflow_forecast = self.inflow_model.predict(future_df)
            outflow_forecast = self.outflow_model.predict(future_df)
            
            # Build forecast DataFrame
            forecast_df = pd.DataFrame({
                'date': future_dates,
                'forecast_inflow': inflow_forecast['yhat'].values,
                'forecast_inflow_lower': inflow_forecast['yhat_lower'].values,
                'forecast_inflow_upper': inflow_forecast['yhat_upper'].values,
                'forecast_outflow': outflow_forecast['yhat'].values,
                'forecast_outflow_lower': outflow_forecast['yhat_lower'].values,
                'forecast_outflow_upper': outflow_forecast['yhat_upper'].values,
            })
            
            # Ensure no negative forecasts
            forecast_df['forecast_inflow'] = forecast_df['forecast_inflow'].clip(lower=0)
            forecast_df['forecast_outflow'] = forecast_df['forecast_outflow'].clip(lower=0)
            
            # Calculate net cash flow
            forecast_df['forecast_net'] = forecast_df['forecast_inflow'] - forecast_df['forecast_outflow']
            
            # Calculate closing balances iteratively
            # T1f = T0a + Tfr - Tfp
            opening_balances = []
            closing_balances = []
            
            current_balance = self.last_actual_balance
            
            for i in range(len(forecast_df)):
                opening_balances.append(current_balance)
                
                # Closing = Opening + Receipts - Payments
                closing = current_balance + forecast_df.iloc[i]['forecast_inflow'] - forecast_df.iloc[i]['forecast_outflow']
                closing_balances.append(closing)
                
                # Next day's opening = Today's closing
                current_balance = closing
            
            forecast_df['opening_balance'] = opening_balances
            forecast_df['closing_balance'] = closing_balances
            
            # Calculate confidence bounds for closing balance
            # Using cumulative uncertainty
            forecast_df['closing_balance_lower'] = self.last_actual_balance + (
                forecast_df['forecast_inflow_lower'].cumsum() - 
                forecast_df['forecast_outflow_upper'].cumsum()
            )
            forecast_df['closing_balance_upper'] = self.last_actual_balance + (
                forecast_df['forecast_inflow_upper'].cumsum() - 
                forecast_df['forecast_outflow_lower'].cumsum()
            )
            
            # Add metadata
            forecast_df['horizon'] = hz_name
            forecast_df['model'] = hz_config['model']
            
            # For compatibility with existing dashboard, add these columns
            forecast_df['forecast'] = forecast_df['forecast_net']
            forecast_df['lower_bound'] = forecast_df['forecast_inflow_lower'] - forecast_df['forecast_outflow_upper']
            forecast_df['upper_bound'] = forecast_df['forecast_inflow_upper'] - forecast_df['forecast_outflow_lower']
            
            results[hz_name] = forecast_df
        
        return results
    
    def get_forecast_with_actuals(self) -> pd.DataFrame:
        """
        Return combined DataFrame with actual history and forecasts
        for visualization of the complete cash position timeline.
        """
        # Actual data
        actuals = self.training_data[['date', 'inflow', 'outflow', 'net_cash_flow', 'cash_position']].copy()
        actuals['type'] = 'actual'
        actuals = actuals.rename(columns={
            'inflow': 'receipts',
            'outflow': 'payments', 
            'net_cash_flow': 'net',
            'cash_position': 'closing_balance'
        })
        
        return actuals
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Evaluate forecast accuracy against actuals.
        """
        results = {}
        forecasts = self.predict()
        
        for horizon, forecast_df in forecasts.items():
            # Merge with actuals
            merged = forecast_df.merge(
                test_df[['date', 'inflow', 'outflow', 'net_cash_flow', 'cash_position']],
                on='date',
                how='inner'
            )
            
            if len(merged) > 0:
                # MAPE for inflows
                inflow_mape = self._calculate_mape(
                    merged['inflow'].values,
                    merged['forecast_inflow'].values
                )
                
                # MAPE for outflows
                outflow_mape = self._calculate_mape(
                    merged['outflow'].values,
                    merged['forecast_outflow'].values
                )
                
                # MAPE for net cash flow
                net_mape = self._calculate_mape(
                    merged['net_cash_flow'].values,
                    merged['forecast_net'].values
                )
                
                # MAPE for closing balance
                balance_mape = self._calculate_mape(
                    merged['cash_position'].values,
                    merged['closing_balance'].values
                )
                
                results[horizon] = {
                    'inflow_mape': inflow_mape,
                    'outflow_mape': outflow_mape,
                    'net_mape': net_mape,
                    'balance_mape': balance_mape,
                    'mape': net_mape,  # For compatibility
                    'rating': self._get_rating(net_mape, horizon),
                    'samples': len(merged),
                    'mae': np.mean(np.abs(merged['net_cash_flow'] - merged['forecast_net'])),
                    'rmse': np.sqrt(np.mean((merged['net_cash_flow'] - merged['forecast_net'])**2))
                }
        
        return results
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        mask = actual != 0
        if not mask.any():
            return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _get_rating(self, mape: float, horizon: str) -> str:
        """Get rating based on MAPE thresholds."""
        thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
        if mape <= thresholds["excellent"]:
            return "Excellent"
        elif mape <= thresholds["good"]:
            return "Good"
        elif mape <= thresholds["acceptable"]:
            return "Acceptable"
        return "Poor"


def run_backtest(df: pd.DataFrame, test_size: int = 90, target_col: str = "net_cash_flow"):
    """
    Run backtest with proper train/test split.
    """
    # Split data
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # Train model
    forecaster = CashFlowForecaster()
    forecaster.fit(train_df)
    
    # Generate forecasts
    forecasts = forecaster.predict()
    
    # Evaluate
    results = forecaster.evaluate(test_df)
    
    # Add daily analysis for MAPE heatmap
    results["daily_analysis"] = {}
    for horizon, forecast_df in forecasts.items():
        merged = forecast_df.merge(
            test_df[['date', 'net_cash_flow']],
            on='date',
            how='inner'
        )
        if len(merged) > 0:
            merged['day_of_week'] = merged['date'].dt.dayofweek
            merged['pct_error'] = np.abs(
                (merged['net_cash_flow'] - merged['forecast_net']) / merged['net_cash_flow']
            ) * 100
            results["daily_analysis"][horizon] = merged.groupby('day_of_week')['pct_error'].mean().to_dict()
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    # Test the corrected model
    from data_simulator import generate_sample_data
    
    print("Generating test data...")
    data = generate_sample_data(periods=365)
    daily_cash = data['daily_cash_position']
    
    print(f"\nData shape: {daily_cash.shape}")
    print(f"Columns: {daily_cash.columns.tolist()}")
    
    print("\nRunning backtest...")
    results, forecaster, forecasts = run_backtest(daily_cash, test_size=30)
    
    print("\n=== FORECAST RESULTS ===")
    for horizon, forecast_df in forecasts.items():
        print(f"\n{horizon}:")
        print(f"  Last Actual Balance: ${forecaster.last_actual_balance:,.0f}")
        print(f"  Day 1 Forecast:")
        print(f"    Opening: ${forecast_df.iloc[0]['opening_balance']:,.0f}")
        print(f"    + Receipts: ${forecast_df.iloc[0]['forecast_inflow']:,.0f}")
        print(f"    - Payments: ${forecast_df.iloc[0]['forecast_outflow']:,.0f}")
        print(f"    = Closing: ${forecast_df.iloc[0]['closing_balance']:,.0f}")
        print(f"  Final Day Closing: ${forecast_df.iloc[-1]['closing_balance']:,.0f}")
    
    print("\n=== MAPE RESULTS ===")
    for horizon, metrics in results.items():
        if horizon != "daily_analysis":
            print(f"\n{horizon}:")
            print(f"  Inflow MAPE: {metrics.get('inflow_mape', 0):.1f}%")
            print(f"  Outflow MAPE: {metrics.get('outflow_mape', 0):.1f}%")
            print(f"  Net MAPE: {metrics.get('net_mape', 0):.1f}%")
            print(f"  Balance MAPE: {metrics.get('balance_mape', 0):.1f}%")
