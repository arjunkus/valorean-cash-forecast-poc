"""
Full Cash Forecasting Models
=============================
Proper treasury logic with all forecasting models:

- ARIMA: RT+7 (Real-Time + 7 days) - Best for short-term momentum
- Prophet: T+30 (30-day forecast) - Best for seasonality & trends  
- LSTM: T+90 (90-day forecast) - Best for complex patterns
- Ensemble: NT+365 (Annual) - Combines Prophet + LSTM

Treasury Logic:
  Closing Balance = Opening Balance + Forecast Receipts - Forecast Payments
  T1f = T0a + Tfr - Tfp
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any
import warnings
import os

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from config import TIME_HORIZONS, MAPE_THRESHOLDS


class BaseForecaster:
    """Base class for all forecasting models."""
    
    def __init__(self, name: str, horizon_name: str):
        self.name = name
        self.horizon_name = horizon_name
        self.horizon_days = TIME_HORIZONS[horizon_name]['days']
        self.inflow_model = None
        self.outflow_model = None
        self.is_fitted = False
        self.training_data = None
        self.last_actual_balance = None
        self.last_actual_date = None
    
    def _prepare_data(self, df: pd.DataFrame):
        """Prepare training data."""
        df = df.copy().sort_values('date').reset_index(drop=True)
        self.training_data = df
        self.last_actual_balance = df['closing_balance'].iloc[-1]
        self.last_actual_date = df['date'].iloc[-1]
        return df
    
    def _calculate_closing_balances(self, forecast_inflows: np.ndarray, 
                                     forecast_outflows: np.ndarray,
                                     dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Calculate closing balances using treasury logic:
        T1f = T0a + Tfr - Tfp
        """
        opening_balances = []
        closing_balances = []
        
        current_balance = self.last_actual_balance
        
        for i in range(len(forecast_inflows)):
            opening_balances.append(current_balance)
            closing = current_balance + forecast_inflows[i] - forecast_outflows[i]
            closing_balances.append(closing)
            current_balance = closing
        
        return pd.DataFrame({
            'date': dates,
            'opening_balance': opening_balances,
            'forecast_inflow': forecast_inflows,
            'forecast_outflow': forecast_outflows,
            'forecast_net': forecast_inflows - forecast_outflows,
            'closing_balance': closing_balances,
            'forecast': forecast_inflows - forecast_outflows,  # For compatibility
            'horizon': self.horizon_name,
            'model': self.name
        })
    
    def _calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        mask = actual != 0
        if not mask.any():
            return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def _get_rating(self, mape: float) -> str:
        """Get rating based on MAPE thresholds."""
        thresholds = MAPE_THRESHOLDS.get(self.horizon_name, MAPE_THRESHOLDS["T+30"])
        if mape <= thresholds["excellent"]:
            return "Excellent"
        elif mape <= thresholds["good"]:
            return "Good"
        elif mape <= thresholds["acceptable"]:
            return "Acceptable"
        return "Poor"


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA model for RT+7 (Real-Time + 7 days).
    Best for: Short-term forecasting, capturing recent momentum.
    """
    
    def __init__(self):
        super().__init__("ARIMA", "RT+7")
        self.order = (5, 1, 2)
        self.seasonal_order = (1, 1, 1, 7)  # Weekly seasonality
    
    def fit(self, df: pd.DataFrame):
        """Fit ARIMA models for inflows and outflows."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        df = self._prepare_data(df)
        
        print(f"  Training ARIMA for Inflows...")
        self.inflow_model = SARIMAX(
            df['inflow'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        print(f"  Training ARIMA for Outflows...")
        self.outflow_model = SARIMAX(
            df['outflow'],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        steps = steps or self.horizon_days
        
        # Forecast inflows and outflows
        inflow_forecast = self.inflow_model.get_forecast(steps=steps)
        outflow_forecast = self.outflow_model.get_forecast(steps=steps)
        
        forecast_inflows = np.maximum(inflow_forecast.predicted_mean.values, 0)
        forecast_outflows = np.maximum(outflow_forecast.predicted_mean.values, 0)
        
        # Generate dates
        dates = pd.date_range(
            start=self.last_actual_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Calculate closing balances
        result = self._calculate_closing_balances(forecast_inflows, forecast_outflows, dates)
        
        # Add confidence intervals
        inflow_ci = inflow_forecast.conf_int()
        outflow_ci = outflow_forecast.conf_int()
        
        result['inflow_lower'] = np.maximum(inflow_ci.iloc[:, 0].values, 0)
        result['inflow_upper'] = np.maximum(inflow_ci.iloc[:, 1].values, 0)
        result['outflow_lower'] = np.maximum(outflow_ci.iloc[:, 0].values, 0)
        result['outflow_upper'] = np.maximum(outflow_ci.iloc[:, 1].values, 0)
        result['lower_bound'] = result['inflow_lower'] - result['outflow_upper']
        result['upper_bound'] = result['inflow_upper'] - result['outflow_lower']
        
        return result


class ProphetForecaster(BaseForecaster):
    """
    Prophet model for T+30 (30-day forecast).
    Best for: Medium-term forecasting with strong seasonality.
    """
    
    def __init__(self, horizon_name: str = "T+30"):
        super().__init__("Prophet", horizon_name)
    
    def fit(self, df: pd.DataFrame):
        """Fit Prophet models for inflows and outflows."""
        from prophet import Prophet
        
        df = self._prepare_data(df)
        
        # Prepare Prophet DataFrames
        inflow_df = pd.DataFrame({'ds': df['date'], 'y': df['inflow']})
        outflow_df = pd.DataFrame({'ds': df['date'], 'y': df['outflow']})
        
        print(f"  Training Prophet for Inflows...")
        self.inflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.inflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.inflow_model.fit(inflow_df)
        
        print(f"  Training Prophet for Outflows...")
        self.outflow_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        self.outflow_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.outflow_model.add_seasonality(name='biweekly', period=15.25, fourier_order=3)  # For payroll
        self.outflow_model.fit(outflow_df)
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        steps = steps or self.horizon_days
        
        # Generate future dates
        dates = pd.date_range(
            start=self.last_actual_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        future = pd.DataFrame({'ds': dates})
        
        # Forecast
        inflow_pred = self.inflow_model.predict(future)
        outflow_pred = self.outflow_model.predict(future)
        
        forecast_inflows = np.maximum(inflow_pred['yhat'].values, 0)
        forecast_outflows = np.maximum(outflow_pred['yhat'].values, 0)
        
        # Calculate closing balances
        result = self._calculate_closing_balances(forecast_inflows, forecast_outflows, dates)
        
        # Add confidence intervals
        result['inflow_lower'] = np.maximum(inflow_pred['yhat_lower'].values, 0)
        result['inflow_upper'] = np.maximum(inflow_pred['yhat_upper'].values, 0)
        result['outflow_lower'] = np.maximum(outflow_pred['yhat_lower'].values, 0)
        result['outflow_upper'] = np.maximum(outflow_pred['yhat_upper'].values, 0)
        result['lower_bound'] = result['inflow_lower'] - result['outflow_upper']
        result['upper_bound'] = result['inflow_upper'] - result['outflow_lower']
        result['trend'] = inflow_pred['trend'].values - outflow_pred['trend'].values
        
        return result


class LSTMForecaster(BaseForecaster):
    """
    LSTM model for T+90 (90-day forecast).
    Best for: Longer-term forecasting, capturing complex patterns.
    
    Optimized for POC (CPU) with reduced epochs.
    For production, increase epochs and use GPU.
    """
    
    def __init__(self):
        super().__init__("LSTM", "T+90")
        # POC-optimized parameters (fast on CPU)
        self.sequence_length = 14
        self.n_units = 32
        self.epochs = 20  # Increase to 100+ for production with GPU
        self.batch_size = 32
    
    def _build_model(self, input_shape):
        """Build LSTM model architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(self.n_units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.n_units, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def _train_lstm(self, data: np.ndarray, name: str):
        """Train a single LSTM model."""
        from sklearn.preprocessing import MinMaxScaler
        import tensorflow as tf
        
        # Scale data
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(scaled)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train
        model = self._build_model((X.shape[1], 1))
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5, restore_best_weights=True
        )
        
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size,
                  callbacks=[early_stop], verbose=0)
        
        return model, scaler, scaled
    
    def fit(self, df: pd.DataFrame):
        """Fit LSTM models for inflows and outflows."""
        df = self._prepare_data(df)
        
        print(f"  Training LSTM for Inflows (this may take a moment)...")
        self.inflow_model, self.inflow_scaler, self.inflow_scaled = \
            self._train_lstm(df['inflow'].values, "inflow")
        
        print(f"  Training LSTM for Outflows...")
        self.outflow_model, self.outflow_scaler, self.outflow_scaled = \
            self._train_lstm(df['outflow'].values, "outflow")
        
        self.is_fitted = True
        return self
    
    def _predict_lstm(self, model, scaler, last_scaled, steps):
        """Generate predictions from a single LSTM model."""
        predictions = []
        current_seq = last_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        for _ in range(steps):
            pred = model.predict(current_seq, verbose=0)[0, 0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, 0] = pred
        
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        steps = steps or self.horizon_days
        
        # Generate predictions
        forecast_inflows = np.maximum(
            self._predict_lstm(self.inflow_model, self.inflow_scaler, self.inflow_scaled, steps), 0
        )
        forecast_outflows = np.maximum(
            self._predict_lstm(self.outflow_model, self.outflow_scaler, self.outflow_scaled, steps), 0
        )
        
        # Generate dates
        dates = pd.date_range(
            start=self.last_actual_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Calculate closing balances
        result = self._calculate_closing_balances(forecast_inflows, forecast_outflows, dates)
        
        # Estimate confidence intervals based on training data volatility
        inflow_std = self.training_data['inflow'].std() * 0.5
        outflow_std = self.training_data['outflow'].std() * 0.5
        
        result['inflow_lower'] = np.maximum(forecast_inflows - 1.96 * inflow_std, 0)
        result['inflow_upper'] = forecast_inflows + 1.96 * inflow_std
        result['outflow_lower'] = np.maximum(forecast_outflows - 1.96 * outflow_std, 0)
        result['outflow_upper'] = forecast_outflows + 1.96 * outflow_std
        result['lower_bound'] = result['inflow_lower'] - result['outflow_upper']
        result['upper_bound'] = result['inflow_upper'] - result['outflow_lower']
        
        return result


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble model for NT+365 (Annual forecast).
    Combines Prophet (60%) and LSTM (40%) for robust long-term predictions.
    """
    
    def __init__(self):
        super().__init__("Ensemble", "NT+365")
        self.prophet_weight = 0.6
        self.lstm_weight = 0.4
        self.prophet_model = ProphetForecaster("NT+365")
        self.lstm_model = LSTMForecaster()
        # Override LSTM horizon for ensemble
        self.lstm_model.horizon_days = TIME_HORIZONS["NT+365"]["days"]
    
    def fit(self, df: pd.DataFrame):
        """Fit both Prophet and LSTM models."""
        df_copy = df.copy()
        self.training_data = df_copy
        self.last_actual_balance = df_copy['closing_balance'].iloc[-1]
        self.last_actual_date = df_copy['date'].iloc[-1]
        
        print(f"  Training Prophet component...")
        self.prophet_model.fit(df_copy)
        
        print(f"  Training LSTM component...")
        self.lstm_model.fit(df_copy)
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate ensemble forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        steps = steps or self.horizon_days
        
        # Get predictions from both models
        prophet_pred = self.prophet_model.predict(steps)
        lstm_pred = self.lstm_model.predict(steps)
        
        # Weighted ensemble for inflows and outflows
        forecast_inflows = (
            self.prophet_weight * prophet_pred['forecast_inflow'].values +
            self.lstm_weight * lstm_pred['forecast_inflow'].values
        )
        forecast_outflows = (
            self.prophet_weight * prophet_pred['forecast_outflow'].values +
            self.lstm_weight * lstm_pred['forecast_outflow'].values
        )
        
        # Generate dates
        dates = prophet_pred['date']
        
        # Calculate closing balances with ensemble forecasts
        result = self._calculate_closing_balances(forecast_inflows, forecast_outflows, dates)
        
        # Combine confidence intervals
        result['inflow_lower'] = np.minimum(prophet_pred['inflow_lower'], lstm_pred['inflow_lower'])
        result['inflow_upper'] = np.maximum(prophet_pred['inflow_upper'], lstm_pred['inflow_upper'])
        result['outflow_lower'] = np.minimum(prophet_pred['outflow_lower'], lstm_pred['outflow_lower'])
        result['outflow_upper'] = np.maximum(prophet_pred['outflow_upper'], lstm_pred['outflow_upper'])
        result['lower_bound'] = result['inflow_lower'] - result['outflow_upper']
        result['upper_bound'] = result['inflow_upper'] - result['outflow_lower']
        
        # Store component forecasts
        result['prophet_net'] = prophet_pred['forecast_net']
        result['lstm_net'] = lstm_pred['forecast_net']
        result['trend'] = prophet_pred.get('trend', 0)
        
        return result


class CashFlowForecaster:
    """
    Main forecaster class that orchestrates all models.
    
    Model Selection by Horizon:
    - RT+7: ARIMA (best for short-term momentum)
    - T+30: Prophet (best for seasonality)
    - T+90: LSTM (best for complex patterns)
    - NT+365: Ensemble (Prophet + LSTM for robustness)
    """
    
    def __init__(self):
        self.models = {
            "RT+7": ARIMAForecaster(),
            "T+30": ProphetForecaster("T+30"),
            "T+90": LSTMForecaster(),
            "NT+365": EnsembleForecaster(),
        }
        self.is_fitted = False
        self.training_data = None
        self.last_actual_balance = None
        self.last_actual_date = None
    
    def fit(self, df: pd.DataFrame, verbose: int = 0):
        """Fit all models."""
        self.training_data = df.copy()
        self.last_actual_balance = df['closing_balance'].iloc[-1]
        self.last_actual_date = df['date'].iloc[-1]
        
        print("\n" + "="*60)
        print("TRAINING FORECASTING MODELS")
        print("="*60)
        
        print(f"\n[1/4] ARIMA (RT+7) - Short-term momentum...")
        self.models["RT+7"].fit(df)
        
        print(f"\n[2/4] Prophet (T+30) - Seasonality & trends...")
        self.models["T+30"].fit(df)
        
        print(f"\n[3/4] LSTM (T+90) - Complex patterns...")
        self.models["T+90"].fit(df)
        
        print(f"\n[4/4] Ensemble (NT+365) - Long-term forecast...")
        self.models["NT+365"].fit(df)
        
        self.is_fitted = True
        print("\n" + "="*60)
        print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return self
    
    def predict(self, horizon: str = None) -> Dict[str, pd.DataFrame]:
        """Generate forecasts."""
        if not self.is_fitted:
            raise ValueError("Models not fitted.")
        
        if horizon:
            return {horizon: self.models[horizon].predict()}
        
        return {hz: model.predict() for hz, model in self.models.items()}
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Dict]:
        """Evaluate forecast accuracy."""
        results = {}
        forecasts = self.predict()
        
        for horizon, forecast_df in forecasts.items():
            merged = forecast_df.merge(
                test_df[['date', 'inflow', 'outflow', 'net_cash_flow', 'closing_balance']],
                on='date', how='inner'
            )
            
            if len(merged) > 0:
                model = self.models[horizon]
                
                inflow_mape = model._calculate_mape(
                    merged['inflow'].values, merged['forecast_inflow'].values
                )
                outflow_mape = model._calculate_mape(
                    merged['outflow'].values, merged['forecast_outflow'].values
                )
                net_mape = model._calculate_mape(
                    merged['net_cash_flow'].values, merged['forecast_net'].values
                )
                balance_mape = model._calculate_mape(
                    merged['closing_balance'].values, merged['closing_balance_y'].values
                ) if 'closing_balance_y' in merged.columns else net_mape
                
                results[horizon] = {
                    'inflow_mape': inflow_mape,
                    'outflow_mape': outflow_mape,
                    'net_mape': net_mape,
                    'balance_mape': balance_mape,
                    'mape': net_mape,
                    'rating': model._get_rating(net_mape),
                    'samples': len(merged),
                    'mae': np.mean(np.abs(merged['net_cash_flow'] - merged['forecast_net'])),
                    'rmse': np.sqrt(np.mean((merged['net_cash_flow'] - merged['forecast_net'])**2))
                }
        
        return results


def run_backtest(df: pd.DataFrame, test_size: int = 90, target_col: str = "net_cash_flow"):
    """
    Run backtest with proper train/test split.
    """
    print(f"\nBacktest Configuration:")
    print(f"  Total data: {len(df)} days")
    print(f"  Training: {len(df) - test_size} days")
    print(f"  Testing: {test_size} days")
    print(f"  Last actual balance (T0): ${df['closing_balance'].iloc[-test_size-1]:,.0f}")
    
    # Split data
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # Train models
    forecaster = CashFlowForecaster()
    forecaster.fit(train_df)
    
    # Generate forecasts
    forecasts = forecaster.predict()
    
    # Evaluate
    results = forecaster.evaluate(test_df)
    
    # Add daily MAPE analysis
    results["daily_analysis"] = {}
    for horizon, forecast_df in forecasts.items():
        merged = forecast_df.merge(
            test_df[['date', 'net_cash_flow']],
            on='date', how='inner'
        )
        if len(merged) > 0:
            merged['day_of_week'] = merged['date'].dt.dayofweek
            merged['pct_error'] = np.abs(
                (merged['net_cash_flow'] - merged['forecast_net']) / 
                merged['net_cash_flow'].replace(0, np.nan)
            ) * 100
            results["daily_analysis"][horizon] = \
                merged.groupby('day_of_week')['pct_error'].mean().to_dict()
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    from data_simulator_realistic import generate_sample_data
    
    print("="*70)
    print("CASH FORECASTING SYSTEM - FULL MODEL TEST")
    print("="*70)
    
    print("\nGenerating realistic test data...")
    data = generate_sample_data(periods=365)
    daily_cash = data['daily_cash_position']
    
    print(f"Data: {len(daily_cash)} days")
    print(f"Starting Balance: ${daily_cash['opening_balance'].iloc[0]:,.0f}")
    print(f"Ending Balance (T0): ${daily_cash['closing_balance'].iloc[-1]:,.0f}")
    
    print("\nRunning backtest (test_size=30 days)...")
    results, forecaster, forecasts = run_backtest(daily_cash, test_size=30)
    
    print("\n" + "="*70)
    print("FORECAST RESULTS")
    print("="*70)
    
    for horizon, forecast_df in forecasts.items():
        print(f"\n{horizon} ({TIME_HORIZONS[horizon]['model']}):")
        print(f"  T0 (Last Actual): ${forecaster.last_actual_balance:,.0f}")
        print(f"  Day 1:")
        print(f"    Opening:  ${forecast_df.iloc[0]['opening_balance']:,.0f}")
        print(f"    +Inflows: ${forecast_df.iloc[0]['forecast_inflow']:,.0f}")
        print(f"    -Outflows: ${forecast_df.iloc[0]['forecast_outflow']:,.0f}")
        print(f"    =Closing: ${forecast_df.iloc[0]['closing_balance']:,.0f}")
        print(f"  Final Day ({len(forecast_df)}):")
        print(f"    Closing:  ${forecast_df.iloc[-1]['closing_balance']:,.0f}")
    
    print("\n" + "="*70)
    print("MAPE RESULTS")
    print("="*70)
    
    for horizon, metrics in results.items():
        if horizon != "daily_analysis":
            print(f"\n{horizon}:")
            print(f"  Inflow MAPE:  {metrics.get('inflow_mape', 0):.1f}%")
            print(f"  Outflow MAPE: {metrics.get('outflow_mape', 0):.1f}%")
            print(f"  Net MAPE:     {metrics.get('net_mape', 0):.1f}%")
            print(f"  Rating:       {metrics.get('rating', 'N/A')}")
