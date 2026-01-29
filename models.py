"""
Forecasting Models
==================
Implementation of multiple forecasting models for different time horizons:
- ARIMA: RT+7 (Real-Time + 7 days)
- Prophet: T+30 (30-day forecast)
- LSTM: T+90 (90-day forecast)
- Ensemble: NT+365 (Annual forecast)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

from config import (
    TIME_HORIZONS, ARIMA_PARAMS, PROPHET_PARAMS, LSTM_PARAMS,
    MAPE_THRESHOLDS
)


class BaseForecaster:
    """Base class for all forecasting models."""
    
    def __init__(self, name: str, horizon_days: int):
        self.name = name
        self.horizon_days = horizon_days
        self.model = None
        self.scaler = None
        self.is_fitted = False
        self.training_data = None
        self.feature_names = None
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = "net_cash_flow") -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        # Ensure we have the target column
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        self.training_data = df
        return df, df[target_col]
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        # Avoid division by zero
        mask = actual != 0
        if not mask.any():
            return np.nan
        
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        return mape
    
    def get_mape_rating(self, mape: float, horizon: str) -> str:
        """Get rating based on MAPE thresholds."""
        thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
        
        if mape <= thresholds["excellent"]:
            return "Excellent"
        elif mape <= thresholds["good"]:
            return "Good"
        elif mape <= thresholds["acceptable"]:
            return "Acceptable"
        else:
            return "Poor"


class ARIMAForecaster(BaseForecaster):
    """
    ARIMA model for RT+7 (short-term) forecasting.
    Best for capturing recent patterns and short-term momentum.
    """
    
    def __init__(self):
        super().__init__("ARIMA", TIME_HORIZONS["RT+7"]["days"])
        self.order = ARIMA_PARAMS.order
        self.seasonal_order = ARIMA_PARAMS.seasonal_order
    
    def fit(self, df: pd.DataFrame, target_col: str = "net_cash_flow") -> 'ARIMAForecaster':
        """Fit ARIMA model."""
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        
        df, y = self.prepare_data(df, target_col)
        
        # Fit SARIMAX model
        self.model = SARIMAX(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=ARIMA_PARAMS.enforce_stationarity,
            enforce_invertibility=ARIMA_PARAMS.enforce_invertibility
        )
        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True
        
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if steps is None:
            steps = self.horizon_days
        
        # Generate forecast
        forecast = self.fitted_model.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create date range for forecast
        last_date = self.training_data["date"].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        result = pd.DataFrame({
            "date": forecast_dates,
            "forecast": forecast_mean.values,
            "lower_bound": conf_int.iloc[:, 0].values,
            "upper_bound": conf_int.iloc[:, 1].values,
            "model": "ARIMA",
            "horizon": "RT+7"
        })
        
        return result
    
    def get_fitted_values(self) -> pd.DataFrame:
        """Get in-sample fitted values for comparison."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        fitted = self.fitted_model.fittedvalues
        
        result = pd.DataFrame({
            "date": self.training_data["date"],
            "actual": self.training_data["net_cash_flow"],
            "fitted": fitted.values
        })
        
        return result


class ProphetForecaster(BaseForecaster):
    """
    Prophet model for T+30 (medium-term) forecasting.
    Excellent for handling seasonality and holiday effects.
    """
    
    def __init__(self):
        super().__init__("Prophet", TIME_HORIZONS["T+30"]["days"])
    
    def fit(self, df: pd.DataFrame, target_col: str = "net_cash_flow") -> 'ProphetForecaster':
        """Fit Prophet model."""
        from prophet import Prophet
        
        df, y = self.prepare_data(df, target_col)
        
        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            "ds": df["date"],
            "y": y
        })
        
        # Initialize and fit Prophet
        self.model = Prophet(
            yearly_seasonality=PROPHET_PARAMS.yearly_seasonality,
            weekly_seasonality=PROPHET_PARAMS.weekly_seasonality,
            daily_seasonality=PROPHET_PARAMS.daily_seasonality,
            changepoint_prior_scale=PROPHET_PARAMS.changepoint_prior_scale,
            seasonality_prior_scale=PROPHET_PARAMS.seasonality_prior_scale,
        )
        
        # Add custom seasonalities
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        
        self.model.fit(prophet_df)
        self.is_fitted = True
        
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if steps is None:
            steps = self.horizon_days
        
        # Create future dataframe
        last_date = self.training_data["date"].max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        future = pd.DataFrame({"ds": future_dates})
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        result = pd.DataFrame({
            "date": forecast["ds"],
            "forecast": forecast["yhat"],
            "lower_bound": forecast["yhat_lower"],
            "upper_bound": forecast["yhat_upper"],
            "trend": forecast["trend"],
            "model": "Prophet",
            "horizon": "T+30"
        })
        
        return result
    
    def get_components(self) -> Dict[str, pd.DataFrame]:
        """Get seasonality and trend components."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        # Get historical predictions with components
        historical = pd.DataFrame({"ds": self.training_data["date"]})
        forecast = self.model.predict(historical)
        
        components = {
            "trend": forecast[["ds", "trend"]],
            "weekly": forecast[["ds", "weekly"]] if "weekly" in forecast.columns else None,
            "yearly": forecast[["ds", "yearly"]] if "yearly" in forecast.columns else None,
            "monthly": forecast[["ds", "monthly"]] if "monthly" in forecast.columns else None,
        }
        
        return {k: v for k, v in components.items() if v is not None}


class LSTMForecaster(BaseForecaster):
    """
    LSTM model for T+90 (longer-term) forecasting.
    Captures complex non-linear patterns and long-term dependencies.
    """
    
    def __init__(self):
        super().__init__("LSTM", TIME_HORIZONS["T+90"]["days"])
        self.sequence_length = LSTM_PARAMS.sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
    
    def _create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple) -> Any:
        """Build LSTM model architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        
        model = Sequential([
            LSTM(LSTM_PARAMS.n_units, return_sequences=True, input_shape=input_shape),
            Dropout(LSTM_PARAMS.dropout),
            LSTM(LSTM_PARAMS.n_units, return_sequences=False),
            Dropout(LSTM_PARAMS.dropout),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LSTM_PARAMS.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, df: pd.DataFrame, target_col: str = "net_cash_flow", 
            validation_split: float = 0.1, verbose: int = 0) -> 'LSTMForecaster':
        """Fit LSTM model."""
        import tensorflow as tf
        
        df, y = self.prepare_data(df, target_col)
        
        # Scale data
        y_scaled = self.scaler.fit_transform(y.values.reshape(-1, 1))
        
        # Create sequences
        X, y_seq = self._create_sequences(y_scaled, self.sequence_length)
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build and train model
        self.model = self._build_model((X.shape[1], X.shape[2]))
        
        # Early stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X, y_seq,
            epochs=LSTM_PARAMS.epochs,
            batch_size=LSTM_PARAMS.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=verbose
        )
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if steps is None:
            steps = self.horizon_days
        
        # Get the last sequence from training data
        y = self.training_data["net_cash_flow"].values
        y_scaled = self.scaler.transform(y.reshape(-1, 1))
        
        # Initialize with last sequence
        current_sequence = y_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        for _ in range(steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Create date range
        last_date = self.training_data["date"].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=steps,
            freq='D'
        )
        
        # Estimate confidence intervals (using training std)
        std = np.std(y)
        lower_bound = predictions - 1.96 * std * 0.5  # Simplified CI
        upper_bound = predictions + 1.96 * std * 0.5
        
        result = pd.DataFrame({
            "date": forecast_dates,
            "forecast": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "model": "LSTM",
            "horizon": "T+90"
        })
        
        return result


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble model for NT+365 (annual) forecasting.
    Combines Prophet and LSTM predictions with weighted averaging.
    """
    
    def __init__(self, prophet_weight: float = 0.6, lstm_weight: float = 0.4):
        super().__init__("Ensemble", TIME_HORIZONS["NT+365"]["days"])
        self.prophet_weight = prophet_weight
        self.lstm_weight = lstm_weight
        self.prophet_model = ProphetForecaster()
        self.lstm_model = LSTMForecaster()
    
    def fit(self, df: pd.DataFrame, target_col: str = "net_cash_flow", verbose: int = 0) -> 'EnsembleForecaster':
        """Fit both Prophet and LSTM models."""
        df, y = self.prepare_data(df, target_col)
        
        # Fit both models
        self.prophet_model.fit(df, target_col)
        self.lstm_model.fit(df, target_col, verbose=verbose)
        
        self.is_fitted = True
        return self
    
    def predict(self, steps: int = None) -> pd.DataFrame:
        """Generate ensemble forecast."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if steps is None:
            steps = self.horizon_days
        
        # Get predictions from both models
        prophet_pred = self.prophet_model.predict(steps)
        lstm_pred = self.lstm_model.predict(steps)
        
        # Weighted ensemble
        ensemble_forecast = (
            self.prophet_weight * prophet_pred["forecast"].values +
            self.lstm_weight * lstm_pred["forecast"].values
        )
        
        # Combine confidence intervals
        lower_bound = np.minimum(prophet_pred["lower_bound"].values, lstm_pred["lower_bound"].values)
        upper_bound = np.maximum(prophet_pred["upper_bound"].values, lstm_pred["upper_bound"].values)
        
        result = pd.DataFrame({
            "date": prophet_pred["date"],
            "forecast": ensemble_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "prophet_forecast": prophet_pred["forecast"].values,
            "lstm_forecast": lstm_pred["forecast"].values,
            "trend": prophet_pred["trend"].values if "trend" in prophet_pred.columns else None,
            "model": "Ensemble",
            "horizon": "NT+365"
        })
        
        return result


class CashFlowForecaster:
    """
    Main forecasting class that orchestrates all models.
    Provides unified interface for training and prediction.
    """
    
    def __init__(self):
        self.models = {
            "RT+7": ARIMAForecaster(),
            "T+30": ProphetForecaster(),
            "T+90": LSTMForecaster(),
            "NT+365": EnsembleForecaster(),
        }
        self.is_fitted = False
        self.training_data = None
    
    def fit(self, df: pd.DataFrame, target_col: str = "net_cash_flow", verbose: int = 0) -> 'CashFlowForecaster':
        """Fit all models."""
        self.training_data = df.copy()
        
        print("Training ARIMA (RT+7)...")
        self.models["RT+7"].fit(df, target_col)
        
        print("Training Prophet (T+30)...")
        self.models["T+30"].fit(df, target_col)
        
        print("Training LSTM (T+90)...")
        self.models["T+90"].fit(df, target_col, verbose=verbose)
        
        print("Training Ensemble (NT+365)...")
        self.models["NT+365"].fit(df, target_col, verbose=verbose)
        
        self.is_fitted = True
        print("All models trained successfully!")
        
        return self
    
    def predict(self, horizon: str = None) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for specified horizon or all horizons.
        
        Args:
            horizon: Specific horizon ('RT+7', 'T+30', 'T+90', 'NT+365') or None for all
        
        Returns:
            Dictionary of forecasts by horizon
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")
        
        if horizon:
            if horizon not in self.models:
                raise ValueError(f"Unknown horizon: {horizon}")
            return {horizon: self.models[horizon].predict()}
        
        # Generate all forecasts
        forecasts = {}
        for hz, model in self.models.items():
            forecasts[hz] = model.predict()
        
        return forecasts
    
    def evaluate(self, test_df: pd.DataFrame, target_col: str = "net_cash_flow") -> Dict[str, Dict]:
        """
        Evaluate model performance on test data.
        
        Returns:
            Dictionary with MAPE and other metrics for each horizon
        """
        results = {}
        
        for horizon, model in self.models.items():
            forecast = model.predict()
            
            # Match forecast dates with test data
            merged = forecast.merge(
                test_df[["date", target_col]],
                on="date",
                how="inner"
            )
            
            if len(merged) > 0:
                mape = model.calculate_mape(merged[target_col].values, merged["forecast"].values)
                rating = model.get_mape_rating(mape, horizon)
                
                results[horizon] = {
                    "mape": mape,
                    "rating": rating,
                    "samples": len(merged),
                    "mae": np.mean(np.abs(merged[target_col].values - merged["forecast"].values)),
                    "rmse": np.sqrt(np.mean((merged[target_col].values - merged["forecast"].values) ** 2))
                }
        
        return results


def run_backtest(df: pd.DataFrame, 
                 test_size: int = 90,
                 target_col: str = "net_cash_flow") -> Dict[str, Any]:
    """
    Run time-series backtesting on all models.
    
    Args:
        df: Full dataset
        test_size: Number of days to use for testing
        target_col: Target column name
    
    Returns:
        Backtest results including daily MAPE breakdown
    """
    # Split data
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # Train models
    forecaster = CashFlowForecaster()
    forecaster.fit(train_df, target_col, verbose=0)
    
    # Get forecasts
    forecasts = forecaster.predict()
    
    # Evaluate
    results = forecaster.evaluate(test_df, target_col)
    
    # Add daily breakdown
    results["daily_analysis"] = {}
    for horizon, forecast_df in forecasts.items():
        merged = forecast_df.merge(
            test_df[["date", target_col]],
            on="date",
            how="inner"
        )
        
        if len(merged) > 0:
            merged["day_of_week"] = merged["date"].dt.dayofweek
            merged["error"] = np.abs(merged[target_col] - merged["forecast"])
            merged["pct_error"] = np.abs((merged[target_col] - merged["forecast"]) / merged[target_col]) * 100
            
            # Daily MAPE by day of week
            daily_mape = merged.groupby("day_of_week")["pct_error"].mean().to_dict()
            results["daily_analysis"][horizon] = daily_mape
    
    return results, forecaster, forecasts


if __name__ == "__main__":
    # Test the models
    from data_simulator import generate_sample_data
    
    print("Generating sample data...")
    data = generate_sample_data(periods=730)
    daily_cash = data["daily_cash_position"]
    
    print(f"\nData shape: {daily_cash.shape}")
    print(f"Date range: {daily_cash['date'].min()} to {daily_cash['date'].max()}")
    
    print("\nRunning backtest...")
    results, forecaster, forecasts = run_backtest(daily_cash)
    
    print("\nBacktest Results:")
    for horizon, metrics in results.items():
        if horizon != "daily_analysis":
            print(f"\n{horizon}:")
            print(f"  MAPE: {metrics['mape']:.2f}%")
            print(f"  Rating: {metrics['rating']}")
