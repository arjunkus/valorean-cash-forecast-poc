"""
Fast Models for POC - Replaces LSTM with Prophet for T+90
For production, use models.py with GPU infrastructure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from config import TIME_HORIZONS, MAPE_THRESHOLDS

class BaseForecaster:
    def __init__(self, name: str, horizon_days: int):
        self.name = name
        self.horizon_days = horizon_days
        self.model = None
        self.is_fitted = False
        self.training_data = None

    def prepare_data(self, df, target_col="net_cash_flow"):
        df = df.copy().sort_values("date").reset_index(drop=True)
        self.training_data = df
        return df, df[target_col]

    def calculate_mape(self, actual, predicted):
        actual, predicted = np.array(actual), np.array(predicted)
        mask = actual != 0
        if not mask.any(): return np.nan
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

    def get_mape_rating(self, mape, horizon):
        thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
        if mape <= thresholds["excellent"]: return "Excellent"
        elif mape <= thresholds["good"]: return "Good"
        elif mape <= thresholds["acceptable"]: return "Acceptable"
        return "Poor"


class ARIMAForecaster(BaseForecaster):
    def __init__(self):
        super().__init__("ARIMA", TIME_HORIZONS["RT+7"]["days"])

    def fit(self, df, target_col="net_cash_flow"):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        df, y = self.prepare_data(df, target_col)
        self.model = SARIMAX(y, order=(5,1,2), seasonal_order=(1,1,1,7),
                            enforce_stationarity=False, enforce_invertibility=False)
        self.fitted_model = self.model.fit(disp=False)
        self.is_fitted = True
        return self

    def predict(self, steps=None):
        steps = steps or self.horizon_days
        forecast = self.fitted_model.get_forecast(steps=steps)
        conf_int = forecast.conf_int()
        last_date = self.training_data["date"].max()
        dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        return pd.DataFrame({
            "date": dates, "forecast": forecast.predicted_mean.values,
            "lower_bound": conf_int.iloc[:, 0].values, "upper_bound": conf_int.iloc[:, 1].values,
            "model": "ARIMA", "horizon": "RT+7"
        })


class ProphetForecaster(BaseForecaster):
    def __init__(self, horizon_name="T+30"):
        days = TIME_HORIZONS.get(horizon_name, {}).get("days", 30)
        super().__init__("Prophet", days)
        self.horizon_name = horizon_name

    def fit(self, df, target_col="net_cash_flow"):
        from prophet import Prophet
        df, y = self.prepare_data(df, target_col)
        prophet_df = pd.DataFrame({"ds": df["date"], "y": y})
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=True,
                            daily_seasonality=False, changepoint_prior_scale=0.05)
        self.model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        self.model.fit(prophet_df)
        self.is_fitted = True
        return self

    def predict(self, steps=None):
        steps = steps or self.horizon_days
        last_date = self.training_data["date"].max()
        future = pd.DataFrame({"ds": pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')})
        forecast = self.model.predict(future)
        return pd.DataFrame({
            "date": forecast["ds"], "forecast": forecast["yhat"],
            "lower_bound": forecast["yhat_lower"], "upper_bound": forecast["yhat_upper"],
            "trend": forecast["trend"], "model": "Prophet", "horizon": self.horizon_name
        })


class EnsembleForecaster(BaseForecaster):
    def __init__(self):
        super().__init__("Ensemble", TIME_HORIZONS["NT+365"]["days"])
        self.prophet1 = ProphetForecaster("NT+365")
        self.prophet2 = ProphetForecaster("NT+365")

    def fit(self, df, target_col="net_cash_flow", verbose=0):
        self.training_data = df.copy()
        self.prophet1.fit(df, target_col)
        self.prophet2.fit(df, target_col)
        self.is_fitted = True
        return self

    def predict(self, steps=None):
        steps = steps or self.horizon_days
        p1 = self.prophet1.predict(steps)
        p2 = self.prophet2.predict(steps)
        return pd.DataFrame({
            "date": p1["date"], "forecast": (p1["forecast"] + p2["forecast"]) / 2,
            "lower_bound": p1["lower_bound"], "upper_bound": p2["upper_bound"],
            "prophet_forecast": p1["forecast"], "lstm_forecast": p2["forecast"],
            "trend": p1["trend"], "model": "Ensemble", "horizon": "NT+365"
        })


class CashFlowForecaster:
    def __init__(self):
        self.models = {
            "RT+7": ARIMAForecaster(),
            "T+30": ProphetForecaster("T+30"),
            "T+90": ProphetForecaster("T+90"),
            "NT+365": EnsembleForecaster(),
        }
        self.is_fitted = False
        self.training_data = None

    def fit(self, df, target_col="net_cash_flow", verbose=0):
        self.training_data = df.copy()
        print("Training ARIMA (RT+7)...")
        self.models["RT+7"].fit(df, target_col)
        print("Training Prophet (T+30)...")
        self.models["T+30"].fit(df, target_col)
        print("Training Prophet (T+90)...")
        self.models["T+90"].fit(df, target_col)
        print("Training Ensemble (NT+365)...")
        self.models["NT+365"].fit(df, target_col, verbose)
        self.is_fitted = True
        print("All models trained successfully!")
        return self

    def predict(self, horizon=None):
        if horizon:
            return {horizon: self.models[horizon].predict()}
        return {hz: model.predict() for hz, model in self.models.items()}

    def evaluate(self, test_df, target_col="net_cash_flow"):
        results = {}
        for horizon, model in self.models.items():
            forecast = model.predict()
            merged = forecast.merge(test_df[["date", target_col]], on="date", how="inner")
            if len(merged) > 0:
                mape = model.calculate_mape(merged[target_col].values, merged["forecast"].values)
                results[horizon] = {
                    "mape": mape, "rating": model.get_mape_rating(mape, horizon),
                    "samples": len(merged),
                    "mae": np.mean(np.abs(merged[target_col].values - merged["forecast"].values)),
                    "rmse": np.sqrt(np.mean((merged[target_col].values - merged["forecast"].values) ** 2))
                }
        return results


def run_backtest(df, test_size=90, target_col="net_cash_flow"):
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    forecaster = CashFlowForecaster()
    forecaster.fit(train_df, target_col, verbose=0)
    forecasts = forecaster.predict()
    results = forecaster.evaluate(test_df, target_col)
    results["daily_analysis"] = {}
    for horizon, forecast_df in forecasts.items():
        merged = forecast_df.merge(test_df[["date", target_col]], on="date", how="inner")
        if len(merged) > 0:
            merged["day_of_week"] = merged["date"].dt.dayofweek
            merged["pct_error"] = np.abs((merged[target_col] - merged["forecast"]) / merged[target_col]) * 100
            results["daily_analysis"][horizon] = merged.groupby("day_of_week")["pct_error"].mean().to_dict()
    return results, forecaster, forecasts
