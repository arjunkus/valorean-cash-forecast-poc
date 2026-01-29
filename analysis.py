"""
Analysis Module
===============
Comprehensive analysis tools for cash flow forecasting:
- Daily MAPE Analysis
- Trend Analysis
- SHAP (SHapley Additive exPlanations) Analysis
- Outlier Detection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.signal import find_peaks

from config import (
    MAPE_THRESHOLDS, OUTLIER_THRESHOLDS, TIME_HORIZONS
)


# =============================================================================
# MAPE ANALYSIS
# =============================================================================

@dataclass
class MAPEResult:
    """Container for MAPE analysis results."""
    overall_mape: float
    rating: str
    daily_mape: Dict[int, float]  # By day of week (0=Monday)
    weekly_mape: Dict[int, float]  # By week number
    monthly_mape: Dict[int, float]  # By month
    best_day: Tuple[int, float]
    worst_day: Tuple[int, float]
    trend: str  # 'improving', 'stable', 'degrading'
    insights: List[str]


class MAPEAnalyzer:
    """
    Comprehensive MAPE (Mean Absolute Percentage Error) analysis.
    Provides granular error breakdown by day, week, and month.
    """
    
    DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    def __init__(self, horizon: str):
        """
        Initialize MAPE Analyzer.
        
        Args:
            horizon: Time horizon ('RT+7', 'T+30', 'T+90', 'NT+365')
        """
        self.horizon = horizon
        self.thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
    
    def analyze(self, actual: pd.Series, predicted: pd.Series, 
                dates: pd.Series) -> MAPEResult:
        """
        Perform comprehensive MAPE analysis.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            dates: Corresponding dates
        
        Returns:
            MAPEResult with detailed breakdown
        """
        # Create analysis DataFrame
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'actual': actual.values,
            'predicted': predicted.values
        })
        
        # Calculate errors
        df['error'] = np.abs(df['actual'] - df['predicted'])
        df['pct_error'] = np.where(
            df['actual'] != 0,
            np.abs((df['actual'] - df['predicted']) / df['actual']) * 100,
            0
        )
        
        # Add time components
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        
        # Overall MAPE
        overall_mape = df['pct_error'].mean()
        rating = self._get_rating(overall_mape)
        
        # Daily MAPE (by day of week)
        daily_mape = df.groupby('day_of_week')['pct_error'].mean().to_dict()
        
        # Weekly MAPE
        weekly_mape = df.groupby('week')['pct_error'].mean().to_dict()
        
        # Monthly MAPE
        monthly_mape = df.groupby('month')['pct_error'].mean().to_dict()
        
        # Best and worst days
        best_day_idx = min(daily_mape, key=daily_mape.get)
        worst_day_idx = max(daily_mape, key=daily_mape.get)
        best_day = (best_day_idx, daily_mape[best_day_idx])
        worst_day = (worst_day_idx, daily_mape[worst_day_idx])
        
        # Trend analysis (is accuracy improving over time?)
        trend = self._analyze_trend(df)
        
        # Generate insights
        insights = self._generate_insights(
            overall_mape, daily_mape, monthly_mape, 
            best_day, worst_day, trend
        )
        
        return MAPEResult(
            overall_mape=overall_mape,
            rating=rating,
            daily_mape=daily_mape,
            weekly_mape=weekly_mape,
            monthly_mape=monthly_mape,
            best_day=best_day,
            worst_day=worst_day,
            trend=trend,
            insights=insights
        )
    
    def _get_rating(self, mape: float) -> str:
        """Get rating based on MAPE thresholds."""
        if mape <= self.thresholds["excellent"]:
            return "Excellent"
        elif mape <= self.thresholds["good"]:
            return "Good"
        elif mape <= self.thresholds["acceptable"]:
            return "Acceptable"
        else:
            return "Poor"
    
    def _analyze_trend(self, df: pd.DataFrame) -> str:
        """Analyze if forecast accuracy is improving over time."""
        # Calculate rolling MAPE
        df_sorted = df.sort_values('date')
        rolling_mape = df_sorted['pct_error'].rolling(window=7).mean()
        
        if len(rolling_mape.dropna()) < 14:
            return "insufficient_data"
        
        # Compare first half vs second half
        mid = len(rolling_mape) // 2
        first_half = rolling_mape.iloc[:mid].mean()
        second_half = rolling_mape.iloc[mid:].mean()
        
        change_pct = (second_half - first_half) / first_half * 100
        
        if change_pct < -10:
            return "improving"
        elif change_pct > 10:
            return "degrading"
        else:
            return "stable"
    
    def _generate_insights(self, overall_mape: float, daily_mape: Dict,
                          monthly_mape: Dict, best_day: Tuple,
                          worst_day: Tuple, trend: str) -> List[str]:
        """Generate actionable insights from MAPE analysis."""
        insights = []
        
        # Overall accuracy insight
        if overall_mape <= self.thresholds["excellent"]:
            insights.append(f"✅ Excellent forecast accuracy ({overall_mape:.1f}% MAPE) for {self.horizon} horizon")
        elif overall_mape > self.thresholds["poor"]:
            insights.append(f"⚠️ Poor forecast accuracy ({overall_mape:.1f}% MAPE) - consider model retraining or data review")
        
        # Day-of-week insights
        best_day_name = self.DAY_NAMES[best_day[0]]
        worst_day_name = self.DAY_NAMES[worst_day[0]]
        
        if worst_day[1] > best_day[1] * 1.5:
            insights.append(
                f"📊 {worst_day_name} shows significantly higher errors ({worst_day[1]:.1f}%) "
                f"vs {best_day_name} ({best_day[1]:.1f}%). Consider day-specific adjustments."
            )
        
        # Weekend vs weekday
        weekday_mape = np.mean([daily_mape.get(i, 0) for i in range(5)])
        weekend_mape = np.mean([daily_mape.get(i, 0) for i in range(5, 7)])
        
        if weekend_mape > weekday_mape * 1.3:
            insights.append("📅 Weekend forecasts less accurate - limited business activity may affect predictions")
        
        # Monthly patterns
        if monthly_mape:
            high_error_months = [m for m, e in monthly_mape.items() if e > overall_mape * 1.5]
            if high_error_months:
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                high_months = [month_names[m-1] for m in high_error_months if 1 <= m <= 12]
                if high_months:
                    insights.append(f"📈 Higher forecast errors in: {', '.join(high_months)}. Review seasonal factors.")
        
        # Trend insights
        if trend == "improving":
            insights.append("📈 Forecast accuracy is improving over time - model is adapting well")
        elif trend == "degrading":
            insights.append("📉 Forecast accuracy declining - consider model refresh or data quality review")
        
        return insights
    
    def get_daily_breakdown_df(self, actual: pd.Series, predicted: pd.Series,
                               dates: pd.Series) -> pd.DataFrame:
        """
        Get detailed daily MAPE breakdown as DataFrame.
        
        Returns:
            DataFrame with date, actual, predicted, error, and day information
        """
        df = pd.DataFrame({
            'date': pd.to_datetime(dates),
            'actual': actual.values,
            'predicted': predicted.values
        })
        
        df['error'] = np.abs(df['actual'] - df['predicted'])
        df['pct_error'] = np.where(
            df['actual'] != 0,
            np.abs((df['actual'] - df['predicted']) / df['actual']) * 100,
            0
        )
        df['day_of_week'] = df['date'].dt.day_name()
        df['is_weekend'] = df['date'].dt.dayofweek >= 5
        
        return df.sort_values('date')


# =============================================================================
# TREND ANALYSIS
# =============================================================================

@dataclass
class TrendResult:
    """Container for trend analysis results."""
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_slope: float
    trend_values: np.ndarray
    seasonality: Dict[str, np.ndarray]
    residuals: np.ndarray
    change_points: List[datetime]
    forecast_trend: np.ndarray
    insights: List[str]


class TrendAnalyzer:
    """
    Decompose and analyze trends in cash flow data.
    Uses STL (Seasonal-Trend decomposition using Loess).
    """
    
    def __init__(self, period: int = 7):
        """
        Initialize Trend Analyzer.
        
        Args:
            period: Seasonality period (default: 7 for weekly)
        """
        self.period = period
    
    def analyze(self, df: pd.DataFrame, value_col: str = "net_cash_flow") -> TrendResult:
        """
        Perform comprehensive trend analysis.
        
        Args:
            df: DataFrame with date and value columns
            value_col: Name of the value column
        
        Returns:
            TrendResult with decomposition and insights
        """
        from statsmodels.tsa.seasonal import STL
        
        # Prepare data
        df = df.copy().sort_values('date')
        values = df[value_col].values
        dates = df['date'].values
        
        # STL Decomposition
        stl = STL(values, period=self.period, robust=True)
        result = stl.fit()
        
        trend = result.trend
        seasonal = result.seasonal
        residuals = result.resid
        
        # Determine trend direction
        trend_slope = np.polyfit(range(len(trend)), trend, 1)[0]
        
        if trend_slope > 0.01 * np.mean(np.abs(values)):
            trend_direction = "increasing"
        elif trend_slope < -0.01 * np.mean(np.abs(values)):
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Detect change points
        change_points = self._detect_change_points(trend, dates)
        
        # Seasonality patterns
        seasonality = self._analyze_seasonality(df, value_col, seasonal)
        
        # Forecast trend (simple linear extrapolation)
        n_forecast = 30
        x_future = np.arange(len(trend), len(trend) + n_forecast)
        forecast_trend = np.poly1d(np.polyfit(range(len(trend)), trend, 1))(x_future)
        
        # Generate insights
        insights = self._generate_insights(
            trend_direction, trend_slope, change_points, 
            seasonality, np.mean(values)
        )
        
        return TrendResult(
            trend_direction=trend_direction,
            trend_slope=trend_slope,
            trend_values=trend,
            seasonality=seasonality,
            residuals=residuals,
            change_points=change_points,
            forecast_trend=forecast_trend,
            insights=insights
        )
    
    def _detect_change_points(self, trend: np.ndarray, 
                              dates: np.ndarray) -> List[datetime]:
        """Detect significant change points in the trend."""
        # Calculate rate of change
        diff = np.diff(trend)
        
        # Find peaks in absolute rate of change
        abs_diff = np.abs(diff)
        threshold = np.mean(abs_diff) + 2 * np.std(abs_diff)
        
        peaks, _ = find_peaks(abs_diff, height=threshold, distance=14)
        
        change_points = [pd.Timestamp(dates[i+1]) for i in peaks if i+1 < len(dates)]
        
        return change_points
    
    def _analyze_seasonality(self, df: pd.DataFrame, value_col: str,
                            seasonal_component: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze different seasonality patterns."""
        df = df.copy()
        df['seasonal'] = seasonal_component
        
        # Weekly pattern
        weekly = df.groupby(df['date'].dt.dayofweek)['seasonal'].mean().values
        
        # Monthly pattern (day of month)
        monthly = df.groupby(df['date'].dt.day)['seasonal'].mean().values
        
        return {
            "weekly": weekly,
            "monthly": monthly
        }
    
    def _generate_insights(self, direction: str, slope: float,
                          change_points: List, seasonality: Dict,
                          mean_value: float) -> List[str]:
        """Generate insights from trend analysis."""
        insights = []
        
        # Trend direction insight
        daily_change = slope
        monthly_change = slope * 30
        pct_change = (monthly_change / mean_value) * 100 if mean_value != 0 else 0
        
        if direction == "increasing":
            insights.append(
                f"📈 Upward trend detected: ~${monthly_change:,.0f}/month ({pct_change:+.1f}%). "
                f"Cash position is strengthening."
            )
        elif direction == "decreasing":
            insights.append(
                f"📉 Downward trend detected: ~${monthly_change:,.0f}/month ({pct_change:+.1f}%). "
                f"Monitor liquidity closely."
            )
        else:
            insights.append("➡️ Cash flow trend is relatively stable with no significant drift.")
        
        # Change points
        if change_points:
            insights.append(
                f"🔄 Detected {len(change_points)} significant trend changes. "
                f"Latest: {change_points[-1].strftime('%Y-%m-%d') if change_points else 'N/A'}"
            )
        
        # Weekly seasonality
        weekly = seasonality.get("weekly", [])
        if len(weekly) > 0:
            peak_day = np.argmax(weekly)
            low_day = np.argmin(weekly)
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            
            if peak_day < len(day_names) and low_day < len(day_names):
                insights.append(
                    f"📊 Weekly pattern: Peak on {day_names[peak_day]}, "
                    f"lowest on {day_names[low_day]}"
                )
        
        return insights


# =============================================================================
# OUTLIER DETECTION
# =============================================================================

@dataclass
class OutlierResult:
    """Container for outlier detection results."""
    outlier_indices: np.ndarray
    outlier_dates: List[datetime]
    outlier_values: np.ndarray
    outlier_scores: np.ndarray
    detection_method: str
    total_outliers: int
    outlier_pct: float
    insights: List[str]


class OutlierDetector:
    """
    Multi-method outlier detection for cash flow data.
    Combines statistical and ML-based approaches.
    """
    
    def __init__(self):
        """Initialize Outlier Detector with default thresholds."""
        self.z_threshold = OUTLIER_THRESHOLDS.z_score_threshold
        self.iqr_multiplier = OUTLIER_THRESHOLDS.iqr_multiplier
        self.isolation_contamination = OUTLIER_THRESHOLDS.isolation_contamination
    
    def detect(self, df: pd.DataFrame, value_col: str = "net_cash_flow",
              method: str = "ensemble") -> OutlierResult:
        """
        Detect outliers using specified method.
        
        Args:
            df: DataFrame with values
            value_col: Column to analyze
            method: 'zscore', 'iqr', 'isolation_forest', or 'ensemble'
        
        Returns:
            OutlierResult with detected outliers
        """
        df = df.copy().sort_values('date').reset_index(drop=True)
        values = df[value_col].values
        
        if method == "zscore":
            outliers, scores = self._zscore_method(values)
        elif method == "iqr":
            outliers, scores = self._iqr_method(values)
        elif method == "isolation_forest":
            outliers, scores = self._isolation_forest_method(values)
        else:  # ensemble
            outliers, scores = self._ensemble_method(values)
        
        outlier_indices = np.where(outliers)[0]
        outlier_dates = df.loc[outlier_indices, 'date'].tolist()
        outlier_values = values[outlier_indices]
        outlier_scores = scores[outlier_indices]
        
        # Generate insights
        insights = self._generate_insights(
            df, outlier_indices, outlier_values, value_col
        )
        
        return OutlierResult(
            outlier_indices=outlier_indices,
            outlier_dates=outlier_dates,
            outlier_values=outlier_values,
            outlier_scores=outlier_scores,
            detection_method=method,
            total_outliers=len(outlier_indices),
            outlier_pct=len(outlier_indices) / len(values) * 100,
            insights=insights
        )
    
    def _zscore_method(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Z-score based outlier detection."""
        z_scores = np.abs(stats.zscore(values))
        outliers = z_scores > self.z_threshold
        return outliers, z_scores
    
    def _iqr_method(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """IQR (Interquartile Range) based outlier detection."""
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        outliers = (values < lower_bound) | (values > upper_bound)
        
        # Score based on distance from bounds
        scores = np.maximum(
            (lower_bound - values) / iqr,
            (values - upper_bound) / iqr
        )
        scores = np.maximum(scores, 0)
        
        return outliers, scores
    
    def _isolation_forest_method(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest based outlier detection."""
        X = values.reshape(-1, 1)
        
        clf = IsolationForest(
            contamination=self.isolation_contamination,
            random_state=42
        )
        predictions = clf.fit_predict(X)
        scores = -clf.score_samples(X)  # Higher = more anomalous
        
        outliers = predictions == -1
        
        return outliers, scores
    
    def _ensemble_method(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Combine multiple methods for robust detection."""
        zscore_outliers, zscore_scores = self._zscore_method(values)
        iqr_outliers, iqr_scores = self._iqr_method(values)
        iso_outliers, iso_scores = self._isolation_forest_method(values)
        
        # Normalize scores
        zscore_norm = zscore_scores / np.max(zscore_scores) if np.max(zscore_scores) > 0 else zscore_scores
        iqr_norm = iqr_scores / np.max(iqr_scores) if np.max(iqr_scores) > 0 else iqr_scores
        iso_norm = iso_scores / np.max(iso_scores) if np.max(iso_scores) > 0 else iso_scores
        
        # Combined score (average)
        combined_scores = (zscore_norm + iqr_norm + iso_norm) / 3
        
        # Vote: outlier if at least 2 methods agree
        votes = zscore_outliers.astype(int) + iqr_outliers.astype(int) + iso_outliers.astype(int)
        outliers = votes >= 2
        
        return outliers, combined_scores
    
    def _generate_insights(self, df: pd.DataFrame, outlier_indices: np.ndarray,
                          outlier_values: np.ndarray, value_col: str) -> List[str]:
        """Generate insights about detected outliers."""
        insights = []
        
        if len(outlier_indices) == 0:
            insights.append("✅ No significant outliers detected in the data.")
            return insights
        
        # Outlier statistics
        mean_value = df[value_col].mean()
        outlier_pct = len(outlier_indices) / len(df) * 100
        
        insights.append(
            f"🔍 Detected {len(outlier_indices)} outliers ({outlier_pct:.1f}% of data)"
        )
        
        # High vs low outliers
        high_outliers = outlier_values > mean_value
        low_outliers = outlier_values < mean_value
        
        if np.sum(high_outliers) > np.sum(low_outliers):
            insights.append("📈 More high-value outliers - check for exceptional inflows or missing outflows")
        elif np.sum(low_outliers) > np.sum(high_outliers):
            insights.append("📉 More low-value outliers - check for exceptional outflows or missing inflows")
        
        # Day-of-week pattern
        if len(outlier_indices) >= 5:
            outlier_days = df.loc[outlier_indices, 'date'].dt.dayofweek
            day_counts = pd.Series(outlier_days).value_counts()
            
            if day_counts.max() > len(outlier_indices) * 0.4:
                peak_day = day_counts.idxmax()
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                            'Friday', 'Saturday', 'Sunday']
                insights.append(
                    f"📅 Outliers concentrated on {day_names[peak_day]}s - "
                    f"review recurring transactions on this day"
                )
        
        # Month-end pattern
        outlier_dates = df.loc[outlier_indices, 'date']
        month_end_outliers = ((outlier_dates + pd.Timedelta(days=1)).dt.month != outlier_dates.dt.month)
        
        if month_end_outliers.sum() > len(outlier_indices) * 0.3:
            insights.append("📆 Many outliers occur at month-end - typical for closing entries and settlements")
        
        return insights


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

@dataclass
class SHAPResult:
    """Container for SHAP analysis results."""
    feature_importance: Dict[str, float]
    shap_values: np.ndarray
    base_value: float
    top_features: List[Tuple[str, float]]
    feature_effects: Dict[str, str]
    insights: List[str]


class SHAPAnalyzer:
    """
    SHAP (SHapley Additive exPlanations) analysis for model interpretability.
    Explains which features drive cash flow predictions.
    """
    
    def __init__(self, max_features: int = 10):
        """
        Initialize SHAP Analyzer.
        
        Args:
            max_features: Maximum number of features to display
        """
        self.max_features = max_features
        self.model = None
        self.feature_names = None
    
    def analyze(self, df: pd.DataFrame, target_col: str = "net_cash_flow",
               sample_size: int = 100) -> SHAPResult:
        """
        Perform SHAP analysis using a Random Forest surrogate model.
        
        Args:
            df: DataFrame with features and target
            target_col: Target column name
            sample_size: Number of samples for SHAP calculation
        
        Returns:
            SHAPResult with feature importance and insights
        """
        import shap
        
        df = df.copy()
        
        # Define features
        feature_cols = [
            'day_of_week', 'day_of_month', 'month', 'quarter',
            'is_weekend', 'is_month_end', 'is_quarter_end'
        ]
        
        # Add lag features if enough data
        if len(df) > 30:
            for lag in [1, 7, 30]:
                df[f'lag_{lag}'] = df[target_col].shift(lag)
            feature_cols.extend(['lag_1', 'lag_7', 'lag_30'])
        
        # Add rolling features
        if len(df) > 7:
            df['rolling_mean_7'] = df[target_col].rolling(7).mean()
            df['rolling_std_7'] = df[target_col].rolling(7).std()
            feature_cols.extend(['rolling_mean_7', 'rolling_std_7'])
        
        # Remove NaN rows
        df = df.dropna(subset=feature_cols + [target_col])
        
        if len(df) < 50:
            return self._empty_result("Insufficient data for SHAP analysis")
        
        self.feature_names = feature_cols
        
        # Prepare data
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Train surrogate Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X, y)
        
        # Sample for SHAP (for performance)
        sample_idx = np.random.choice(len(X), min(sample_size, len(X)), replace=False)
        X_sample = X[sample_idx]
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        # Feature importance (mean absolute SHAP value)
        feature_importance = {}
        for i, feat in enumerate(feature_cols):
            feature_importance[feat] = np.mean(np.abs(shap_values[:, i]))
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:self.max_features]
        
        # Determine feature effects (positive or negative influence)
        feature_effects = {}
        for i, feat in enumerate(feature_cols):
            mean_shap = np.mean(shap_values[:, i])
            if mean_shap > 0:
                feature_effects[feat] = "positive"
            elif mean_shap < 0:
                feature_effects[feat] = "negative"
            else:
                feature_effects[feat] = "neutral"
        
        # Generate insights
        insights = self._generate_insights(top_features, feature_effects)
        
        return SHAPResult(
            feature_importance=feature_importance,
            shap_values=shap_values,
            base_value=explainer.expected_value,
            top_features=top_features,
            feature_effects=feature_effects,
            insights=insights
        )
    
    def _empty_result(self, message: str) -> SHAPResult:
        """Return empty result with message."""
        return SHAPResult(
            feature_importance={},
            shap_values=np.array([]),
            base_value=0,
            top_features=[],
            feature_effects={},
            insights=[message]
        )
    
    def _generate_insights(self, top_features: List[Tuple[str, float]],
                          feature_effects: Dict[str, str]) -> List[str]:
        """Generate actionable insights from SHAP analysis."""
        insights = []
        
        if not top_features:
            return ["Insufficient features for SHAP analysis"]
        
        # Top driver
        top_feat, top_imp = top_features[0]
        insights.append(f"🎯 Top predictor: '{top_feat}' - most influential factor in cash flow predictions")
        
        # Feature interpretations
        feature_descriptions = {
            'day_of_week': 'Day of week patterns significantly impact cash flow',
            'day_of_month': 'Day of month timing is important (likely payment cycles)',
            'month': 'Monthly seasonality affects predictions',
            'quarter': 'Quarterly patterns are influential',
            'is_weekend': 'Weekend vs weekday distinction matters',
            'is_month_end': 'Month-end timing is a key factor',
            'is_quarter_end': 'Quarter-end effects are significant',
            'lag_1': 'Yesterday\'s cash flow strongly predicts today\'s',
            'lag_7': 'Weekly patterns (7-day lag) are important',
            'lag_30': 'Monthly patterns (30-day lag) influence predictions',
            'rolling_mean_7': 'Recent 7-day average is a strong predictor',
            'rolling_std_7': 'Recent volatility affects predictions',
        }
        
        # Add top 3 feature insights
        for feat, imp in top_features[:3]:
            if feat in feature_descriptions:
                effect = feature_effects.get(feat, "neutral")
                effect_text = "increases" if effect == "positive" else "decreases" if effect == "negative" else "affects"
                insights.append(f"📊 {feature_descriptions[feat]} ({effect_text} forecast)")
        
        # Actionable recommendations
        lag_features = [f for f, _ in top_features if 'lag' in f]
        if lag_features:
            insights.append("💡 Historical patterns are strong predictors - ensure data quality in recent transactions")
        
        time_features = [f for f, _ in top_features if f in ['day_of_week', 'day_of_month', 'is_month_end']]
        if time_features:
            insights.append("💡 Calendar timing is crucial - consider scheduling major transactions strategically")
        
        return insights


# =============================================================================
# UNIFIED ANALYZER
# =============================================================================

class CashFlowAnalyzer:
    """
    Unified analyzer combining all analysis methods.
    Provides comprehensive insights for cash flow forecasting.
    """
    
    def __init__(self):
        self.mape_analyzer = None
        self.trend_analyzer = TrendAnalyzer()
        self.outlier_detector = OutlierDetector()
        self.shap_analyzer = SHAPAnalyzer()
    
    def full_analysis(self, actual_df: pd.DataFrame, 
                     forecast_df: pd.DataFrame = None,
                     horizon: str = "T+30") -> Dict[str, Any]:
        """
        Perform comprehensive analysis.
        
        Args:
            actual_df: DataFrame with actual values
            forecast_df: DataFrame with forecast values (optional)
            horizon: Forecast horizon for MAPE analysis
        
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # Trend Analysis
        print("Analyzing trends...")
        results["trend"] = self.trend_analyzer.analyze(actual_df)
        
        # Outlier Detection
        print("Detecting outliers...")
        results["outliers"] = self.outlier_detector.detect(actual_df)
        
        # SHAP Analysis
        print("Running SHAP analysis...")
        results["shap"] = self.shap_analyzer.analyze(actual_df)
        
        # MAPE Analysis (if forecast provided)
        if forecast_df is not None:
            print("Analyzing forecast accuracy...")
            self.mape_analyzer = MAPEAnalyzer(horizon)
            
            # Merge actual and forecast
            merged = actual_df.merge(
                forecast_df[['date', 'forecast']],
                on='date',
                how='inner'
            )
            
            if len(merged) > 0:
                results["mape"] = self.mape_analyzer.analyze(
                    merged['net_cash_flow'],
                    merged['forecast'],
                    merged['date']
                )
        
        # Compile all insights
        all_insights = []
        for key, result in results.items():
            if hasattr(result, 'insights'):
                all_insights.extend(result.insights)
        
        results["all_insights"] = all_insights
        
        return results


if __name__ == "__main__":
    # Test the analyzers
    from data_simulator import generate_sample_data
    
    print("Generating sample data...")
    data = generate_sample_data(periods=730)
    daily_cash = data["daily_cash_position"]
    
    print(f"Data shape: {daily_cash.shape}")
    
    # Run analysis
    analyzer = CashFlowAnalyzer()
    results = analyzer.full_analysis(daily_cash)
    
    print("\n=== TREND ANALYSIS ===")
    print(f"Direction: {results['trend'].trend_direction}")
    print(f"Insights: {results['trend'].insights}")
    
    print("\n=== OUTLIER DETECTION ===")
    print(f"Outliers found: {results['outliers'].total_outliers}")
    print(f"Insights: {results['outliers'].insights}")
    
    print("\n=== SHAP ANALYSIS ===")
    print(f"Top features: {results['shap'].top_features[:5]}")
    print(f"Insights: {results['shap'].insights}")
