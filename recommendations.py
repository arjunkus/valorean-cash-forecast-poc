"""
Recommendations Engine
======================
Translates analysis results into actionable business recommendations.
Provides prioritized insights for treasury and cash management.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from config import (
    BUSINESS_RULES, LIQUIDITY, MAPE_THRESHOLDS, TIME_HORIZONS
)


class Severity(Enum):
    """Recommendation severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"
    SUCCESS = "success"


class Category(Enum):
    """Recommendation categories."""
    LIQUIDITY = "Liquidity Management"
    FORECAST = "Forecast Accuracy"
    TREND = "Trend & Seasonality"
    OUTLIER = "Data Quality & Outliers"
    OPERATIONAL = "Operational Efficiency"
    INVESTMENT = "Investment Opportunity"


@dataclass
class Recommendation:
    """Container for a single recommendation."""
    id: str
    title: str
    description: str
    severity: Severity
    category: Category
    action: str
    impact: str
    priority: int  # 1 = highest
    metrics: Dict[str, Any] = field(default_factory=dict)
    related_dates: List[datetime] = field(default_factory=list)


class RecommendationEngine:
    """
    Generates prioritized recommendations from analysis results.
    Combines multiple signals into actionable treasury insights.
    """
    
    def __init__(self, daily_operating_expense: float = None):
        """
        Initialize the recommendation engine.
        
        Args:
            daily_operating_expense: Average daily operating expense for threshold calculations
        """
        self.daily_operating_expense = daily_operating_expense or 100000  # Default $100k/day
        self.recommendations: List[Recommendation] = []
    
    def generate_all_recommendations(
        self,
        actual_df: pd.DataFrame,
        forecast_results: Dict[str, pd.DataFrame],
        analysis_results: Dict[str, Any],
        mape_results: Dict[str, Any] = None
    ) -> List[Recommendation]:
        """
        Generate all recommendations from analysis results.
        
        Args:
            actual_df: DataFrame with actual cash positions
            forecast_results: Dictionary of forecasts by horizon
            analysis_results: Results from CashFlowAnalyzer
            mape_results: MAPE analysis results by horizon
        
        Returns:
            Sorted list of recommendations
        """
        self.recommendations = []
        
        # Calculate daily operating expense if not provided
        if self.daily_operating_expense is None:
            if 'outflow' in actual_df.columns:
                self.daily_operating_expense = actual_df['outflow'].mean()
        
        # Liquidity recommendations
        self._analyze_liquidity(actual_df, forecast_results)
        
        # Forecast accuracy recommendations
        if mape_results:
            self._analyze_forecast_accuracy(mape_results)
        
        # Trend recommendations
        if 'trend' in analysis_results:
            self._analyze_trend(analysis_results['trend'], actual_df)
        
        # Outlier recommendations
        if 'outliers' in analysis_results:
            self._analyze_outliers(analysis_results['outliers'], actual_df)
        
        # SHAP-based recommendations
        if 'shap' in analysis_results:
            self._analyze_shap(analysis_results['shap'])
        
        # Sort by priority
        self.recommendations.sort(key=lambda x: x.priority)
        
        return self.recommendations
    
    def _analyze_liquidity(self, actual_df: pd.DataFrame, 
                          forecast_results: Dict[str, pd.DataFrame]):
        """Generate liquidity-related recommendations."""
        
        # Current cash position
        current_cash = actual_df['cash_position'].iloc[-1]
        avg_daily_outflow = actual_df['outflow'].mean()
        
        # Days of cash on hand
        days_of_cash = current_cash / avg_daily_outflow if avg_daily_outflow > 0 else float('inf')
        
        # Critical: Very low cash
        if days_of_cash < LIQUIDITY.minimum_cash_days * (1 - LIQUIDITY.critical_threshold_pct):
            self.recommendations.append(Recommendation(
                id="LIQ001",
                title="Critical Cash Position Alert",
                description=f"Current cash position ({days_of_cash:.0f} days of operating expenses) "
                           f"is below critical threshold ({LIQUIDITY.minimum_cash_days * (1 - LIQUIDITY.critical_threshold_pct):.0f} days).",
                severity=Severity.CRITICAL,
                category=Category.LIQUIDITY,
                action="Immediate action required: Draw on credit facilities, accelerate AR collection, "
                       "or delay non-essential disbursements.",
                impact=f"Risk of cash shortfall within {days_of_cash:.0f} days without intervention.",
                priority=1,
                metrics={"days_of_cash": days_of_cash, "current_cash": current_cash}
            ))
        
        # Warning: Low cash
        elif days_of_cash < LIQUIDITY.minimum_cash_days:
            self.recommendations.append(Recommendation(
                id="LIQ002",
                title="Low Cash Position Warning",
                description=f"Cash position ({days_of_cash:.0f} days) is below minimum target "
                           f"({LIQUIDITY.minimum_cash_days} days).",
                severity=Severity.WARNING,
                category=Category.LIQUIDITY,
                action="Review upcoming payment schedule, identify deferrable items, "
                       "and prepare contingency plans.",
                impact="Reduced financial flexibility and potential liquidity stress.",
                priority=2,
                metrics={"days_of_cash": days_of_cash, "current_cash": current_cash}
            ))
        
        # Excess cash opportunity
        elif days_of_cash > LIQUIDITY.excess_cash_days:
            excess_amount = current_cash - (LIQUIDITY.target_cash_days * avg_daily_outflow)
            self.recommendations.append(Recommendation(
                id="LIQ003",
                title="Excess Cash Investment Opportunity",
                description=f"Cash position ({days_of_cash:.0f} days) exceeds optimal level. "
                           f"${excess_amount:,.0f} could be deployed more productively.",
                severity=Severity.INFO,
                category=Category.INVESTMENT,
                action="Consider short-term investments, debt reduction, or strategic reserves allocation.",
                impact=f"Potential interest income or debt reduction savings of ${excess_amount * 0.04:,.0f}/year at 4%.",
                priority=5,
                metrics={"excess_amount": excess_amount, "days_of_cash": days_of_cash}
            ))
        
        # Forecast-based liquidity alerts
        for horizon, forecast_df in forecast_results.items():
            if 'forecast' in forecast_df.columns:
                # Check for projected negative cash flow periods
                cumulative_forecast = forecast_df['forecast'].cumsum()
                
                # Find potential shortfall
                min_forecast = cumulative_forecast.min()
                if min_forecast < -current_cash * 0.5:  # More than 50% of current cash
                    shortfall_date = forecast_df.loc[cumulative_forecast.idxmin(), 'date']
                    
                    self.recommendations.append(Recommendation(
                        id=f"LIQ004_{horizon}",
                        title=f"Projected Cash Shortfall ({horizon})",
                        description=f"Forecast indicates potential cash pressure around "
                                   f"{shortfall_date.strftime('%Y-%m-%d')} based on {horizon} projection.",
                        severity=Severity.WARNING,
                        category=Category.LIQUIDITY,
                        action="Review forecast assumptions, identify mitigation options, "
                               "and consider pre-arranging credit facilities.",
                        impact="Proactive management can prevent liquidity crisis.",
                        priority=3,
                        metrics={"shortfall_amount": min_forecast},
                        related_dates=[shortfall_date]
                    ))
    
    def _analyze_forecast_accuracy(self, mape_results: Dict[str, Any]):
        """Generate forecast accuracy recommendations."""
        
        for horizon, metrics in mape_results.items():
            if horizon == "daily_analysis":
                continue
            
            if not isinstance(metrics, dict):
                continue
                
            mape = metrics.get('mape', 0)
            rating = metrics.get('rating', 'Unknown')
            
            thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
            
            # Poor accuracy
            if mape > thresholds["poor"]:
                self.recommendations.append(Recommendation(
                    id=f"FORE001_{horizon}",
                    title=f"Poor Forecast Accuracy ({horizon})",
                    description=f"MAPE of {mape:.1f}% significantly exceeds acceptable threshold "
                               f"({thresholds['acceptable']}%). Current rating: {rating}.",
                    severity=Severity.WARNING,
                    category=Category.FORECAST,
                    action="Review input data quality, check for missing transactions, "
                           "and consider model retraining with recent data.",
                    impact="Inaccurate forecasts lead to suboptimal cash management decisions.",
                    priority=3,
                    metrics={"mape": mape, "rating": rating, "threshold": thresholds["poor"]}
                ))
            
            # Acceptable but could improve
            elif mape > thresholds["good"]:
                self.recommendations.append(Recommendation(
                    id=f"FORE002_{horizon}",
                    title=f"Forecast Accuracy Could Improve ({horizon})",
                    description=f"MAPE of {mape:.1f}% is acceptable but has room for improvement. "
                               f"Current rating: {rating}.",
                    severity=Severity.INFO,
                    category=Category.FORECAST,
                    action="Consider adding external factors (holidays, business events) "
                           "and review category-level forecasts.",
                    impact="Improved accuracy enables better cash positioning.",
                    priority=6,
                    metrics={"mape": mape, "rating": rating}
                ))
            
            # Excellent accuracy - positive reinforcement
            elif mape <= thresholds["excellent"]:
                self.recommendations.append(Recommendation(
                    id=f"FORE003_{horizon}",
                    title=f"Excellent Forecast Accuracy ({horizon})",
                    description=f"MAPE of {mape:.1f}% demonstrates excellent prediction quality. "
                               f"Rating: {rating}.",
                    severity=Severity.SUCCESS,
                    category=Category.FORECAST,
                    action="Maintain current model and data practices. "
                           "Consider using this model as baseline for other horizons.",
                    impact="High accuracy enables confident cash management decisions.",
                    priority=10,
                    metrics={"mape": mape, "rating": rating}
                ))
        
        # Daily pattern analysis
        if "daily_analysis" in mape_results:
            for horizon, daily_mape in mape_results["daily_analysis"].items():
                if daily_mape:
                    max_day = max(daily_mape, key=daily_mape.get)
                    min_day = min(daily_mape, key=daily_mape.get)
                    
                    if daily_mape[max_day] > daily_mape[min_day] * 2:
                        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                    'Friday', 'Saturday', 'Sunday']
                        
                        self.recommendations.append(Recommendation(
                            id=f"FORE004_{horizon}",
                            title=f"Day-Specific Forecast Issues ({horizon})",
                            description=f"{day_names[max_day]} shows significantly higher errors "
                                       f"({daily_mape[max_day]:.1f}%) vs {day_names[min_day]} "
                                       f"({daily_mape[min_day]:.1f}%).",
                            severity=Severity.INFO,
                            category=Category.FORECAST,
                            action=f"Review {day_names[max_day]} transactions for patterns "
                                   f"not captured by the model.",
                            impact="Day-specific adjustments could improve overall accuracy.",
                            priority=7,
                            metrics={"worst_day": day_names[max_day], "best_day": day_names[min_day]}
                        ))
    
    def _analyze_trend(self, trend_result: Any, actual_df: pd.DataFrame):
        """Generate trend-related recommendations."""
        
        if not hasattr(trend_result, 'trend_direction'):
            return
        
        direction = trend_result.trend_direction
        slope = trend_result.trend_slope
        
        # Declining trend
        if direction == "decreasing":
            monthly_decline = abs(slope) * 30
            self.recommendations.append(Recommendation(
                id="TREND001",
                title="Declining Cash Flow Trend Detected",
                description=f"Net cash flow shows a declining trend of approximately "
                           f"${monthly_decline:,.0f} per month.",
                severity=Severity.WARNING,
                category=Category.TREND,
                action="Analyze outflow categories for cost reduction opportunities. "
                       "Review AR collection efficiency and payment terms.",
                impact="Continued decline could lead to liquidity pressure in coming months.",
                priority=3,
                metrics={"monthly_decline": monthly_decline, "slope": slope}
            ))
        
        # Increasing trend (opportunity)
        elif direction == "increasing":
            monthly_increase = slope * 30
            self.recommendations.append(Recommendation(
                id="TREND002",
                title="Positive Cash Flow Trend",
                description=f"Net cash flow shows an increasing trend of approximately "
                           f"${monthly_increase:,.0f} per month.",
                severity=Severity.SUCCESS,
                category=Category.TREND,
                action="Consider deploying excess cash strategically. "
                       "Review investment policies for optimization.",
                impact="Positive trend provides opportunity for strategic initiatives.",
                priority=8,
                metrics={"monthly_increase": monthly_increase, "slope": slope}
            ))
        
        # Change points
        if hasattr(trend_result, 'change_points') and trend_result.change_points:
            recent_changes = [cp for cp in trend_result.change_points 
                            if cp > datetime.now() - timedelta(days=90)]
            
            if recent_changes:
                self.recommendations.append(Recommendation(
                    id="TREND003",
                    title="Recent Trend Change Detected",
                    description=f"Significant trend changes detected in the past 90 days "
                               f"(most recent: {recent_changes[-1].strftime('%Y-%m-%d')}).",
                    severity=Severity.INFO,
                    category=Category.TREND,
                    action="Investigate cause of trend changes. Consider if this represents "
                           "a new normal or temporary anomaly.",
                    impact="Understanding trend changes improves forecast reliability.",
                    priority=6,
                    related_dates=recent_changes
                ))
    
    def _analyze_outliers(self, outlier_result: Any, actual_df: pd.DataFrame):
        """Generate outlier-related recommendations."""
        
        if not hasattr(outlier_result, 'total_outliers'):
            return
        
        total_outliers = outlier_result.total_outliers
        outlier_pct = outlier_result.outlier_pct
        
        if total_outliers == 0:
            self.recommendations.append(Recommendation(
                id="OUT001",
                title="Clean Data Quality",
                description="No significant outliers detected in cash flow data.",
                severity=Severity.SUCCESS,
                category=Category.OUTLIER,
                action="Maintain current data governance practices.",
                impact="Clean data supports reliable forecasting.",
                priority=10
            ))
            return
        
        # High outlier count
        if outlier_pct > 5:
            self.recommendations.append(Recommendation(
                id="OUT002",
                title="High Outlier Rate Detected",
                description=f"{total_outliers} outliers detected ({outlier_pct:.1f}% of data), "
                           f"exceeding typical threshold of 2-3%.",
                severity=Severity.WARNING,
                category=Category.OUTLIER,
                action="Review flagged transactions for data entry errors, "
                       "miscategorized items, or exceptional transactions requiring explanation.",
                impact="High outlier rates can distort forecasts and hide true patterns.",
                priority=4,
                metrics={"total_outliers": total_outliers, "outlier_pct": outlier_pct}
            ))
        
        # Moderate outliers
        elif total_outliers > 10:
            self.recommendations.append(Recommendation(
                id="OUT003",
                title="Outliers Require Review",
                description=f"{total_outliers} outliers detected. "
                           f"Some may represent legitimate exceptional transactions.",
                severity=Severity.INFO,
                category=Category.OUTLIER,
                action="Review top outliers by magnitude. Categorize as data quality issues "
                       "or legitimate exceptions.",
                impact="Proper handling improves model training and forecast reliability.",
                priority=6,
                metrics={"total_outliers": total_outliers}
            ))
        
        # Check for patterns in outliers
        if hasattr(outlier_result, 'outlier_dates') and outlier_result.outlier_dates:
            outlier_dates = pd.Series(outlier_result.outlier_dates)
            
            # Month-end concentration
            month_end = ((outlier_dates + pd.Timedelta(days=1)).dt.month != outlier_dates.dt.month)
            if month_end.sum() > len(outlier_dates) * 0.4:
                self.recommendations.append(Recommendation(
                    id="OUT004",
                    title="Month-End Outlier Pattern",
                    description="Significant concentration of outliers at month-end periods.",
                    severity=Severity.INFO,
                    category=Category.OUTLIER,
                    action="Review month-end closing procedures. Consider if model needs "
                           "explicit month-end adjustments.",
                    impact="Addressing month-end patterns improves forecast accuracy.",
                    priority=7
                ))
    
    def _analyze_shap(self, shap_result: Any):
        """Generate SHAP-based recommendations."""
        
        if not hasattr(shap_result, 'top_features') or not shap_result.top_features:
            return
        
        top_features = shap_result.top_features[:3]
        
        # Feature-specific recommendations
        feature_actions = {
            'day_of_week': (
                "Day of Week Drives Predictions",
                "Consider day-specific cash management strategies and payment scheduling."
            ),
            'is_month_end': (
                "Month-End Timing Critical",
                "Optimize month-end cash positioning and prepare for settlement spikes."
            ),
            'lag_1': (
                "Strong Daily Momentum",
                "Yesterday's cash flow strongly predicts today. Ensure daily data accuracy."
            ),
            'rolling_mean_7': (
                "Recent Trends Matter",
                "7-day rolling average is highly predictive. Monitor for trend changes."
            ),
            'is_weekend': (
                "Weekend Pattern Important",
                "Weekend/weekday distinction significant. Plan around weekly cycles."
            )
        }
        
        for feat, importance in top_features:
            if feat in feature_actions:
                title, action = feature_actions[feat]
                self.recommendations.append(Recommendation(
                    id=f"SHAP_{feat.upper()}",
                    title=title,
                    description=f"'{feat}' is a top-3 driver of cash flow predictions "
                               f"(importance score: {importance:.3f}).",
                    severity=Severity.INFO,
                    category=Category.OPERATIONAL,
                    action=action,
                    impact="Understanding key drivers enables targeted improvements.",
                    priority=7,
                    metrics={"feature": feat, "importance": importance}
                ))
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all recommendations."""
        summary = {
            "total": len(self.recommendations),
            "by_severity": {},
            "by_category": {},
            "top_actions": [],
            "critical_count": 0,
            "warning_count": 0
        }
        
        for rec in self.recommendations:
            # By severity
            sev = rec.severity.value
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1
            
            # By category
            cat = rec.category.value
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1
            
            # Counts
            if rec.severity == Severity.CRITICAL:
                summary["critical_count"] += 1
            elif rec.severity == Severity.WARNING:
                summary["warning_count"] += 1
        
        # Top actions (first 5 by priority)
        summary["top_actions"] = [
            {"title": r.title, "action": r.action, "severity": r.severity.value}
            for r in self.recommendations[:5]
        ]
        
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert recommendations to DataFrame."""
        records = []
        for rec in self.recommendations:
            records.append({
                "ID": rec.id,
                "Priority": rec.priority,
                "Severity": rec.severity.value,
                "Category": rec.category.value,
                "Title": rec.title,
                "Description": rec.description,
                "Action": rec.action,
                "Impact": rec.impact
            })
        
        return pd.DataFrame(records)


def generate_recommendations(
    actual_df: pd.DataFrame,
    forecast_results: Dict[str, pd.DataFrame],
    analysis_results: Dict[str, Any],
    mape_results: Dict[str, Any] = None,
    daily_operating_expense: float = None
) -> Tuple[List[Recommendation], Dict[str, Any]]:
    """
    Convenience function to generate all recommendations.
    
    Returns:
        Tuple of (recommendations list, summary dict)
    """
    engine = RecommendationEngine(daily_operating_expense)
    recommendations = engine.generate_all_recommendations(
        actual_df, forecast_results, analysis_results, mape_results
    )
    summary = engine.get_summary()
    
    return recommendations, summary


if __name__ == "__main__":
    # Test the recommendation engine
    from data_simulator import generate_sample_data
    from models import run_backtest
    from analysis import CashFlowAnalyzer
    
    print("Generating sample data...")
    data = generate_sample_data(periods=730)
    daily_cash = data["daily_cash_position"]
    
    print("Running backtest...")
    mape_results, forecaster, forecasts = run_backtest(daily_cash)
    
    print("Running analysis...")
    analyzer = CashFlowAnalyzer()
    analysis_results = analyzer.full_analysis(daily_cash)
    
    print("Generating recommendations...")
    recommendations, summary = generate_recommendations(
        daily_cash, forecasts, analysis_results, mape_results
    )
    
    print(f"\n=== RECOMMENDATION SUMMARY ===")
    print(f"Total recommendations: {summary['total']}")
    print(f"Critical: {summary['critical_count']}")
    print(f"Warnings: {summary['warning_count']}")
    
    print("\n=== TOP RECOMMENDATIONS ===")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"\n{i}. [{rec.severity.value.upper()}] {rec.title}")
        print(f"   Action: {rec.action}")
