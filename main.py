#!/usr/bin/env python3
"""
Cash Forecasting Intelligence System
=====================================
Enterprise-grade multi-horizon cash flow forecasting with AI-powered insights.

Usage:
    python main.py --mode dashboard    # Launch interactive dashboard
    python main.py --mode backtest     # Run backtest and show results
    python main.py --mode generate     # Generate sample data only
    python main.py --mode full         # Run full pipeline and save results
"""

import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "dashboard.py"
    subprocess.run(["streamlit", "run", str(dashboard_path), "--server.headless", "true"])


def run_backtest(periods: int = 730, output_dir: str = None):
    """Run backtest and display/save results."""
    print("=" * 60)
    print("CASH FORECASTING INTELLIGENCE SYSTEM")
    print("Backtest Mode")
    print("=" * 60)
    
    from data_simulator import generate_sample_data
    from models import run_backtest as model_backtest
    from analysis import CashFlowAnalyzer
    from recommendations import generate_recommendations
    
    # Generate data
    print("\n📊 Generating sample SAP FQM data...")
    data = generate_sample_data(periods=periods)
    daily_cash = data["daily_cash_position"]
    
    print(f"   ✓ Generated {len(daily_cash)} days of data")
    print(f"   ✓ Date range: {daily_cash['date'].min().date()} to {daily_cash['date'].max().date()}")
    print(f"   ✓ Total transactions: {len(data['fqm_flow']):,}")
    
    # Run backtest
    print("\n🔮 Training forecasting models...")
    mape_results, forecaster, forecasts = model_backtest(daily_cash)
    
    print("\n📈 MAPE Results by Horizon:")
    print("-" * 40)
    for horizon, metrics in mape_results.items():
        if horizon != "daily_analysis" and isinstance(metrics, dict):
            mape = metrics.get('mape', 0)
            rating = metrics.get('rating', 'N/A')
            print(f"   {horizon:8s}: {mape:6.2f}% MAPE ({rating})")
    
    # Run analysis
    print("\n🔍 Running comprehensive analysis...")
    analyzer = CashFlowAnalyzer()
    analysis_results = analyzer.full_analysis(daily_cash)
    
    # Trend summary
    trend = analysis_results.get('trend')
    if trend:
        print(f"\n📈 Trend Analysis:")
        print(f"   Direction: {trend.trend_direction}")
        print(f"   Monthly change: ${trend.trend_slope * 30:,.0f}")
    
    # Outlier summary
    outliers = analysis_results.get('outliers')
    if outliers:
        print(f"\n⚠️ Outlier Detection:")
        print(f"   Total outliers: {outliers.total_outliers}")
        print(f"   Percentage: {outliers.outlier_pct:.1f}%")
    
    # Generate recommendations
    print("\n💡 Generating recommendations...")
    recommendations, summary = generate_recommendations(
        daily_cash, forecasts, analysis_results, mape_results
    )
    
    print(f"\n📋 Recommendation Summary:")
    print(f"   Total: {summary['total']}")
    print(f"   Critical: {summary['critical_count']}")
    print(f"   Warnings: {summary['warning_count']}")
    
    print("\n🎯 Top 5 Actions:")
    print("-" * 60)
    for i, rec in enumerate(recommendations[:5], 1):
        severity = rec.severity.value.upper()
        print(f"\n{i}. [{severity}] {rec.title}")
        print(f"   Action: {rec.action[:80]}...")
    
    # Save results if output directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save data
        daily_cash.to_csv(output_path / "daily_cash_position.csv", index=False)
        data['fqm_flow'].to_csv(output_path / "fqm_flow.csv", index=False)
        
        # Save forecasts
        for horizon, forecast_df in forecasts.items():
            forecast_df.to_csv(output_path / f"forecast_{horizon.replace('+', '_')}.csv", index=False)
        
        # Save recommendations
        from recommendations import RecommendationEngine
        engine = RecommendationEngine()
        engine.recommendations = recommendations
        engine.to_dataframe().to_csv(output_path / "recommendations.csv", index=False)
        
        # Save summary
        with open(output_path / "summary.json", "w") as f:
            json.dump({
                "run_date": datetime.now().isoformat(),
                "periods": periods,
                "mape_results": {k: v for k, v in mape_results.items() if k != "daily_analysis"},
                "recommendation_summary": summary,
                "trend_direction": trend.trend_direction if trend else None,
                "outlier_count": outliers.total_outliers if outliers else 0,
            }, f, indent=2, default=str)
        
        print(f"\n✅ Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)
    
    return {
        "data": data,
        "mape_results": mape_results,
        "forecasts": forecasts,
        "analysis": analysis_results,
        "recommendations": recommendations
    }


def generate_data_only(periods: int = 730, output_dir: str = "./output"):
    """Generate sample data and save to files."""
    print("Generating sample SAP FQM data...")
    
    from data_simulator import generate_sample_data
    
    data = generate_sample_data(periods=periods)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save all datasets
    for name, df in data.items():
        filepath = output_path / f"{name}.csv"
        df.to_csv(filepath, index=False)
        print(f"   ✓ Saved {name}: {len(df):,} records → {filepath}")
    
    print(f"\n✅ All data saved to: {output_path}")
    
    return data


def run_full_pipeline(periods: int = 730, output_dir: str = "./output"):
    """Run the complete pipeline and save all artifacts."""
    print("Running full pipeline...")
    return run_backtest(periods=periods, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Cash Forecasting Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode dashboard              # Launch interactive dashboard
  python main.py --mode backtest               # Run backtest with defaults
  python main.py --mode backtest --periods 365 # Run with 1 year of data
  python main.py --mode full --output ./results # Full pipeline with output
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["dashboard", "backtest", "generate", "full"],
        default="backtest",
        help="Execution mode (default: backtest)"
    )
    
    parser.add_argument(
        "--periods",
        type=int,
        default=730,
        help="Number of days of historical data to generate (default: 730)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output directory for results (default: ./output)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "backtest":
        run_backtest(periods=args.periods)
    elif args.mode == "generate":
        generate_data_only(periods=args.periods, output_dir=args.output)
    elif args.mode == "full":
        run_full_pipeline(periods=args.periods, output_dir=args.output)


if __name__ == "__main__":
    main()
