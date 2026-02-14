"""
Cash Forecasting Intelligence Dashboard v2
==========================================
Focused on detailed analysis tabs with simple overview.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Cash Forecasting Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# IMPORTS
# =============================================================================
try:
    from config import TIME_HORIZONS, MAPE_THRESHOLDS
    from data_simulator_realistic import generate_sample_data
    from models_prophet_v2 import ProphetCashForecaster, run_backtest
    from analysis_v2 import SHAPAnalyzer, OutlierDetector
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'daily_cash' not in st.session_state:
        st.session_state.daily_cash = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = None
    if 'shap_results' not in st.session_state:
        st.session_state.shap_results = None
    if 'outlier_results' not in st.session_state:
        st.session_state.outlier_results = None


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(periods: int = 730):
    data = generate_sample_data(periods=periods)
    return data['daily_cash_position']


def train_models(daily_cash: pd.DataFrame, test_size: int = 90):
    results, forecaster, forecasts = run_backtest(daily_cash, test_size=test_size)
    return results, forecaster, forecasts


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    st.sidebar.title("💰 Cash Forecasting")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📊 Data Settings")
    periods = st.sidebar.slider("Historical Data (days)", 365, 1095, 730, 30)
    test_size = st.sidebar.slider("Test Period (days)", 30, 180, 90, 10)
    
    if st.sidebar.button("🚀 Load Data & Train Models", type="primary", use_container_width=True):
        with st.spinner("Loading data and training models..."):
            st.session_state.daily_cash = load_data(periods)
            
            results, forecaster, forecasts = train_models(
                st.session_state.daily_cash, 
                test_size=test_size
            )
            
            st.session_state.forecaster = forecaster
            st.session_state.forecasts = forecasts
            st.session_state.backtest_results = results
            
            # SHAP analysis
            shap_analyzer = SHAPAnalyzer(forecaster)
            st.session_state.shap_results = shap_analyzer.analyze()
            
            # Outlier detection
            outlier_detector = OutlierDetector()
            st.session_state.outlier_results = outlier_detector.detect(st.session_state.daily_cash.copy())
            st.session_state.outlier_summary = outlier_detector.get_outlier_summary()
            
            st.session_state.data_loaded = True
            
        st.sidebar.success("✅ Models trained!")
    
    st.sidebar.markdown("---")
    
    if st.session_state.data_loaded:
        t0 = st.session_state.forecaster.last_actual_date
        balance = st.session_state.forecaster.last_actual_closing_balance
        st.sidebar.metric("T0 Date", t0.strftime('%Y-%m-%d'))
        st.sidebar.metric("T0 Balance", f"${balance:,.0f}")
    
    return periods, test_size


# =============================================================================
# OVERVIEW TAB (Simplified)
# =============================================================================
def render_overview_tab():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' in the sidebar to get started.")
        return
    
    forecaster = st.session_state.forecaster
    forecasts = st.session_state.forecasts
    results = st.session_state.backtest_results
    
    # Key Metrics Header
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "T0 Balance", 
            f"${forecaster.last_actual_closing_balance:,.0f}",
            help="Current cash position as of last actual date"
        )
    
    with col2:
        st.metric(
            "T0 Date",
            forecaster.last_actual_date.strftime('%Y-%m-%d'),
            help="Last date with actual data"
        )
    
    with col3:
        if 'T+7' in results:
            st.metric(
                "T+7 Accuracy",
                f"{results['T+7']['balance_mape']:.1f}% error",
                results['T+7']['rating']
            )

    with col4:
        if 'T+30' in results:
            st.metric(
                "T+30 Accuracy",
                f"{results['T+30']['balance_mape']:.1f}% error",
                results['T+30']['rating']
            )
    
    st.markdown("---")
    
    # Forecast Summary Table
    st.subheader("📈 Forecast Summary")
    
    summary_data = []
    for horizon in ['T+7', 'T+30', 'T+90']:
        if horizon in forecasts:
            fcast = forecasts[horizon]
            mape = results[horizon]['balance_mape'] if horizon in results else 0
            rating = results[horizon]['rating'] if horizon in results else 'N/A'
            
            summary_data.append({
                'Horizon': horizon,
                'Banking Days': len(fcast),
                'Opening (T0)': f"${forecaster.last_actual_closing_balance:,.0f}",
                'Total Inflows': f"${fcast['forecast_inflow'].sum():,.0f}",
                'Total Outflows': f"${fcast['forecast_outflow'].sum():,.0f}",
                'Net Change': f"${fcast['forecast_net'].sum():+,.0f}",
                'Closing Balance': f"${fcast['closing_balance'].iloc[-1]:,.0f}",
                'Accuracy': f"{mape:.1f}% error",
                'Rating': rating
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Quick Links
    st.subheader("🔗 Detailed Analysis")
    st.markdown("""
    - **📈 Forecasts Tab**: Day-by-day forecast with inflows, outflows, and balances
    - **🎯 Accuracy Tab**: Error analysis by horizon day - see how accuracy changes over time
    - **🔍 Drivers Tab**: What's driving the forecasts (weekly patterns, monthly cycles, etc.)
    - **⚠️ Outliers Tab**: Unusual cash flow days flagged for review
    """)


# =============================================================================
# FORECASTS TAB
# =============================================================================
def render_forecasts_tab():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' to view forecasts.")
        return
    
    forecasts = st.session_state.forecasts
    forecaster = st.session_state.forecaster
    results = st.session_state.backtest_results
    
    horizon = st.selectbox("Select Forecast Horizon", ['T+7', 'T+30', 'T+90'], key='fcast_hz')
    
    if horizon not in forecasts:
        st.warning(f"No forecast available for {horizon}")
        return
    
    fcast = forecasts[horizon].sort_values('date', ascending=True).copy()
    
    # Treasury Summary
    st.markdown("### Treasury Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    t0_balance = forecaster.last_actual_closing_balance
    total_inflows = fcast['forecast_inflow'].sum()
    total_outflows = fcast['forecast_outflow'].sum()
    final_balance = fcast['closing_balance'].iloc[-1]
    mape = results[horizon]['balance_mape'] if horizon in results else 0
    rating = results[horizon]['rating'] if horizon in results else 'N/A'
    
    with col1:
        st.metric("Opening (T0)", f"${t0_balance:,.0f}")
    with col2:
        st.metric("+ Receipts", f"${total_inflows:,.0f}")
    with col3:
        st.metric("− Payments", f"${total_outflows:,.0f}")
    with col4:
        st.metric("= Closing", f"${final_balance:,.0f}")
    with col5:
        st.metric("Accuracy", f"{mape:.1f}% error", rating)
    
    st.markdown("---")
    
    # Charts
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=('Closing Balance by Day', 'Daily Inflows / Outflows'))
    
    fig.add_trace(go.Scatter(
        x=fcast['date'], y=fcast['closing_balance'],
        mode='lines+markers', name='Closing Balance',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%b %d}<br>$%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=fcast['date'], y=fcast['forecast_inflow'],
        name='Inflows', marker_color='#2ca02c',
        hovertemplate='%{x|%b %d}<br>+$%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Bar(
        x=fcast['date'], y=-fcast['forecast_outflow'],
        name='Outflows', marker_color='#d62728',
        hovertemplate='%{x|%b %d}<br>-$%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    fig.update_layout(
        height=500, 
        barmode='relative', 
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    fig.update_yaxes(tickformat='$,.0f')
    fig.update_xaxes(tickformat='%b %d')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("### Forecast Data")
    display_df = fcast[['date', 'horizon_day', 'day_name', 'opening_balance',
                        'forecast_inflow', 'forecast_outflow', 'forecast_net', 'closing_balance']].copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df.columns = ['Date', 'Day', 'Day Name', 'Opening', 'Inflow', 'Outflow', 'Net', 'Closing']
    
    for col in ['Opening', 'Inflow', 'Outflow', 'Net', 'Closing']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# =============================================================================
# ACCURACY TAB
# =============================================================================
def render_accuracy_tab():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' to view accuracy.")
        return
    
    results = st.session_state.backtest_results
    
    horizon = st.selectbox("Select Horizon", ['T+7', 'T+30', 'T+90'], key='acc_hz')
    
    if horizon not in results:
        st.warning(f"No results for {horizon}")
        return
    
    # Summary metrics
    st.markdown("### Overall Accuracy")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Balance Accuracy", f"{results[horizon]['balance_mape']:.2f}% error")
    with col2:
        st.metric("Inflow Accuracy", f"{results[horizon]['inflow_mape']:.2f}% error")
    with col3:
        st.metric("Outflow Accuracy", f"{results[horizon]['outflow_mape']:.2f}% error")
    with col4:
        st.metric("Rating", results[horizon]['rating'])
    
    st.markdown("---")
    
    # Accuracy by horizon day chart
    st.markdown("### Accuracy by Horizon Day")
    st.caption("Shows how forecast accuracy degrades as we look further into the future")
    
    if 'mape_by_horizon_day' in results[horizon]:
        mape_df = results[horizon]['mape_by_horizon_day']
        
        # Color based on MAPE value
        colors = ['#4CAF50' if m <= 2 else '#FFC107' if m <= 5 else '#F44336' 
                  for m in mape_df['balance_mape']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=mape_df['horizon_day'],
            y=mape_df['balance_mape'],
            marker_color=colors,
            hovertemplate='Day %{x}<br>Error: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_hline(y=2, line_dash="dot", line_color="green", line_width=1,
                     annotation_text="Excellent (2%)", annotation_position="right")
        fig.add_hline(y=5, line_dash="dot", line_color="orange", line_width=1,
                     annotation_text="Good (5%)", annotation_position="right")
        
        fig.update_layout(
            xaxis_title="Horizon Day",
            yaxis_title="Balance Error (%)",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.markdown("### Accuracy by Day (Detail)")
        table_df = mape_df[['horizon_day', 'balance_mape', 'inflow_mape', 'outflow_mape']].copy()
        table_df.columns = ['Horizon Day', 'Balance Error %', 'Inflow Error %', 'Outflow Error %']
        table_df['Balance Error %'] = table_df['Balance Error %'].apply(lambda x: f"{x:.2f}%")
        table_df['Inflow Error %'] = table_df['Inflow Error %'].apply(lambda x: f"{x:.2f}%")
        table_df['Outflow Error %'] = table_df['Outflow Error %'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(table_df, use_container_width=True, hide_index=True)


# =============================================================================
# SHAP TAB
# =============================================================================
def render_shap_tab():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' to view driver analysis.")
        return

    shap_results = st.session_state.shap_results

    if not shap_results:
        st.warning("Driver analysis not available.")
        return

    st.markdown("### Key Driver Analysis")
    st.caption("What's driving the cash flow forecasts?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Inflow Model")
        if 'inflow' in shap_results:
            importance_df = shap_results['inflow']['importance']
            fig = go.Figure(go.Bar(
                x=importance_df['importance_pct'],
                y=importance_df['component'],
                orientation='h',
                marker_color='#2ca02c',
                hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
            ))
            fig.update_layout(
                height=300, 
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance %",
                margin=dict(l=100, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📉 Outflow Model")
        if 'outflow' in shap_results:
            importance_df = shap_results['outflow']['importance']
            fig = go.Figure(go.Bar(
                x=importance_df['importance_pct'],
                y=importance_df['component'],
                orientation='h',
                marker_color='#d62728',
                hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
            ))
            fig.update_layout(
                height=300, 
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance %",
                margin=dict(l=100, r=20, t=20, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Component Interpretation")
    
    st.markdown("""
    | Component | Description | Typical Pattern |
    |-----------|-------------|-----------------|
    | **Weekly** | Day-of-week effects | Friday AP runs, Monday AR collections |
    | **Biweekly** | Two-week cycles | Bi-weekly payroll (15th & month-end) |
    | **Monthly** | Day-of-month patterns | Month-end settlements |
    | **Month-end** | End-of-month spike | AR collection push before close |
    | **Quarterly** | Quarterly patterns | Tax payments (Apr, Jun, Sep, Dec) |
    | **Yearly** | Annual seasonality | Holiday slowdowns, fiscal year patterns |
    | **Trend** | Long-term direction | Growth or decline trajectory |
    """)


# =============================================================================
# OUTLIERS TAB
# =============================================================================
def render_outliers_tab():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' to view outliers.")
        return
    
    outlier_df = st.session_state.outlier_results
    outlier_summary = st.session_state.outlier_summary
    daily_cash = st.session_state.daily_cash.sort_values('date', ascending=True).reset_index(drop=True)
    
    if outlier_df is None:
        st.warning("Outlier analysis not available.")
        return
    
    # Sort outlier_df same way
    outlier_df = outlier_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    st.markdown("### Outlier Detection Summary")
    
    total_days = len(outlier_df)
    outlier_days = outlier_df['is_outlier'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days Analyzed", f"{total_days:,}")
    with col2:
        st.metric("Outlier Days Detected", f"{outlier_days:,}")
    with col3:
        st.metric("Outlier Rate", f"{outlier_days/total_days*100:.1f}%")
    
    st.markdown("---")
    
    # Chart
    st.markdown("### Net Cash Flow with Outliers")
    
    # Merge data properly
    merged = daily_cash.copy()
    merged['is_outlier'] = outlier_df['is_outlier'].values
    
    fig = go.Figure()
    
    normal_df = merged[~merged['is_outlier']]
    outlier_data = merged[merged['is_outlier']]
    
    fig.add_trace(go.Scatter(
        x=normal_df['date'],
        y=normal_df['net_cash_flow'],
        mode='markers', 
        name='Normal',
        marker=dict(color='#1f77b4', size=5, opacity=0.6),
        hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>Normal</extra>'
    ))
    
    if len(outlier_data) > 0:
        fig.add_trace(go.Scatter(
            x=outlier_data['date'],
            y=outlier_data['net_cash_flow'],
            mode='markers', 
            name='Outlier',
            marker=dict(color='#d62728', size=10, symbol='x'),
            hovertemplate='%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra>OUTLIER</extra>'
        ))
    
    fig.update_layout(
        height=400, 
        yaxis_tickformat='$,.0f',
        xaxis_title="Date", 
        yaxis_title="Net Cash Flow",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode='closest'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Details table
    st.markdown("### Outlier Details")
    
    if outlier_summary is not None and len(outlier_summary) > 0 and 'message' not in outlier_summary.columns:
        display = outlier_summary.head(15).copy()
        display['date'] = pd.to_datetime(display['date']).dt.strftime('%Y-%m-%d')
        for col in ['inflow', 'outflow', 'net']:
            if col in display.columns:
                display[col] = display[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No significant outliers detected!")
    
    st.markdown("---")
    st.markdown("### Interpretation")
    st.markdown("""
    **Outliers may indicate:**
    - 📅 Scheduled large payments (payroll, tax, debt service)
    - 🔴 Data entry errors requiring correction
    - 💼 One-time transactions (M&A, special dividends)
    - 📈 Seasonal spikes (bonus payouts, year-end settlements)
    
    **Recommended Action:** Review flagged days with treasury team to validate data accuracy.
    """)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not IMPORTS_OK:
        st.error(f"Import error: {IMPORT_ERROR}")
        return
    
    init_session_state()
    render_sidebar()
    
    st.title("💰 Cash Forecasting Intelligence")
    st.caption("Enterprise Treasury Forecasting with Prophet • Banking Days Only • US Holiday Calendar")
    
    tabs = st.tabs(["📊 Overview", "📈 Forecasts", "🎯 Accuracy", "🔍 Key Drivers", "⚠️ Outliers"])
    
    with tabs[0]:
        render_overview_tab()
    with tabs[1]:
        render_forecasts_tab()
    with tabs[2]:
        render_accuracy_tab()
    with tabs[3]:
        render_shap_tab()
    with tabs[4]:
        render_outliers_tab()


if __name__ == "__main__":
    main()
