"""
Cash Forecasting Dashboard v3 - With Category Breakdown
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

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
    from data_simulator_v2 import generate_category_data
    from models_prophet_v3 import ProphetCashForecaster, ForecastAnalyzer, run_backtest
    from analysis_v2 import SHAPAnalyzer, OutlierDetector
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# =============================================================================
# COLORS
# =============================================================================
INFLOW_COLORS = {
    'AR': '#27ae60',
    'INV_INC': '#2ecc71',
    'IC_IN': '#1abc9c',
}

OUTFLOW_COLORS = {
    'PAYROLL': '#e74c3c',
    'AP': '#c0392b',
    'TAX': '#9b59b6',
    'CAPEX': '#8e44ad',
    'DEBT': '#e67e22',
    'IC_OUT': '#d35400',
}

# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        'data_loaded': False,
        'daily_cash': None,
        'category_df': None,
        'forecaster': None,
        'forecasts': None,
        'backtest_results': None,
        'shap_results': None,
        'outlier_results': None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# =============================================================================
# CATEGORY CHART
# =============================================================================
def create_category_chart(forecast_df: pd.DataFrame, horizon: str) -> go.Figure:
    """3-panel chart: Balance, Inflows by category, Outflows by category."""
    
    df = forecast_df.sort_values('date', ascending=True).copy()
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            'Closing Balance',
            'Inflows: AR, Investment Income, Intercompany In',
            'Outflows: Payroll, AP, Tax, CAPEX, Debt, Intercompany Out'
        ),
        row_heights=[0.34, 0.33, 0.33]
    )
    
    # Row 1: Balance line
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['closing_balance'],
        mode='lines+markers',
        name='Closing Balance',
        line=dict(color='#3498db', width=3),
        marker=dict(size=8),
        hovertemplate='%{x|%b %d %a}<br>Balance: $%{y:,.0f}<extra></extra>'
    ), row=1, col=1)
    
    # Row 2: Inflow categories as lines
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['AR'],
        mode='lines+markers',
        name='AR (Receivables)',
        line=dict(color=INFLOW_COLORS['AR'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>AR: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['INV_INC'],
        mode='lines+markers',
        name='Investment Income',
        line=dict(color=INFLOW_COLORS['INV_INC'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>Inv Inc: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['IC_IN'],
        mode='lines+markers',
        name='Intercompany In',
        line=dict(color=INFLOW_COLORS['IC_IN'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>IC In: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Total inflow dashed
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['forecast_inflow'],
        mode='lines',
        name='Total Inflow',
        line=dict(color='#000000', width=2, dash='dot'),
        hovertemplate='%{x|%b %d}<br>Total: $%{y:,.0f}<extra></extra>'
    ), row=2, col=1)
    
    # Row 3: Outflow categories as lines
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['PAYROLL'],
        mode='lines+markers',
        name='Payroll',
        line=dict(color=OUTFLOW_COLORS['PAYROLL'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>Payroll: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['AP'],
        mode='lines+markers',
        name='AP (Payables)',
        line=dict(color=OUTFLOW_COLORS['AP'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>AP: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['TAX'],
        mode='lines+markers',
        name='Tax',
        line=dict(color=OUTFLOW_COLORS['TAX'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>Tax: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['CAPEX'],
        mode='lines+markers',
        name='CAPEX',
        line=dict(color=OUTFLOW_COLORS['CAPEX'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>CAPEX: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['DEBT'],
        mode='lines+markers',
        name='Debt Service',
        line=dict(color=OUTFLOW_COLORS['DEBT'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>Debt: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['IC_OUT'],
        mode='lines+markers',
        name='Intercompany Out',
        line=dict(color=OUTFLOW_COLORS['IC_OUT'], width=2),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>IC Out: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    # Total outflow dashed
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['forecast_outflow'],
        mode='lines',
        name='Total Outflow',
        line=dict(color='#000000', width=2, dash='dot'),
        hovertemplate='%{x|%b %d}<br>Total: $%{y:,.0f}<extra></extra>'
    ), row=3, col=1)
    
    fig.update_layout(
        height=850,
        title=f'{horizon} Forecast with Category Breakdown',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        margin=dict(b=120)
    )
    
    fig.update_yaxes(tickformat='$,.0f', row=1, col=1)
    fig.update_yaxes(tickformat='$,.0f', row=2, col=1)
    fig.update_yaxes(tickformat='$,.0f', row=3, col=1)
    fig.update_xaxes(tickformat='%b %d (%a)', row=3, col=1)
    
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar():
    st.sidebar.title("💰 Cash Forecasting")
    st.sidebar.markdown("---")
    
    periods = st.sidebar.slider("Historical Data (days)", 365, 1095, 730, 30)
    test_size = st.sidebar.slider("Test Period (days)", 30, 180, 90, 10)
    
    if st.sidebar.button("🚀 Load Data & Train", type="primary", use_container_width=True):
        with st.spinner("Training models..."):
            data = generate_category_data(periods=periods)
            st.session_state.daily_cash = data['daily_cash_position']
            st.session_state.category_df = data['category_details']
            
            results, forecaster, forecasts = run_backtest(
                st.session_state.daily_cash,
                test_size=test_size,
                category_df=st.session_state.category_df
            )
            
            st.session_state.forecaster = forecaster
            st.session_state.forecasts = forecasts
            st.session_state.backtest_results = results
            
            # SHAP
            shap_analyzer = SHAPAnalyzer(forecaster)
            st.session_state.shap_results = shap_analyzer.analyze()
            
            # Outliers
            outlier_detector = OutlierDetector()
            st.session_state.outlier_results = outlier_detector.detect(st.session_state.daily_cash.copy())
            st.session_state.outlier_summary = outlier_detector.get_outlier_summary()
            
            st.session_state.data_loaded = True
        st.sidebar.success("✅ Ready!")
    
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.metric("T0 Date", st.session_state.forecaster.last_actual_date.strftime('%Y-%m-%d'))
        st.sidebar.metric("T0 Balance", f"${st.session_state.forecaster.last_actual_closing_balance:,.0f}")

# =============================================================================
# TABS
# =============================================================================
def render_overview():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train' to start")
        return
    
    forecaster = st.session_state.forecaster
    forecasts = st.session_state.forecasts
    results = st.session_state.backtest_results
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("T0 Balance", f"${forecaster.last_actual_closing_balance:,.0f}")
    with col2:
        st.metric("T0 Date", forecaster.last_actual_date.strftime('%Y-%m-%d'))
    with col3:
        if 'T+7' in results:
            st.metric("T+7 Accuracy", f"{results['T+7']['balance_mape']:.1f}% error", results['T+7']['rating'])
    with col4:
        if 'T+30' in results:
            st.metric("T+30 Accuracy", f"{results['T+30']['balance_mape']:.1f}% error", results['T+30']['rating'])
    
    st.markdown("---")
    
    summary = []
    for hz in ['T+7', 'T+30', 'T+90']:
        if hz in forecasts:
            f = forecasts[hz]
            summary.append({
                'Horizon': hz,
                'Days': len(f),
                'Opening': f"${forecaster.last_actual_closing_balance:,.0f}",
                'Total Inflows': f"${f['forecast_inflow'].sum():,.0f}",
                'Total Outflows': f"${f['forecast_outflow'].sum():,.0f}",
                'Closing': f"${f['closing_balance'].iloc[-1]:,.0f}",
                'Accuracy': f"{results[hz]['balance_mape']:.1f}% error" if hz in results else 'N/A',
                'Rating': results[hz]['rating'] if hz in results else 'N/A'
            })
    
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


def render_forecasts():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train' to view forecasts")
        return
    
    forecasts = st.session_state.forecasts
    forecaster = st.session_state.forecaster
    results = st.session_state.backtest_results
    
    horizon = st.selectbox("Forecast Horizon", ['T+7', 'T+30', 'T+90'])
    
    if horizon not in forecasts:
        st.warning(f"No forecast for {horizon}")
        return
    
    fcast = forecasts[horizon].sort_values('date', ascending=True)
    
    # Treasury Summary
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Opening", f"${forecaster.last_actual_closing_balance:,.0f}")
    with col2:
        st.metric("+ Receipts", f"${fcast['forecast_inflow'].sum():,.0f}")
    with col3:
        st.metric("− Payments", f"${fcast['forecast_outflow'].sum():,.0f}")
    with col4:
        st.metric("= Closing", f"${fcast['closing_balance'].iloc[-1]:,.0f}")
    with col5:
        mape = results[horizon]['balance_mape'] if horizon in results else 0
        st.metric("Accuracy", f"{mape:.1f}% error", results[horizon]['rating'] if horizon in results else '')
    
    st.markdown("---")
    
    # Category Chart
    fig = create_category_chart(fcast, horizon)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Data Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Inflow Breakdown")
        inflow_df = fcast[['date', 'day_name', 'AR', 'INV_INC', 'IC_IN', 'forecast_inflow']].copy()
        inflow_df['date'] = inflow_df['date'].dt.strftime('%Y-%m-%d')
        inflow_df.columns = ['Date', 'Day', 'AR', 'Inv Income', 'IC In', 'Total']
        for c in ['AR', 'Inv Income', 'IC In', 'Total']:
            inflow_df[c] = inflow_df[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(inflow_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Outflow Breakdown")
        outflow_df = fcast[['date', 'day_name', 'PAYROLL', 'AP', 'TAX', 'CAPEX', 'DEBT', 'IC_OUT', 'forecast_outflow']].copy()
        outflow_df['date'] = outflow_df['date'].dt.strftime('%Y-%m-%d')
        outflow_df.columns = ['Date', 'Day', 'Payroll', 'AP', 'Tax', 'CAPEX', 'Debt', 'IC Out', 'Total']
        for c in ['Payroll', 'AP', 'Tax', 'CAPEX', 'Debt', 'IC Out', 'Total']:
            outflow_df[c] = outflow_df[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(outflow_df, use_container_width=True, hide_index=True)


def render_accuracy():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train' to view accuracy")
        return
    
    results = st.session_state.backtest_results
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'], key='acc_hz')
    
    if horizon not in results:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Balance Accuracy", f"{results[horizon]['balance_mape']:.2f}% error")
    with col2:
        st.metric("Inflow Accuracy", f"{results[horizon]['inflow_mape']:.2f}% error")
    with col3:
        st.metric("Outflow Accuracy", f"{results[horizon]['outflow_mape']:.2f}% error")
    with col4:
        st.metric("Rating", results[horizon]['rating'])
    
    if 'mape_by_horizon_day' in results[horizon]:
        mape_df = results[horizon]['mape_by_horizon_day']
        colors = ['#4CAF50' if m <= 2 else '#FFC107' if m <= 5 else '#F44336' for m in mape_df['balance_mape']]
        
        fig = go.Figure(go.Bar(x=mape_df['horizon_day'], y=mape_df['balance_mape'], marker_color=colors))
        fig.add_hline(y=2, line_dash="dot", line_color="green")
        fig.add_hline(y=5, line_dash="dot", line_color="orange")
        fig.update_layout(title="Balance Accuracy by Horizon Day", xaxis_title="Day", yaxis_title="Error %", height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_shap():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train' to view Key Drivers")
        return
    
    shap_results = st.session_state.shap_results
    if not shap_results:
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Inflow Drivers")
        if 'inflow' in shap_results:
            df = shap_results['inflow']['importance']
            fig = go.Figure(go.Bar(x=df['importance_pct'], y=df['component'], orientation='h', marker_color='#27ae60'))
            fig.update_layout(height=300, yaxis=dict(autorange="reversed"), xaxis_title="Importance %")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Outflow Drivers")
        if 'outflow' in shap_results:
            df = shap_results['outflow']['importance']
            fig = go.Figure(go.Bar(x=df['importance_pct'], y=df['component'], orientation='h', marker_color='#e74c3c'))
            fig.update_layout(height=300, yaxis=dict(autorange="reversed"), xaxis_title="Importance %")
            st.plotly_chart(fig, use_container_width=True)


def render_outliers():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train' to view outliers")
        return
    
    outlier_df = st.session_state.outlier_results
    daily_cash = st.session_state.daily_cash.sort_values('date').reset_index(drop=True)
    outlier_df = outlier_df.sort_values('date').reset_index(drop=True)
    
    total = len(outlier_df)
    outliers = outlier_df['is_outlier'].sum()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days", total)
    with col2:
        st.metric("Outliers", outliers)
    with col3:
        st.metric("Rate", f"{outliers/total*100:.1f}%")
    
    merged = daily_cash.copy()
    merged['is_outlier'] = outlier_df['is_outlier'].values
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged[~merged['is_outlier']]['date'],
        y=merged[~merged['is_outlier']]['net_cash_flow'],
        mode='markers', name='Normal', marker=dict(color='#3498db', size=5)
    ))
    fig.add_trace(go.Scatter(
        x=merged[merged['is_outlier']]['date'],
        y=merged[merged['is_outlier']]['net_cash_flow'],
        mode='markers', name='Outlier', marker=dict(color='#e74c3c', size=10, symbol='x')
    ))
    fig.update_layout(height=400, yaxis_tickformat='$,.0f')
    st.plotly_chart(fig, use_container_width=True)

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
    st.caption("Prophet Forecasting • Category Breakdown • Banking Days Only")
    
    tabs = st.tabs(["📊 Overview", "📈 Forecasts", "🎯 Accuracy", "🔍 Key Drivers", "⚠️ Outliers"])
    
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_forecasts()
    with tabs[2]:
        render_accuracy()
    with tabs[3]:
        render_shap()
    with tabs[4]:
        render_outliers()


if __name__ == "__main__":
    main()
