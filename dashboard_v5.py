"""
Cash Forecasting Dashboard v5
=============================
Enhanced accuracy analysis with daily MAPE breakdown.
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
    from data_simulator_v3 import generate_category_data
    from models_prophet_v6 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar
    from analysis_v3 import SHAPAnalyzer, OutlierDetector
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    IMPORT_ERROR = str(e)

# =============================================================================
# COLORS & CONSTANTS
# =============================================================================
INFLOW_COLORS = {'AR': '#27ae60', 'INV_INC': '#2ecc71', 'IC_IN': '#1abc9c'}
OUTFLOW_COLORS = {
    'PAYROLL': '#e74c3c', 'AP': '#c0392b', 'TAX': '#9b59b6',
    'CAPEX': '#f39c12', 'DEBT': '#e67e22', 'IC_OUT': '#d35400'
}
DOW_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

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
        'capex_schedule': {},
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# =============================================================================
# ENHANCED BACKTEST
# =============================================================================
def run_detailed_backtest(daily_cash, category_df, test_size=90):
    """Run backtest with detailed daily error analysis."""
    
    train_df = daily_cash.iloc[:-test_size].copy()
    test_df = daily_cash.iloc[-test_size:].copy()
    train_cat = category_df.iloc[:-test_size].copy() if category_df is not None else None
    test_cat = category_df.iloc[-test_size:].copy() if category_df is not None else None
    
    # Train model
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df, train_cat)
    
    # Get holidays
    holidays = USBankingCalendar.get_us_holidays(
        daily_cash['date'].min().year,
        daily_cash['date'].max().year + 1
    )
    
    # Prepare test data
    test_df = test_df.copy()
    test_df['is_banking_day'] = test_df['date'].apply(
        lambda x: USBankingCalendar.is_banking_day(x, holidays)
    )
    test_df['day_of_week'] = test_df['date'].dt.dayofweek
    test_banking = test_df[test_df['is_banking_day']].copy()
    
    # Add outflow_ex_capex if category data available
    if test_cat is not None:
        test_cat = test_cat.copy()
        test_cat['is_banking_day'] = test_cat['date'].apply(
            lambda x: USBankingCalendar.is_banking_day(x, holidays)
        )
        test_cat_banking = test_cat[test_cat['is_banking_day']].copy()
        test_cat_banking['outflow_ex_capex'] = (
            test_cat_banking['PAYROLL'] + test_cat_banking['AP'] + 
            test_cat_banking['TAX'] + test_cat_banking['DEBT'] + test_cat_banking['IC_OUT']
        )
        test_banking = test_banking.merge(
            test_cat_banking[['date', 'outflow_ex_capex']],
            on='date', how='left'
        )
    
    # Generate forecasts
    forecasts = forecaster.predict()
    
    # Analyze each horizon
    detailed_results = {}
    all_daily_errors = []
    
    for horizon, forecast_df in forecasts.items():
        # Prepare merge columns
        merge_cols = ['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week']
        if 'outflow_ex_capex' in test_banking.columns:
            merge_cols.append('outflow_ex_capex')
        
        # Merge forecast with actuals
        merged = forecast_df.merge(
            test_banking[merge_cols],
            on='date', 
            how='inner',
            suffixes=('_forecast', '_actual')
        )
        
        if len(merged) == 0:
            continue
        
        # Add day_of_week from date (most reliable)
        merged['day_of_week'] = merged['date'].dt.dayofweek
        
        # Handle column naming after merge
        if 'closing_balance_forecast' in merged.columns:
            merged['forecast_balance'] = merged['closing_balance_forecast']
            merged['actual_balance'] = merged['closing_balance_actual']
        elif 'closing_balance' in merged.columns:
            merged['forecast_balance'] = merged['closing_balance']
            merged['actual_balance'] = merged['closing_balance']
        
        # Calculate errors
        merged['inflow_error'] = merged['forecast_inflow'] - merged['inflow']
        merged['inflow_pct_error'] = np.abs(merged['inflow_error'] / merged['inflow'].replace(0, np.nan)) * 100
        
        if 'outflow_ex_capex' in merged.columns:
            merged['outflow_error'] = merged['forecast_outflow_ex_capex'] - merged['outflow_ex_capex']
            merged['outflow_pct_error'] = np.abs(merged['outflow_error'] / merged['outflow_ex_capex'].replace(0, np.nan)) * 100
        else:
            merged['outflow_error'] = merged['forecast_outflow'] - merged['outflow']
            merged['outflow_pct_error'] = np.abs(merged['outflow_error'] / merged['outflow'].replace(0, np.nan)) * 100
        
        merged['balance_error'] = merged['forecast_balance'] - merged['actual_balance']
        merged['balance_pct_error'] = np.abs(merged['balance_error'] / merged['actual_balance'].replace(0, np.nan)) * 100
        
        # Fill NaN with 0 for aggregation
        merged['inflow_pct_error'] = merged['inflow_pct_error'].fillna(0)
        merged['outflow_pct_error'] = merged['outflow_pct_error'].fillna(0)
        merged['balance_pct_error'] = merged['balance_pct_error'].fillna(0)
        
        # MAPE by Horizon Day
        mape_by_horizon = merged.groupby('horizon_day').agg({
            'inflow_pct_error': 'mean',
            'outflow_pct_error': 'mean',
            'balance_pct_error': 'mean'
        }).reset_index()
        mape_by_horizon.columns = ['horizon_day', 'inflow_mape', 'outflow_mape', 'balance_mape']
        
        # MAPE by Day of Week
        mape_by_dow = merged.groupby('day_of_week').agg({
            'inflow_pct_error': 'mean',
            'outflow_pct_error': 'mean',
            'balance_pct_error': 'mean'
        }).reset_index()
        mape_by_dow.columns = ['day_of_week', 'inflow_mape', 'outflow_mape', 'balance_mape']
        mape_by_dow['day_name'] = mape_by_dow['day_of_week'].apply(
            lambda x: DOW_NAMES[int(x)] if x < 5 else f'Day {int(x)}'
        )
        
        # Overall metrics
        inflow_mape = merged['inflow_pct_error'].mean()
        outflow_mape = merged['outflow_pct_error'].mean()
        balance_mape = merged['balance_pct_error'].mean()
        
        # Rating
        thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
        if balance_mape <= thresholds["excellent"]:
            rating = "Excellent"
        elif balance_mape <= thresholds["good"]:
            rating = "Good"
        elif balance_mape <= thresholds["acceptable"]:
            rating = "Acceptable"
        else:
            rating = "Poor"
        
        detailed_results[horizon] = {
            'inflow_mape': inflow_mape,
            'outflow_mape': outflow_mape,
            'balance_mape': balance_mape,
            'outflow_label': 'Outflow (ex-CAPEX)',
            'rating': rating,
            'samples': len(merged),
            'mape_by_horizon_day': mape_by_horizon,
            'mape_by_dow': mape_by_dow,
            'daily_errors': merged,
        }
        
        merged['horizon'] = horizon
        all_daily_errors.append(merged)
    
    combined_errors = pd.concat(all_daily_errors, ignore_index=True) if all_daily_errors else pd.DataFrame()
    
    return detailed_results, forecaster, forecasts, combined_errors

# =============================================================================
# CHARTS
# =============================================================================
def create_category_chart(forecast_df: pd.DataFrame, horizon: str) -> go.Figure:
    df = forecast_df.sort_values('date', ascending=True).copy()
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=('Closing Balance', 'Inflows', 'Outflows'),
        row_heights=[0.34, 0.33, 0.33]
    )
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['closing_balance'],
        mode='lines+markers', name='Balance',
        line=dict(color='#3498db', width=3), marker=dict(size=8)
    ), row=1, col=1)
    
    for cat, color in INFLOW_COLORS.items():
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[cat], mode='lines+markers', name=cat,
            line=dict(color=color, width=2), marker=dict(size=6)
        ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['forecast_inflow'], mode='lines', name='Total Inflow',
        line=dict(color='#000', width=2, dash='dot')
    ), row=2, col=1)
    
    for cat in ['PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT']:
        fig.add_trace(go.Scatter(
            x=df['date'], y=df[cat], mode='lines+markers', name=cat,
            line=dict(color=OUTFLOW_COLORS[cat], width=2), marker=dict(size=6)
        ), row=3, col=1)
    
    if df['CAPEX'].sum() > 0:
        fig.add_trace(go.Bar(
            x=df['date'], y=df['CAPEX'], name='CAPEX (User)',
            marker_color=OUTFLOW_COLORS['CAPEX'], opacity=0.8
        ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['forecast_outflow'], mode='lines', name='Total Outflow',
        line=dict(color='#000', width=2, dash='dot')
    ), row=3, col=1)
    
    fig.update_layout(
        height=850, title=f'{horizon} Forecast', hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5),
        margin=dict(b=120)
    )
    fig.update_yaxes(tickformat='$,.0f')
    fig.update_xaxes(tickformat='%b %d (%a)', row=3, col=1)
    
    return fig


def create_mape_by_horizon_chart(mape_df: pd.DataFrame) -> go.Figure:
    colors = ['#4CAF50' if m <= 2 else '#FFC107' if m <= 5 else '#F44336' for m in mape_df['balance_mape']]
    
    fig = go.Figure(go.Bar(
        x=mape_df['horizon_day'], y=mape_df['balance_mape'],
        marker_color=colors,
        hovertemplate='Day %{x}<br>Error: %{y:.2f}%<extra></extra>'
    ))

    fig.add_hline(y=2, line_dash="dot", line_color="green", annotation_text="2%")
    fig.add_hline(y=5, line_dash="dot", line_color="orange", annotation_text="5%")

    fig.update_layout(
        title="Balance Accuracy by Horizon Day",
        xaxis_title="Forecast Day",
        yaxis_title="Error %",
        height=350
    )
    return fig


def create_mape_by_dow_chart(mape_df: pd.DataFrame) -> go.Figure:
    mape_df = mape_df.sort_values('day_of_week')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=mape_df['day_name'], y=mape_df['balance_mape'],
        name='Balance', marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        x=mape_df['day_name'], y=mape_df['inflow_mape'],
        name='Inflow', marker_color='#27ae60'
    ))
    fig.add_trace(go.Bar(
        x=mape_df['day_name'], y=mape_df['outflow_mape'],
        name='Outflow', marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title="Accuracy by Day of Week",
        xaxis_title="Day",
        yaxis_title="Error %",
        barmode='group',
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    return fig


def create_error_heatmap(daily_errors: pd.DataFrame, horizon: str) -> go.Figure:
    df = daily_errors[daily_errors['horizon'] == horizon].copy()
    
    if len(df) == 0:
        return go.Figure()
    
    pivot = df.pivot_table(
        index='day_of_week',
        columns='horizon_day',
        values='balance_pct_error',
        aggfunc='mean'
    )
    
    pivot.index = [DOW_NAMES[int(i)] if i < 5 else f'Day {int(i)}' for i in pivot.index]
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'Day {c}' for c in pivot.columns],
        y=pivot.index,
        colorscale='RdYlGn_r',
        hovertemplate='%{y}, %{x}<br>Error: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        title=f"Balance Accuracy Heatmap ({horizon})",
        xaxis_title="Horizon Day",
        yaxis_title="Day of Week",
        height=300
    )
    return fig


def create_horizon_comparison_chart(results: dict) -> go.Figure:
    horizons = list(results.keys())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=horizons,
        y=[results[h]['balance_mape'] for h in horizons],
        name='Balance', marker_color='#3498db'
    ))
    fig.add_trace(go.Bar(
        x=horizons,
        y=[results[h]['inflow_mape'] for h in horizons],
        name='Inflow', marker_color='#27ae60'
    ))
    fig.add_trace(go.Bar(
        x=horizons,
        y=[results[h]['outflow_mape'] for h in horizons],
        name='Outflow', marker_color='#e74c3c'
    ))
    
    fig.update_layout(
        title="Accuracy Across Horizons",
        xaxis_title="Horizon",
        yaxis_title="Error %",
        barmode='group',
        height=350
    )
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
        with st.spinner("Training..."):
            data = generate_category_data(periods=periods)
            st.session_state.daily_cash = data['daily_cash_position']
            st.session_state.category_df = data['category_details']
            
            results, forecaster, forecasts, daily_errors = run_detailed_backtest(
                st.session_state.daily_cash,
                st.session_state.category_df,
                test_size=test_size
            )
            
            st.session_state.forecaster = forecaster
            st.session_state.forecasts = forecasts
            st.session_state.backtest_results = results
            st.session_state.daily_errors = daily_errors
            
            shap_analyzer = SHAPAnalyzer(forecaster)
            st.session_state.shap_results = shap_analyzer.analyze()
            
            outlier_detector = OutlierDetector()
            outlier_detector.detect(st.session_state.daily_cash.copy(), st.session_state.category_df)
            st.session_state.outlier_detector = outlier_detector
            st.session_state.outlier_results = outlier_detector.results
            st.session_state.outlier_summary = outlier_detector.get_outlier_summary()
            
            st.session_state.data_loaded = True
        st.sidebar.success("✅ Ready!")
    
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.metric("T0", st.session_state.forecaster.last_actual_date.strftime('%Y-%m-%d'))
        st.sidebar.metric("Balance", f"${st.session_state.forecaster.last_actual_closing_balance:,.0f}")

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
            st.metric("T+7 Accuracy", f"{results['T+7']['balance_mape']:.2f}% error", results['T+7']['rating'])
    with col4:
        if 'T+30' in results:
            st.metric("T+30 Accuracy", f"{results['T+30']['balance_mape']:.2f}% error", results['T+30']['rating'])
    
    st.markdown("---")
    
    summary = []
    for hz in ['T+7', 'T+30', 'T+90']:
        if hz in forecasts and hz in results:
            f = forecasts[hz]
            summary.append({
                'Horizon': hz,
                'Days': len(f),
                'Inflows': f"${f['forecast_inflow'].sum():,.0f}",
                'Outflows': f"${f['forecast_outflow_ex_capex'].sum():,.0f}",
                'Closing': f"${f['closing_balance'].iloc[-1]:,.0f}",
                'Balance Accuracy': f"{results[hz]['balance_mape']:.2f}% error",
                'Rating': results[hz]['rating']
            })
    
    st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


def render_forecasts():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train'")
        return
    
    forecaster = st.session_state.forecaster
    results = st.session_state.backtest_results
    
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'])
    
    st.markdown("---")
    
    # CAPEX Input
    st.subheader("📝 Planned CAPEX")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        base_forecast = forecaster.predict(horizon)[horizon]
        dates = base_forecast['date'].dt.strftime('%Y-%m-%d').tolist()
        capex_date = st.selectbox("Date", dates, key='capex_date')
    with col2:
        capex_amt = st.number_input("Amount ($)", 0, 100_000_000, 0, 100_000, key='capex_amt')
    with col3:
        st.write("")
        st.write("")
        if st.button("➕ Add"):
            if capex_amt > 0:
                st.session_state.capex_schedule[capex_date] = capex_amt
                st.rerun()
    
    if st.session_state.capex_schedule:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(pd.DataFrame([
                {'Date': k, 'Amount': f"${v:,.0f}"} 
                for k, v in sorted(st.session_state.capex_schedule.items())
            ]), use_container_width=True, hide_index=True)
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.capex_schedule = {}
                st.rerun()
    
    st.markdown("---")
    
    forecasts = forecaster.predict(horizon, capex_schedule=st.session_state.capex_schedule)
    fcast = forecasts[horizon].sort_values('date', ascending=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Opening", f"${forecaster.last_actual_closing_balance:,.0f}")
    with col2:
        st.metric("+ Receipts", f"${fcast['forecast_inflow'].sum():,.0f}")
    with col3:
        st.metric("− Payments", f"${fcast['forecast_outflow_ex_capex'].sum():,.0f}")
    with col4:
        st.metric("− CAPEX", f"${fcast['CAPEX'].sum():,.0f}")
    with col5:
        st.metric("= Closing", f"${fcast['closing_balance'].iloc[-1]:,.0f}")
    
    fig = create_category_chart(fcast, horizon)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Inflows")
        df = fcast[['date', 'day_name', 'AR', 'INV_INC', 'IC_IN', 'forecast_inflow']].copy()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.columns = ['Date', 'Day', 'AR', 'Inv Inc', 'IC In', 'Total']
        for c in ['AR', 'Inv Inc', 'IC In', 'Total']:
            df[c] = df[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Outflows")
        df = fcast[['date', 'day_name', 'PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT', 'CAPEX', 'forecast_outflow']].copy()
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.columns = ['Date', 'Day', 'Payroll', 'AP', 'Tax', 'Debt', 'IC Out', 'CAPEX', 'Total']
        for c in ['Payroll', 'AP', 'Tax', 'Debt', 'IC Out', 'CAPEX', 'Total']:
            df[c] = df[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_accuracy():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train'")
        return
    
    results = st.session_state.backtest_results
    daily_errors = st.session_state.daily_errors
    
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'], key='acc_hz')
    
    if horizon not in results:
        return
    
    # Overall metrics
    st.subheader("📊 Overall Accuracy")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Balance Accuracy", f"{results[horizon]['balance_mape']:.2f}% error", results[horizon]['rating'])
    with col2:
        st.metric("Inflow Accuracy", f"{results[horizon]['inflow_mape']:.2f}% error")
    with col3:
        st.metric("Outflow Accuracy", f"{results[horizon]['outflow_mape']:.2f}% error")
    with col4:
        st.metric("Samples", results[horizon]['samples'])
    
    st.markdown("---")
    
    # Accuracy by Horizon Day
    st.subheader("📈 Accuracy by Horizon Day")
    st.caption("Which forecast days have higher error? Set cash buffers accordingly.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_mape_by_horizon_chart(results[horizon]['mape_by_horizon_day'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        df = results[horizon]['mape_by_horizon_day'].copy()
        df['Balance'] = df['balance_mape'].apply(lambda x: f"{x:.2f}%")
        df['Inflow'] = df['inflow_mape'].apply(lambda x: f"{x:.2f}%")
        df['Outflow'] = df['outflow_mape'].apply(lambda x: f"{x:.2f}%")
        df['Buffer'] = df['balance_mape'].apply(
            lambda x: '✅ Normal' if x <= 2 else '⚠️ +5%' if x <= 5 else '🔴 +10%'
        )
        df = df[['horizon_day', 'Balance', 'Inflow', 'Outflow', 'Buffer']]
        df.columns = ['Day', 'Balance', 'Inflow', 'Outflow', 'Buffer']
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Accuracy by Day of Week
    st.subheader("📅 Accuracy by Day of Week")
    st.caption("Are certain weekdays harder to forecast?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_mape_by_dow_chart(results[horizon]['mape_by_dow'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        df = results[horizon]['mape_by_dow'].sort_values('day_of_week').copy()
        df['Balance'] = df['balance_mape'].apply(lambda x: f"{x:.2f}%")
        df['Inflow'] = df['inflow_mape'].apply(lambda x: f"{x:.2f}%")
        df['Outflow'] = df['outflow_mape'].apply(lambda x: f"{x:.2f}%")
        df = df[['day_name', 'Balance', 'Inflow', 'Outflow']]
        df.columns = ['Day', 'Balance', 'Inflow', 'Outflow']
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Heatmap
    st.subheader("🗺️ Error Heatmap")
    if daily_errors is not None and len(daily_errors) > 0:
        fig = create_error_heatmap(daily_errors, horizon)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Cross-horizon comparison
    st.subheader("📊 Cross-Horizon Comparison")
    fig = create_horizon_comparison_chart(results)
    st.plotly_chart(fig, use_container_width=True)


def render_shap():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train'")
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
            fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Outflow Drivers")
        if 'outflow' in shap_results:
            df = shap_results['outflow']['importance']
            fig = go.Figure(go.Bar(x=df['importance_pct'], y=df['component'], orientation='h', marker_color='#e74c3c'))
            fig.update_layout(height=300, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


def render_outliers():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train'")
        return

    outlier_detector = st.session_state.outlier_detector
    summary = outlier_detector.get_outlier_summary()
    outliers_df = outlier_detector.get_outliers()

    # Summary metrics
    st.subheader("📊 Outlier Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Days Analyzed", summary.get('total_days', 0))
    with col2:
        st.metric("Outliers Found", summary.get('outlier_count', 0))
    with col3:
        st.metric("High Priority", summary.get('by_severity', {}).get('High', 0))
    with col4:
        st.metric("Medium Priority", summary.get('by_severity', {}).get('Medium', 0))

    st.markdown("---")

    # Actionable outliers
    if outliers_df is not None and len(outliers_df) > 0:
        st.subheader("🎯 Items Requiring Attention")

        for sev in ['High', 'Medium']:
            sev_df = outliers_df[outliers_df['severity'] == sev]
            if len(sev_df) == 0:
                continue

            icon = "🔴" if sev == 'High' else "🟡"
            st.markdown(f"#### {icon} {sev} Priority ({len(sev_df)} items)")

            for _, row in sev_df.iterrows():
                variance = row['value'] - row['expected']
                variance_pct = (variance / abs(row['expected']) * 100) if row['expected'] != 0 else 0

                with st.expander(
                    f"{row['date'].strftime('%b %d, %Y')} ({row['day_name']}) — {row['anomaly_type']}",
                    expanded=(sev == 'High')
                ):
                    st.markdown(f"**{row['description']}**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual Amount", f"${row['value']:,.0f}")
                    with col2:
                        st.metric("Expected Amount", f"${row['expected']:,.0f}")
                    with col3:
                        delta_color = "inverse" if variance < 0 else "normal"
                        st.metric(
                            "Variance",
                            f"${abs(variance):,.0f}",
                            delta=f"{variance_pct:+.0f}%",
                            delta_color=delta_color
                        )

                    st.info(f"**Recommended Action:** {row['recommended_action']}")
    else:
        st.success("✅ No outliers detected. All cash flows are within normal ranges.")
    
    st.markdown("---")
    
    # Visual chart
    st.subheader("📈 Cash Flow Distribution")
    daily_cash = st.session_state.daily_cash.copy()
    daily_cash = daily_cash[daily_cash['is_banking_day']].sort_values('date')
    
    outlier_dates = set(outliers_df['date'].tolist()) if outliers_df is not None and len(outliers_df) > 0 else set()
    daily_cash['is_outlier'] = daily_cash['date'].isin(outlier_dates)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_cash[~daily_cash['is_outlier']]['date'],
        y=daily_cash[~daily_cash['is_outlier']]['net_cash_flow'],
        mode='markers', name='Normal', marker=dict(color='#3498db', size=6, opacity=0.6)
    ))
    if daily_cash['is_outlier'].any():
        fig.add_trace(go.Scatter(
            x=daily_cash[daily_cash['is_outlier']]['date'],
            y=daily_cash[daily_cash['is_outlier']]['net_cash_flow'],
            mode='markers', name='Outlier', marker=dict(color='#e74c3c', size=12, symbol='x', line=dict(width=2))
        ))
    
    # Add threshold lines
    stats = summary.get('net_flow_stats', {})
    if stats:
        fig.add_hline(y=stats.get('high_threshold', 0), line_dash="dot", line_color="orange", 
                     annotation_text="Upper threshold")
        fig.add_hline(y=stats.get('low_threshold', 0), line_dash="dot", line_color="orange",
                     annotation_text="Lower threshold")
        fig.add_hline(y=stats.get('mean', 0), line_dash="dash", line_color="green",
                     annotation_text="Mean")
    
    fig.update_layout(
        height=400, 
        yaxis_tickformat='$,.0f',
        title="Net Cash Flow (Outliers marked with X)",
        xaxis_title="Date",
        yaxis_title="Net Cash Flow"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Detection method explanation
    with st.expander("ℹ️ How outliers are detected"):
        st.markdown("""
        **Detection Methods:**
        
        1. **Net Cash Flow Z-Score**: Identifies days where total net cash movement 
           is significantly different from the historical average.
        
        2. **Category Z-Score**: Identifies unusual amounts within specific categories 
           (AR, AP, Payroll, Tax, Debt) compared to historical norms for that category.
        
        **Thresholds:**
        - 🔴 **High Severity**: > 3.0 standard deviations from mean
        - 🟡 **Medium Severity**: > 2.5 standard deviations from mean
        
        **Note:** Scheduled events (e.g., regular payroll, monthly debt service) are 
        compared against their own historical patterns, not overall averages.
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
    st.caption("Prophet • CAPEX User Input • Daily Accuracy Analysis")
    
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
