"""
Cash Forecasting Dashboard v6 - With Daily Position
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Cash Forecasting", page_icon="💰", layout="wide")

from config import TIME_HORIZONS, MAPE_THRESHOLDS
from data_simulator_v3 import generate_category_data
from models_prophet_v6 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar
from analysis_v3 import SHAPAnalyzer, OutlierDetector
from daily_position import (
    DailyPositionManager, simulate_intraday_data,
    archive_position, get_historical_accuracy,
    seed_historical_archive, clear_archive, DEFAULT_ARCHIVE_PATH,
)

INFLOW_COLORS = {'AR': '#27ae60', 'INV_INC': '#2ecc71', 'IC_IN': '#1abc9c'}
OUTFLOW_COLORS = {'PAYROLL': '#e74c3c', 'AP': '#c0392b', 'TAX': '#9b59b6', 'CAPEX': '#f39c12', 'DEBT': '#e67e22', 'IC_OUT': '#d35400'}
DOW_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

# Client-configurable settings - not exposed in UI
HISTORICAL_DAYS = 730      # Days of historical data to generate (2 years)
TEST_PERIOD_DAYS = 90      # Days held out for backtest accuracy metrics

def _fmt(value):
    """Format large dollar amounts: $226.1M, $45.2K, $3,200."""
    v = abs(value)
    sign = '-' if value < 0 else ''
    if v >= 1_000_000:
        return f"{sign}${v / 1_000_000:,.1f}M"
    if v >= 1_000:
        return f"{sign}${v / 1_000:,.1f}K"
    return f"{sign}${v:,.0f}"

def init_session_state():
    today = datetime.now().strftime('%Y-%m-%d')

    # Clear stale data if date changed
    if 'last_loaded_date' in st.session_state and st.session_state.last_loaded_date != today:
        st.session_state.data_loaded = False
        st.session_state.last_loaded_date = None

    for key, val in {'data_loaded': False, 'daily_cash': None, 'category_df': None, 'forecaster': None, 'forecasts': None, 'backtest_results': None, 'shap_results': None, 'outlier_results': None, 'capex_schedule': {}, 'last_loaded_date': None}.items():
        if key not in st.session_state:
            st.session_state[key] = val


def auto_load_data():
    """Auto-load data on first visit or when date changes."""
    today = datetime.now().strftime('%Y-%m-%d')

    # Skip if already loaded today
    if st.session_state.data_loaded and st.session_state.last_loaded_date == today:
        return

    # Generate fresh data - no caching to ensure dates are always current
    data = generate_category_data(periods=HISTORICAL_DAYS)
    st.session_state.daily_cash = data['daily_cash_position']
    st.session_state.category_df = data['category_details']

    # Run backtest for accuracy metrics (uses older data for training)
    results, backtest_forecaster, forecasts, daily_errors = run_detailed_backtest(
        st.session_state.daily_cash, st.session_state.category_df, test_size=TEST_PERIOD_DAYS
    )
    st.session_state.backtest_results = results
    st.session_state.daily_errors = daily_errors

    # Train LIVE forecaster on ALL data for correct T0 = yesterday
    live_forecaster = ProphetCashForecaster()
    live_forecaster.fit(st.session_state.daily_cash, st.session_state.category_df)
    live_forecasts = live_forecaster.predict()

    st.session_state.forecaster = live_forecaster  # Use live forecaster for T0
    st.session_state.forecasts = live_forecasts    # Use live forecasts

    shap_analyzer = SHAPAnalyzer(live_forecaster)
    st.session_state.shap_results = shap_analyzer.analyze()
    outlier_detector = OutlierDetector()
    outlier_detector.detect(st.session_state.daily_cash.copy(), st.session_state.category_df)
    st.session_state.outlier_detector = outlier_detector
    st.session_state.data_loaded = True
    st.session_state.last_loaded_date = today

def run_detailed_backtest(daily_cash, category_df, test_size=90):
    train_df = daily_cash.iloc[:-test_size].copy()
    test_df = daily_cash.iloc[-test_size:].copy()
    train_cat = category_df.iloc[:-test_size].copy() if category_df is not None else None
    test_cat = category_df.iloc[-test_size:].copy() if category_df is not None else None
    
    forecaster = ProphetCashForecaster()
    forecaster.fit(train_df, train_cat)
    
    holidays = USBankingCalendar.get_us_holidays(daily_cash['date'].min().year, daily_cash['date'].max().year + 1)
    test_df = test_df.copy()
    test_df['is_banking_day'] = test_df['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
    test_df['day_of_week'] = test_df['date'].dt.dayofweek
    test_banking = test_df[test_df['is_banking_day']].copy()
    
    if test_cat is not None:
        test_cat = test_cat.copy()
        test_cat['is_banking_day'] = test_cat['date'].apply(lambda x: USBankingCalendar.is_banking_day(x, holidays))
        test_cat_banking = test_cat[test_cat['is_banking_day']].copy()
        test_cat_banking['outflow_ex_capex'] = test_cat_banking['PAYROLL'] + test_cat_banking['AP'] + test_cat_banking['TAX'] + test_cat_banking['DEBT'] + test_cat_banking['IC_OUT']
        test_banking = test_banking.merge(test_cat_banking[['date', 'outflow_ex_capex']], on='date', how='left')
    
    forecasts = forecaster.predict()
    detailed_results = {}
    all_daily_errors = []
    
    for horizon, forecast_df in forecasts.items():
        merge_cols = ['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week']
        if 'outflow_ex_capex' in test_banking.columns:
            merge_cols.append('outflow_ex_capex')
        merged = forecast_df.merge(test_banking[merge_cols], on='date', how='inner', suffixes=('_forecast', '_actual'))
        if len(merged) == 0:
            continue
        merged['day_of_week'] = merged['date'].dt.dayofweek
        if 'closing_balance_forecast' in merged.columns:
            merged['forecast_balance'] = merged['closing_balance_forecast']
            merged['actual_balance'] = merged['closing_balance_actual']
        merged['inflow_pct_error'] = np.abs((merged['forecast_inflow'] - merged['inflow']) / merged['inflow'].replace(0, np.nan)) * 100
        if 'outflow_ex_capex' in merged.columns:
            merged['outflow_pct_error'] = np.abs((merged['forecast_outflow_ex_capex'] - merged['outflow_ex_capex']) / merged['outflow_ex_capex'].replace(0, np.nan)) * 100
        else:
            merged['outflow_pct_error'] = np.abs((merged['forecast_outflow'] - merged['outflow']) / merged['outflow'].replace(0, np.nan)) * 100
        merged['balance_pct_error'] = np.abs((merged['forecast_balance'] - merged['actual_balance']) / merged['actual_balance'].replace(0, np.nan)) * 100
        merged = merged.fillna(0)
        
        mape_by_horizon = merged.groupby('horizon_day').agg({'inflow_pct_error': 'mean', 'outflow_pct_error': 'mean', 'balance_pct_error': 'mean'}).reset_index()
        mape_by_horizon.columns = ['horizon_day', 'inflow_mape', 'outflow_mape', 'balance_mape']
        mape_by_dow = merged.groupby('day_of_week').agg({'inflow_pct_error': 'mean', 'outflow_pct_error': 'mean', 'balance_pct_error': 'mean'}).reset_index()
        mape_by_dow.columns = ['day_of_week', 'inflow_mape', 'outflow_mape', 'balance_mape']
        mape_by_dow['day_name'] = mape_by_dow['day_of_week'].apply(lambda x: DOW_NAMES[int(x)] if x < 5 else f'Day {int(x)}')
        
        balance_mape = merged['balance_pct_error'].mean()
        thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
        rating = "Excellent" if balance_mape <= thresholds["excellent"] else "Good" if balance_mape <= thresholds["good"] else "Acceptable" if balance_mape <= thresholds["acceptable"] else "Poor"
        
        detailed_results[horizon] = {'inflow_mape': merged['inflow_pct_error'].mean(), 'outflow_mape': merged['outflow_pct_error'].mean(), 'balance_mape': balance_mape, 'outflow_label': 'Outflow (ex-CAPEX)', 'rating': rating, 'samples': len(merged), 'mape_by_horizon_day': mape_by_horizon, 'mape_by_dow': mape_by_dow, 'daily_errors': merged}
        merged['horizon'] = horizon
        all_daily_errors.append(merged)
    
    combined_errors = pd.concat(all_daily_errors, ignore_index=True) if all_daily_errors else pd.DataFrame()
    return detailed_results, forecaster, forecasts, combined_errors

def create_category_chart(forecast_df, horizon):
    df = forecast_df.sort_values('date', ascending=True).copy()
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=('Closing Balance', 'Inflows', 'Outflows'), row_heights=[0.34, 0.33, 0.33])
    fig.add_trace(go.Scatter(x=df['date'], y=df['closing_balance'], mode='lines+markers', name='Balance', line=dict(color='#3498db', width=3)), row=1, col=1)
    for cat, color in INFLOW_COLORS.items():
        fig.add_trace(go.Scatter(x=df['date'], y=df[cat], mode='lines+markers', name=cat, line=dict(color=color, width=2)), row=2, col=1)
    for cat in ['PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT']:
        fig.add_trace(go.Scatter(x=df['date'], y=df[cat], mode='lines+markers', name=cat, line=dict(color=OUTFLOW_COLORS[cat], width=2)), row=3, col=1)
    if df['CAPEX'].sum() > 0:
        fig.add_trace(go.Bar(x=df['date'], y=df['CAPEX'], name='CAPEX', marker_color=OUTFLOW_COLORS['CAPEX']), row=3, col=1)
    fig.update_layout(height=850, title=f'{horizon} Forecast', hovermode='x unified', legend=dict(orientation="h", y=-0.12, x=0.5))
    fig.update_yaxes(tickformat='$,.0f')
    return fig

def create_mape_by_horizon_chart(mape_df):
    colors = ['#4CAF50' if m <= 2 else '#FFC107' if m <= 5 else '#F44336' for m in mape_df['balance_mape']]
    fig = go.Figure(go.Bar(x=mape_df['horizon_day'], y=mape_df['balance_mape'], marker_color=colors))
    fig.add_hline(y=2, line_dash="dot", line_color="green")
    fig.add_hline(y=5, line_dash="dot", line_color="orange")
    fig.update_layout(title="Balance Accuracy by Horizon Day", xaxis_title="Day", yaxis_title="Error %", height=350)
    return fig

def create_mape_by_dow_chart(mape_df):
    mape_df = mape_df.sort_values('day_of_week')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['balance_mape'], name='Balance', marker_color='#3498db'))
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['inflow_mape'], name='Inflow', marker_color='#27ae60'))
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['outflow_mape'], name='Outflow', marker_color='#e74c3c'))
    fig.update_layout(title="Accuracy by Day of Week", barmode='group', height=350)
    return fig

def create_horizon_comparison_chart(results):
    horizons = list(results.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['balance_mape'] for h in horizons], name='Balance', marker_color='#3498db'))
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['inflow_mape'] for h in horizons], name='Inflow', marker_color='#27ae60'))
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['outflow_mape'] for h in horizons], name='Outflow', marker_color='#e74c3c'))
    fig.update_layout(title="Accuracy Across Horizons", barmode='group', height=350)
    return fig

def render_sidebar():
    st.sidebar.title("💰 Cash Forecasting")
    st.sidebar.markdown("---")

    # Show today's date prominently
    today = datetime.now()
    st.sidebar.markdown(f"**Today:** {today.strftime('%B %d, %Y')}")

    # Auto-load on first visit
    if not st.session_state.data_loaded:
        with st.spinner("Loading data..."):
            auto_load_data()

    # Refresh button (for manual reload)
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        # Force reload with fresh data
        st.session_state.data_loaded = False
        st.session_state.last_loaded_date = None
        with st.spinner("Refreshing..."):
            auto_load_data()
        st.sidebar.success("Data refreshed!")

    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.metric("T0 (Forecast Start)", st.session_state.forecaster.last_actual_date.strftime('%b %d, %Y'))
        st.sidebar.metric("Current Balance", f"${st.session_state.forecaster.last_actual_closing_balance:,.0f}")

def _card_css():
    """Inject consistent card styling (called once)."""
    st.markdown("""
    <style>
    .exec-header {
        display: flex; justify-content: space-between; align-items: baseline;
        border-bottom: 2px solid #3498db; padding-bottom: 6px; margin-bottom: 18px;
    }
    .exec-header h2 { margin: 0; font-size: 1.4rem; color: #2c3e50; }
    .exec-header .timestamp { font-size: 0.85rem; color: #7f8c8d; }
    .section-card {
        border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px;
        margin-bottom: 12px; background: #fafbfc;
    }
    .section-card h3 {
        margin: 0 0 10px 0; font-size: 1.05rem; color: #2c3e50;
        border-bottom: 1px solid #ecf0f1; padding-bottom: 6px;
    }
    .alert-row { padding: 6px 0; font-size: 0.95rem; }
    </style>
    """, unsafe_allow_html=True)


def render_overview():
    if not st.session_state.data_loaded:
        st.info("Loading data...")
        return

    _card_css()
    forecaster = st.session_state.forecaster
    results = st.session_state.backtest_results
    t0_date = forecaster.last_actual_date
    t0_balance = forecaster.last_actual_closing_balance

    # --- Header with timestamp ---
    st.markdown(
        f'<div class="exec-header">'
        f'<h2>Executive Cash Position Summary</h2>'
        f'<span class="timestamp">Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  '
        f'&nbsp;|&nbsp; T0: {t0_date.strftime("%Y-%m-%d")}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # =================================================================
    # 1. TODAY'S POSITION
    # =================================================================
    position_date = t0_date.to_pydatetime() + timedelta(days=1)
    while position_date.weekday() >= 5:
        position_date += timedelta(days=1)
    t7_forecast = forecaster.predict('T+7')['T+7']
    manager = DailyPositionManager()
    manager.initialize_position(position_date, t0_balance, t7_forecast)
    intraday = simulate_intraday_data(position_date, t7_forecast)
    manager.load_intraday_from_bank(intraday['bank_transactions'])
    manager.load_sap_payment_queue(intraday['sap_payments'])
    manager.build_position()
    summary = manager.get_position_summary()
    rec = manager.get_investment_borrowing_recommendation()

    net_change = summary['closing_estimated'] - summary['opening_balance']

    with st.container(border=True):
        st.markdown(f"### Today's Position &mdash; {position_date.strftime('%A, %b %d %Y')}")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Opening Balance", f"${summary['opening_balance']:,.0f}")
        with c2:
            st.metric("Projected Closing", f"${summary['closing_estimated']:,.0f}")
        with c3:
            delta_color = "normal" if net_change >= 0 else "inverse"
            st.metric("Net Change", f"${net_change:,.0f}",
                       delta=f"${net_change:,.0f}", delta_color=delta_color)
        with c4:
            conf_icon = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(rec['confidence'], "⚪")
            st.metric("Confidence", f"{conf_icon} {rec['confidence']}")
        # Action strip
        if rec['action'] == 'INVEST':
            st.success(f"**Recommendation:** INVEST ${rec['amount']:,.0f} — {rec['reasoning']}")
        elif rec['action'] == 'BORROW':
            st.error(f"**Recommendation:** BORROW ${rec['amount']:,.0f} — {rec['reasoning']}")
        else:
            st.info(f"**Recommendation:** HOLD — {rec['reasoning']}")

    # =================================================================
    # 2. T+1 SNAPSHOT  |  3. T+7 SNAPSHOT  (side by side)
    # =================================================================
    col_left, col_right = st.columns(2)

    # --- T+1 ---
    with col_left:
        with st.container(border=True):
            st.markdown("### T+1 Snapshot")
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Forecast Close", f"${summary['closing_forecast']:,.0f}")
                st.metric("Estimated Actual", f"${summary['closing_estimated']:,.0f}")
            with m2:
                variance = summary['variance']
                var_icon = "🟢" if abs(variance) < 1_000_000 else "🟡" if abs(variance) < 5_000_000 else "🔴"
                st.metric("Variance", f"${variance:,.0f}",
                           delta=f"{var_icon} {'Over' if variance > 0 else 'Under'}")
                action_colors = {'INVEST': '🟢', 'BORROW': '🔴', 'HOLD': '🔵'}
                st.metric("Action", f"{action_colors.get(rec['action'], '')} {rec['action']}")
            st.caption(f"Posted: {summary['posted_transactions']} txns  |  Scheduled: {summary['scheduled_payments']} payments")

    # --- T+7 ---
    with col_right:
        with st.container(border=True):
            st.markdown("### T+7 Cash Trajectory")
            t7 = t7_forecast.sort_values('date').copy()
            fig7 = go.Figure()
            # Inflow/outflow bars
            fig7.add_trace(go.Bar(
                x=t7['date'], y=t7['forecast_inflow'], name='Inflows',
                marker_color='#27ae60', opacity=0.7,
            ))
            fig7.add_trace(go.Bar(
                x=t7['date'], y=-t7['forecast_outflow'], name='Outflows',
                marker_color='#e74c3c', opacity=0.7,
            ))
            # Balance line
            fig7.add_trace(go.Scatter(
                x=t7['date'], y=t7['closing_balance'], name='Balance',
                mode='lines+markers', line=dict(color='#3498db', width=3),
                yaxis='y2',
            ))
            # Min / Max annotations
            min_idx = t7['closing_balance'].idxmin()
            max_idx = t7['closing_balance'].idxmax()
            min_row = t7.loc[min_idx]
            max_row = t7.loc[max_idx]
            fig7.add_annotation(
                x=min_row['date'], y=min_row['closing_balance'],
                text=f"Min: ${min_row['closing_balance']:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor='#e74c3c',
                font=dict(color='#e74c3c', size=11, family='Arial Black'),
                bgcolor='white', bordercolor='#e74c3c', borderwidth=1, borderpad=3,
                ax=0, ay=40, yref='y2',
            )
            fig7.add_annotation(
                x=max_row['date'], y=max_row['closing_balance'],
                text=f"Max: ${max_row['closing_balance']:,.0f}",
                showarrow=True, arrowhead=2, arrowcolor='#27ae60',
                font=dict(color='#27ae60', size=11, family='Arial Black'),
                bgcolor='white', bordercolor='#27ae60', borderwidth=1, borderpad=3,
                ax=0, ay=-40, yref='y2',
            )
            fig7.update_layout(
                height=300, margin=dict(l=0, r=0, t=10, b=0),
                barmode='relative',
                yaxis=dict(title='Cash Flow ($)', tickformat='$,.0f', showgrid=False),
                yaxis2=dict(title='Balance ($)', tickformat='$,.0f', overlaying='y',
                            side='right', showgrid=True, gridcolor='#ecf0f1'),
                legend=dict(orientation='h', y=-0.25, x=0.5, xanchor='center'),
                hovermode='x unified',
            )
            st.plotly_chart(fig7, use_container_width=True, key='overview_t7')

    # =================================================================
    # 4. T+30 MONTHLY OUTLOOK
    # =================================================================
    with st.container(border=True):
        st.markdown("### T+30 Monthly Outlook")
        t30 = forecaster.predict('T+30')['T+30'].sort_values('date').copy()

        fig30 = make_subplots(specs=[[{"secondary_y": True}]])

        # Inflow / outflow bars
        fig30.add_trace(go.Bar(
            x=t30['date'], y=t30['forecast_inflow'], name='Inflows',
            marker_color='#27ae60', opacity=0.7,
        ), secondary_y=False)
        fig30.add_trace(go.Bar(
            x=t30['date'], y=-t30['forecast_outflow'], name='Outflows',
            marker_color='#e74c3c', opacity=0.7,
        ), secondary_y=False)

        # Balance line on secondary axis
        fig30.add_trace(go.Scatter(
            x=t30['date'], y=t30['closing_balance'], name='Closing Balance',
            mode='lines+markers', line=dict(color='#3498db', width=3),
            marker=dict(size=4),
        ), secondary_y=True)

        # Min / Max balance annotations
        min30_idx = t30['closing_balance'].idxmin()
        max30_idx = t30['closing_balance'].idxmax()
        min30_row = t30.loc[min30_idx]
        max30_row = t30.loc[max30_idx]
        fig30.add_annotation(
            x=min30_row['date'], y=min30_row['closing_balance'],
            text=f"Min: ${min30_row['closing_balance']:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor='#e74c3c',
            font=dict(color='#e74c3c', size=11, family='Arial Black'),
            bgcolor='white', bordercolor='#e74c3c', borderwidth=1, borderpad=3,
            ax=0, ay=40, yref='y2',
        )
        fig30.add_annotation(
            x=max30_row['date'], y=max30_row['closing_balance'],
            text=f"Max: ${max30_row['closing_balance']:,.0f}",
            showarrow=True, arrowhead=2, arrowcolor='#27ae60',
            font=dict(color='#27ae60', size=11, family='Arial Black'),
            bgcolor='white', bordercolor='#27ae60', borderwidth=1, borderpad=3,
            ax=0, ay=-40, yref='y2',
        )

        # Subtle vertical markers for key payment dates
        marker_configs = [
            ('PAYROLL', '#e74c3c', 'Payroll'),
            ('DEBT', '#e67e22', 'Debt Svc'),
            ('TAX', '#9b59b6', 'Tax'),
        ]
        bal_max = t30['closing_balance'].max()
        for col_name, color, label in marker_configs:
            key_dates = t30[t30[col_name] > 0]
            for _, row in key_dates.iterrows():
                fig30.add_vline(
                    x=row['date'].timestamp() * 1000,
                    line_width=1, line_dash='dot', line_color=color, opacity=0.4,
                )
                fig30.add_annotation(
                    x=row['date'], y=bal_max,
                    text=label, showarrow=False,
                    font=dict(size=8, color=color),
                    yshift=12, textangle=-45, yref='y2',
                )

        # Tight Y-axis range for balance (15% padding)
        bal_min = t30['closing_balance'].min()
        bal_range = bal_max - bal_min if bal_max != bal_min else bal_max * 0.1
        pad = bal_range * 0.15
        fig30.update_yaxes(
            title_text='Cash Flow ($)', tickformat='$,.0f', showgrid=False,
            secondary_y=False,
        )
        fig30.update_yaxes(
            title_text='Balance ($)', tickformat='$,.0f',
            range=[bal_min - pad, bal_max + pad],
            showgrid=True, gridcolor='#ecf0f1',
            secondary_y=True,
        )

        # Min-balance threshold line
        fig30.add_hline(y=5_000_000, line_dash='dash', line_color='#e74c3c', opacity=0.4,
                         annotation_text='Min ($5M)', annotation_position='bottom right',
                         annotation_font_size=9, annotation_font_color='#e74c3c',
                         secondary_y=True)

        fig30.update_layout(
            height=400, margin=dict(l=0, r=0, t=10, b=0),
            barmode='relative',
            xaxis=dict(tickformat='%b %d'),
            legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center'),
            hovermode='x unified',
        )
        st.plotly_chart(fig30, use_container_width=True, key='overview_t30')

        # Summary row
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric("Total Inflows", f"${t30['forecast_inflow'].sum():,.0f}")
        with s2:
            st.metric("Total Outflows", f"${t30['forecast_outflow'].sum():,.0f}")
        with s3:
            st.metric("Ending Balance", f"${t30['closing_balance'].iloc[-1]:,.0f}")
        with s4:
            pct_change = (t30['closing_balance'].iloc[-1] - t0_balance) / t0_balance * 100
            st.metric("30-Day Change", f"{pct_change:+.1f}%")

    # =================================================================
    # 5. ALERTS & RISK INDICATORS
    # =================================================================
    with st.container(border=True):
        st.markdown("### Alerts & Risk Indicators")
        alerts = []

        # Alert: T+1 large variance
        t1_abs_var = abs(summary['variance'])
        if t1_abs_var > 5_000_000:
            alerts.append(('critical', f"T+1 variance is ${t1_abs_var:,.0f} — exceeds $5M threshold"))
        elif t1_abs_var > 2_000_000:
            alerts.append(('warning', f"T+1 variance is ${t1_abs_var:,.0f} — exceeds $2M threshold"))

        # Alert: Low cash in T+7 window
        t7_min = t7['closing_balance'].min()
        t7_min_date = t7.loc[t7['closing_balance'].idxmin(), 'date']
        if t7_min < 5_000_000:
            alerts.append(('critical', f"Cash drops below $5M minimum on {t7_min_date.strftime('%b %d')} (${t7_min:,.0f})"))
        elif t7_min < 10_000_000:
            alerts.append(('warning', f"Cash approaches minimum on {t7_min_date.strftime('%b %d')} (${t7_min:,.0f})"))

        # Alert: Low cash in T+30 window
        t30_min = t30['closing_balance'].min()
        t30_min_date = t30.loc[t30['closing_balance'].idxmin(), 'date']
        if t30_min < 5_000_000:
            alerts.append(('critical', f"30-day outlook: cash below $5M on {t30_min_date.strftime('%b %d')} (${t30_min:,.0f})"))

        # Alert: Concentration risk — single outflow category > 40% of total on any T+7 day
        outflow_cats = ['PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT']
        for _, row in t7.iterrows():
            total_out = row['forecast_outflow']
            if total_out == 0:
                continue
            for cat in outflow_cats:
                if cat in row and row[cat] / total_out > 0.40:
                    alerts.append(('warning',
                        f"Concentration risk: {cat} is {row[cat]/total_out:.0%} of outflows "
                        f"on {row['date'].strftime('%b %d')} (${row[cat]:,.0f} / ${total_out:,.0f})"))
                    break  # one alert per day max
            else:
                continue
            break  # surface only the first concentration day

        # Alert: Model accuracy degradation
        for hz in ['T+7', 'T+30']:
            if hz in results and results[hz]['rating'] == 'Poor':
                alerts.append(('warning', f"{hz} model accuracy rated POOR ({results[hz]['balance_mape']:.1f}% error) — consider retraining"))

        # Render alerts
        if not alerts:
            st.markdown('<div class="alert-row">✅ &nbsp; <b>All clear</b> — No warnings or critical alerts.</div>',
                        unsafe_allow_html=True)
        else:
            for level, msg in alerts:
                if level == 'critical':
                    st.error(f"🔴 {msg}")
                else:
                    st.warning(f"⚠️ {msg}")

def render_daily_position():
    if not st.session_state.data_loaded:
        st.info("Loading data...")
        return
    st.subheader("T+1 Daily Cash Position")
    st.caption("Forecast vs Estimated Actuals for investment/borrowing decisions")
    forecaster = st.session_state.forecaster
    position_date = forecaster.last_actual_date.to_pydatetime() + timedelta(days=1)
    while position_date.weekday() >= 5:
        position_date += timedelta(days=1)
    t1_forecast = forecaster.predict('T+7')['T+7']
    manager = DailyPositionManager()
    opening_actual = forecaster.last_actual_closing_balance
    col1, col2 = st.columns([2, 1])
    with col1: st.markdown(f"**Position Date:** {position_date.strftime('%Y-%m-%d (%A)')}")
    with col2: opening_override = st.number_input("Opening Balance", value=float(opening_actual), step=100000.0, format="%.0f", key="opening_bal")
    manager.initialize_position(position_date=position_date, opening_balance_actual=opening_override, forecast_df=t1_forecast)
    st.markdown("---")
    use_simulated = st.checkbox("Load simulated intraday data", value=True)
    if use_simulated:
        intraday = simulate_intraday_data(position_date, t1_forecast)
        manager.load_intraday_from_bank(intraday['bank_transactions'])
        manager.load_sap_payment_queue(intraday['sap_payments'])
    position = manager.build_position()
    summary = manager.get_position_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Opening", f"${summary['opening_balance']:,.0f}")
    with col2: st.metric("Closing (Forecast)", f"${summary['closing_forecast']:,.0f}")
    with col3: st.metric("Closing (Estimated)", f"${summary['closing_estimated']:,.0f}")
    with col4: st.metric("Variance", f"${summary['variance']:,.0f}")
    st.markdown("---")
    st.subheader("Position Worksheet")
    display_df = position[['display_name', 'forecast', 'estimated_actual', 'variance', 'status']].copy()
    display_df.columns = ['Category', 'Forecast', 'Estimated Actual', 'Variance', 'Status']
    for col in ['Forecast', 'Estimated Actual', 'Variance']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True, height=500)
    st.markdown("---")
    st.subheader("Investment/Borrowing Recommendation")
    rec = manager.get_investment_borrowing_recommendation()
    if rec['action'] == 'INVEST': st.success(f"INVEST ${rec['amount']:,.0f}")
    elif rec['action'] == 'BORROW': st.error(f"BORROW ${rec['amount']:,.0f}")
    else: st.info("HOLD - No action needed")
    st.caption(rec['reasoning'])
    col1, col2 = st.columns(2)
    with col1: st.metric("Confidence", rec['confidence'])
    with col2: st.metric("Target Balance", f"${rec['target_balance']:,.0f}")

    # --- Historical Archive Section ---
    st.markdown("---")
    st.subheader("Historical Daily Archives")

    btn_cols = st.columns(3)
    with btn_cols[0]:
        if st.button("Archive Today", type="primary", use_container_width=True):
            archive_position(manager, DEFAULT_ARCHIVE_PATH)
            st.success("Archived today's position.")
            st.rerun()
    with btn_cols[1]:
        if st.button("Seed 30-Day History", use_container_width=True):
            with st.spinner("Generating 30 business days of history..."):
                seed_historical_archive(forecaster, num_days=30, storage_path=DEFAULT_ARCHIVE_PATH)
            st.success("Seeded 30 days of historical data.")
            st.rerun()
    with btn_cols[2]:
        if st.button("Clear Archive", use_container_width=True):
            clear_archive(DEFAULT_ARCHIVE_PATH)
            st.info("Archive cleared.")
            st.rerun()

    history = get_historical_accuracy(DEFAULT_ARCHIVE_PATH)
    if len(history) == 0:
        st.info("No archived history yet. Click **Seed 30-Day History** or **Archive Today** to get started.")
        return

    history = history.sort_values('position_date', ascending=False).reset_index(drop=True)

    # Date range filter
    all_dates = pd.to_datetime(history['position_date']).dt.date
    min_date, max_date = all_dates.min(), all_dates.max()
    filter_cols = st.columns(2)
    with filter_cols[0]:
        start_date = st.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="arch_start")
    with filter_cols[1]:
        end_date = st.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="arch_end")

    mask = (all_dates >= start_date) & (all_dates <= end_date)
    filtered = history[mask.values].copy()

    if len(filtered) == 0:
        st.warning("No data in selected range.")
        return

    # Summary table
    table_df = filtered[['position_date', 'opening_balance', 'closing_forecast', 'closing_actual',
                          'absolute_error', 'bias', 'bias_direction', 'posted_transactions']].copy()
    table_df.columns = ['Date', 'Opening', 'Closing Forecast', 'Closing Actual',
                         'Error', 'Bias', 'Direction', '# Txns']
    table_df['Date'] = pd.to_datetime(table_df['Date']).dt.strftime('%Y-%m-%d')
    for c in ['Opening', 'Closing Forecast', 'Closing Actual', 'Error', 'Bias']:
        table_df[c] = table_df[c].apply(lambda x: f"${x:,.0f}")
    st.dataframe(table_df, use_container_width=True, hide_index=True, height=min(400, 35 * len(table_df) + 38))

    # Accuracy trend chart
    trend = filtered.sort_values('position_date').copy()
    trend['position_date'] = pd.to_datetime(trend['position_date'])
    trend['rolling_rmse'] = trend['absolute_error'].rolling(window=7, min_periods=1).apply(
        lambda x: np.sqrt(np.mean(x ** 2))
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend['position_date'], y=trend['absolute_error'],
        mode='lines+markers', name='Daily Absolute Error',
        line=dict(color='#e74c3c', width=2),
    ))
    fig.add_trace(go.Scatter(
        x=trend['position_date'], y=trend['rolling_rmse'],
        mode='lines', name='7-Day Rolling RMSE',
        line=dict(color='#3498db', width=3, dash='dash'),
    ))
    fig.update_layout(
        title="Forecast Accuracy Over Time",
        xaxis_title="Date", yaxis_title="Error ($)",
        yaxis_tickformat='$,.0f', height=400,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)

def render_forecasts():
    if not st.session_state.data_loaded:
        st.info("Loading data...")
        return
    forecaster = st.session_state.forecaster
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'])
    st.markdown("---")
    st.subheader("Planned CAPEX")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        base_forecast = forecaster.predict(horizon)[horizon]
        dates = base_forecast['date'].dt.strftime('%Y-%m-%d').tolist()
        capex_date = st.selectbox("Date", dates, key='capex_date')
    with col2: capex_amt = st.number_input("Amount ($)", 0, 100_000_000, 0, 100_000, key='capex_amt')
    with col3:
        st.write("")
        st.write("")
        if st.button("Add") and capex_amt > 0:
            st.session_state.capex_schedule[capex_date] = capex_amt
            st.rerun()
    if st.session_state.capex_schedule:
        st.dataframe(pd.DataFrame([{'Date': k, 'Amount': f"${v:,.0f}"} for k, v in sorted(st.session_state.capex_schedule.items())]), hide_index=True)
        if st.button("Clear"): st.session_state.capex_schedule = {}; st.rerun()
    st.markdown("---")
    forecasts = forecaster.predict(horizon, capex_schedule=st.session_state.capex_schedule)
    fcast = forecasts[horizon].sort_values('date', ascending=True).copy()

    MIN_BALANCE_THRESHOLD = 5_000_000
    inflow_cats = ['AR', 'INV_INC', 'IC_IN']
    outflow_cats = ['PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT']

    # ── Derived columns ──
    fcast['net_flow'] = fcast['forecast_inflow'] - fcast['forecast_outflow']

    # ── Find lowest balance day & largest single outflow ──
    low_idx = fcast['closing_balance'].idxmin()
    high_idx = fcast['closing_balance'].idxmax()
    low_row = fcast.loc[low_idx]
    high_row = fcast.loc[high_idx]

    largest_out_val, largest_out_cat, largest_out_date = 0, '', None
    for _, r in fcast.iterrows():
        for cat in outflow_cats + ['CAPEX']:
            if cat in r and r[cat] > largest_out_val:
                largest_out_val, largest_out_cat, largest_out_date = r[cat], cat, r['date']

    # =================================================================
    # 1. KEY METRICS
    # =================================================================
    with st.container(border=True):
        st.markdown(f"### {horizon} Key Metrics")
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1:
            st.metric("Starting Balance", _fmt(fcast['opening_balance'].iloc[0]))
        with c2:
            end_bal = fcast['closing_balance'].iloc[-1]
            st.metric("Ending Balance", _fmt(end_bal))
        with c3:
            st.metric("Total Inflows", _fmt(fcast['forecast_inflow'].sum()))
        with c4:
            st.metric("Total Outflows", _fmt(fcast['forecast_outflow'].sum()))
        with c5:
            net = fcast['net_flow'].sum()
            st.metric("Net Change", _fmt(net),
                       delta=_fmt(net),
                       delta_color="normal" if net >= 0 else "inverse")
        with c6:
            low_icon = "🔴 " if low_row['closing_balance'] < MIN_BALANCE_THRESHOLD else ""
            st.metric("Lowest Balance",
                       _fmt(low_row['closing_balance']),
                       delta=f"{low_icon}{low_row['date'].strftime('%b %d (%a)')}")
        with c7:
            st.metric("Largest Outflow",
                       _fmt(largest_out_val),
                       delta=f"{largest_out_cat} — {largest_out_date.strftime('%b %d') if largest_out_date else ''}")

        if low_row['closing_balance'] < MIN_BALANCE_THRESHOLD:
            st.error(f"🔴 Balance drops below ${MIN_BALANCE_THRESHOLD/1e6:.0f}M minimum on "
                     f"{low_row['date'].strftime('%b %d')} ({_fmt(low_row['closing_balance'])})")

    # =================================================================
    # 2. CASH TRAJECTORY CHART (single Y-axis, candlestick bars + line)
    # =================================================================
    with st.container(border=True):
        st.markdown(f"### {horizon} Cash Trajectory")
        fig_traj = go.Figure()

        # Synthesize intraday high/low from opening + cumulative inflows/outflows
        # High = opening + inflows (peak before outflows settle)
        # Low  = opening - outflows (trough before inflows settle)
        fcast_high = fcast['opening_balance'] + fcast['forecast_inflow']
        fcast_low = fcast['opening_balance'] - fcast['forecast_outflow']

        # Candlestick: open/high/low/close per day (like a stock chart)
        fig_traj.add_trace(go.Candlestick(
            x=fcast['date'],
            open=fcast['opening_balance'],
            high=fcast_high,
            low=fcast_low,
            close=fcast['closing_balance'],
            name='Daily Range',
            increasing_line_color='#27ae60', increasing_fillcolor='rgba(39,174,96,0.3)',
            decreasing_line_color='#e74c3c', decreasing_fillcolor='rgba(231,76,60,0.3)',
        ))

        # Closing balance line on top
        fig_traj.add_trace(go.Scatter(
            x=fcast['date'], y=fcast['closing_balance'], name='Closing Balance',
            mode='lines+markers', line=dict(color='#3498db', width=3),
            marker=dict(size=7, color='#3498db'),
        ))

        # Min threshold band + line
        fig_traj.add_hrect(
            y0=0, y1=MIN_BALANCE_THRESHOLD,
            fillcolor='rgba(231,76,60,0.07)', line_width=0,
        )
        fig_traj.add_hline(
            y=MIN_BALANCE_THRESHOLD, line_dash='dash', line_color='#e74c3c', opacity=0.5,
            annotation_text=f'Min ({_fmt(MIN_BALANCE_THRESHOLD)})',
            annotation_position='bottom right',
            annotation_font_size=9, annotation_font_color='#e74c3c',
        )

        # Min / Max annotations
        fig_traj.add_annotation(
            x=low_row['date'], y=low_row['closing_balance'],
            text=f"Low: {_fmt(low_row['closing_balance'])}<br>{low_row['date'].strftime('%b %d')}",
            showarrow=True, arrowhead=2, arrowcolor='#e74c3c',
            font=dict(color='#e74c3c', size=11, family='Arial Black'),
            bgcolor='white', bordercolor='#e74c3c', borderwidth=1, borderpad=3,
            ax=-60, ay=40,
        )
        fig_traj.add_annotation(
            x=high_row['date'], y=high_row['closing_balance'],
            text=f"High: {_fmt(high_row['closing_balance'])}<br>{high_row['date'].strftime('%b %d')}",
            showarrow=True, arrowhead=2, arrowcolor='#27ae60',
            font=dict(color='#27ae60', size=11, family='Arial Black'),
            bgcolor='white', bordercolor='#27ae60', borderwidth=1, borderpad=3,
            ax=60, ay=-40,
        )

        # Tight Y-axis covering full high/low range (never start at zero)
        all_vals = pd.concat([fcast_low, fcast_high, fcast['closing_balance']])
        bal_min, bal_max = all_vals.min(), all_vals.max()
        bal_range = bal_max - bal_min if bal_max != bal_min else bal_max * 0.1
        pad = bal_range * 0.10
        fig_traj.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title='Cash Position ($)', tickformat='$,.0f',
                       range=[bal_min - pad, bal_max + pad],
                       gridcolor='#ecf0f1'),
            xaxis=dict(tickformat='%b %d (%a)', rangeslider=dict(visible=False)),
            legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
            hovermode='x unified',
        )
        st.plotly_chart(fig_traj, use_container_width=True, key='fcast_trajectory')

    # =================================================================
    # 4. DAILY DETAIL TABLE
    # =================================================================
    with st.container(border=True):
        st.markdown(f"### {horizon} Daily Detail")
        tbl = fcast[['date', 'day_name', 'opening_balance', 'forecast_inflow',
                      'forecast_outflow', 'net_flow', 'closing_balance']].copy()
        tbl.columns = ['Date', 'Day', 'Opening', 'Inflows', 'Outflows', 'Net', 'Closing']
        tbl['Date'] = tbl['Date'].dt.strftime('%Y-%m-%d')

        # Build flags column
        flags = []
        for _, r in fcast.iterrows():
            parts = []
            if r['closing_balance'] < MIN_BALANCE_THRESHOLD:
                parts.append('🔴 Below min')
            if r['net_flow'] < 0:
                parts.append('📉 Neg net')
            if 'is_payroll_day' in r and r.get('is_payroll_day'):
                parts.append('💰 Payroll')
            # Concentration check
            total_out = r['forecast_outflow']
            if total_out > 0:
                for cat in outflow_cats + ['CAPEX']:
                    if cat in r and r[cat] / total_out > 0.40:
                        parts.append(f'⚠️ {cat} {r[cat]/total_out:.0%}')
                        break
            flags.append(' | '.join(parts) if parts else '')
        tbl['Flags'] = flags

        for c in ['Opening', 'Inflows', 'Outflows', 'Net', 'Closing']:
            tbl[c] = tbl[c].apply(lambda x: f"${x:,.0f}")

        st.dataframe(tbl, use_container_width=True, hide_index=True,
                      height=min(450, 35 * len(tbl) + 38))

    # =================================================================
    # 5. VARIANCE / RISK INDICATORS
    # =================================================================
    with st.container(border=True):
        st.markdown(f"### {horizon} Risk Indicators")
        risks = []

        # Below-threshold days
        below_days = fcast[fcast['closing_balance'] < MIN_BALANCE_THRESHOLD]
        for _, r in below_days.iterrows():
            risks.append(('critical',
                f"Balance ${r['closing_balance']:,.0f} on {r['date'].strftime('%b %d (%a)')} "
                f"— below ${MIN_BALANCE_THRESHOLD/1e6:.0f}M minimum"))

        # Concentration risk per day
        for _, r in fcast.iterrows():
            total_out = r['forecast_outflow']
            if total_out == 0:
                continue
            for cat in outflow_cats + ['CAPEX']:
                if cat in r and r[cat] / total_out > 0.40:
                    risks.append(('warning',
                        f"Concentration: {cat} is {r[cat]/total_out:.0%} of outflows on "
                        f"{r['date'].strftime('%b %d')} (${r[cat]:,.0f} / ${total_out:,.0f})"))
                    break

        # Large negative net flow days
        big_neg = fcast[fcast['net_flow'] < -fcast['forecast_outflow'].mean()]
        for _, r in big_neg.iterrows():
            risks.append(('warning',
                f"Large net outflow ${r['net_flow']:,.0f} on {r['date'].strftime('%b %d (%a)')}"))

        if not risks:
            st.success("✅ No risk indicators — all days within normal parameters.")
        else:
            for level, msg in risks:
                if level == 'critical':
                    st.error(f"🔴 {msg}")
                else:
                    st.warning(f"⚠️ {msg}")

def render_accuracy():
    if not st.session_state.data_loaded:
        st.info("Loading data...")
        return
    results = st.session_state.backtest_results
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'], key='acc_hz')
    if horizon not in results: return
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Balance Accuracy", f"{results[horizon]['balance_mape']:.2f}% error", results[horizon]['rating'])
    with col2: st.metric("Inflow Accuracy", f"{results[horizon]['inflow_mape']:.2f}% error")
    with col3: st.metric("Outflow Accuracy", f"{results[horizon]['outflow_mape']:.2f}% error")
    with col4: st.metric("Samples", results[horizon]['samples'])
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_mape_by_horizon_chart(results[horizon]['mape_by_horizon_day']), use_container_width=True)
    with col2: st.plotly_chart(create_mape_by_dow_chart(results[horizon]['mape_by_dow']), use_container_width=True)
    st.plotly_chart(create_horizon_comparison_chart(results), use_container_width=True)

def render_shap():
    if not st.session_state.data_loaded:
        st.info("Loading data...")
        return
    shap_results = st.session_state.shap_results
    if not shap_results: return
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
        st.info("Loading data...")
        return
    outlier_detector = st.session_state.outlier_detector
    summary = outlier_detector.get_outlier_summary()
    outliers_df = outlier_detector.get_outliers()

    # Summary metrics
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

    if outliers_df is not None and len(outliers_df) > 0:
        st.subheader("Items Requiring Attention")

        for sev in ['High', 'Medium']:
            sev_df = outliers_df[outliers_df['severity'] == sev]
            if len(sev_df) == 0:
                continue

            icon = "🔴" if sev == 'High' else "🟡"
            st.markdown(f"#### {icon} {sev} Priority ({len(sev_df)} items)")

            for _, row in sev_df.iterrows():
                # Calculate variance for display
                variance = row['value'] - row['expected']
                variance_pct = (variance / abs(row['expected']) * 100) if row['expected'] != 0 else 0

                with st.expander(
                    f"{row['date'].strftime('%b %d, %Y')} ({row['day_name']}) — {row['anomaly_type']}",
                    expanded=(sev == 'High')
                ):
                    # Description
                    st.markdown(f"**{row['description']}**")

                    # Metrics row
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("Actual Amount", f"${row['value']:,.0f}")
                    with c2:
                        st.metric("Expected Amount", f"${row['expected']:,.0f}")
                    with c3:
                        delta_color = "inverse" if variance < 0 else "normal"
                        st.metric(
                            "Variance",
                            f"${abs(variance):,.0f}",
                            delta=f"{variance_pct:+.0f}%",
                            delta_color=delta_color
                        )

                    # Action
                    st.info(f"**Recommended Action:** {row['recommended_action']}")
    else:
        st.success("No outliers detected. All cash flows are within normal ranges.")

def main():
    init_session_state()
    render_sidebar()
    st.title("Cash Forecasting Intelligence")
    tabs = st.tabs(["Overview", "Daily Position", "Forecasts", "Accuracy", "Key Drivers", "Outliers"])
    with tabs[0]: render_overview()
    with tabs[1]: render_daily_position()
    with tabs[2]: render_forecasts()
    with tabs[3]: render_accuracy()
    with tabs[4]: render_shap()
    with tabs[5]: render_outliers()

if __name__ == "__main__":
    main()
