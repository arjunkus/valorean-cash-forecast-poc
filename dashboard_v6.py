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

def init_session_state():
    for key, val in {'data_loaded': False, 'daily_cash': None, 'category_df': None, 'forecaster': None, 'forecasts': None, 'backtest_results': None, 'shap_results': None, 'outlier_results': None, 'capex_schedule': {}}.items():
        if key not in st.session_state:
            st.session_state[key] = val

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
    fig.update_layout(title="Balance MAPE by Horizon Day", xaxis_title="Day", yaxis_title="MAPE %", height=350)
    return fig

def create_mape_by_dow_chart(mape_df):
    mape_df = mape_df.sort_values('day_of_week')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['balance_mape'], name='Balance', marker_color='#3498db'))
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['inflow_mape'], name='Inflow', marker_color='#27ae60'))
    fig.add_trace(go.Bar(x=mape_df['day_name'], y=mape_df['outflow_mape'], name='Outflow', marker_color='#e74c3c'))
    fig.update_layout(title="MAPE by Day of Week", barmode='group', height=350)
    return fig

def create_horizon_comparison_chart(results):
    horizons = list(results.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['balance_mape'] for h in horizons], name='Balance', marker_color='#3498db'))
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['inflow_mape'] for h in horizons], name='Inflow', marker_color='#27ae60'))
    fig.add_trace(go.Bar(x=horizons, y=[results[h]['outflow_mape'] for h in horizons], name='Outflow', marker_color='#e74c3c'))
    fig.update_layout(title="MAPE Across Horizons", barmode='group', height=350)
    return fig

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
            results, forecaster, forecasts, daily_errors = run_detailed_backtest(st.session_state.daily_cash, st.session_state.category_df, test_size=test_size)
            st.session_state.forecaster = forecaster
            st.session_state.forecasts = forecasts
            st.session_state.backtest_results = results
            st.session_state.daily_errors = daily_errors
            shap_analyzer = SHAPAnalyzer(forecaster)
            st.session_state.shap_results = shap_analyzer.analyze()
            outlier_detector = OutlierDetector()
            outlier_detector.detect(st.session_state.daily_cash.copy(), st.session_state.category_df)
            st.session_state.outlier_detector = outlier_detector
            st.session_state.data_loaded = True
        st.sidebar.success("Ready!")
    
    if st.session_state.data_loaded:
        st.sidebar.markdown("---")
        st.sidebar.metric("T0", st.session_state.forecaster.last_actual_date.strftime('%Y-%m-%d'))
        st.sidebar.metric("Balance", f"${st.session_state.forecaster.last_actual_closing_balance:,.0f}")

def render_overview():
    if not st.session_state.data_loaded:
        st.info("Click Load Data & Train")
        return
    forecaster = st.session_state.forecaster
    results = st.session_state.backtest_results
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("T0 Balance", f"${forecaster.last_actual_closing_balance:,.0f}")
    with col2: st.metric("T0 Date", forecaster.last_actual_date.strftime('%Y-%m-%d'))
    with col3:
        if 'T+7' in results: st.metric("T+7 MAPE", f"{results['T+7']['balance_mape']:.2f}%", results['T+7']['rating'])
    with col4:
        if 'T+30' in results: st.metric("T+30 MAPE", f"{results['T+30']['balance_mape']:.2f}%", results['T+30']['rating'])

def render_daily_position():
    if not st.session_state.data_loaded:
        st.info("Click Load Data & Train")
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
        st.info("Click Load Data & Train")
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
    fcast = forecasts[horizon].sort_values('date', ascending=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.metric("Opening", f"${forecaster.last_actual_closing_balance:,.0f}")
    with col2: st.metric("+ Receipts", f"${fcast['forecast_inflow'].sum():,.0f}")
    with col3: st.metric("- Payments", f"${fcast['forecast_outflow_ex_capex'].sum():,.0f}")
    with col4: st.metric("- CAPEX", f"${fcast['CAPEX'].sum():,.0f}")
    with col5: st.metric("= Closing", f"${fcast['closing_balance'].iloc[-1]:,.0f}")
    fig = create_category_chart(fcast, horizon)
    st.plotly_chart(fig, use_container_width=True)

def render_accuracy():
    if not st.session_state.data_loaded:
        st.info("Click Load Data & Train")
        return
    results = st.session_state.backtest_results
    horizon = st.selectbox("Horizon", ['T+7', 'T+30', 'T+90'], key='acc_hz')
    if horizon not in results: return
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Balance MAPE", f"{results[horizon]['balance_mape']:.2f}%", results[horizon]['rating'])
    with col2: st.metric("Inflow MAPE", f"{results[horizon]['inflow_mape']:.2f}%")
    with col3: st.metric("Outflow MAPE", f"{results[horizon]['outflow_mape']:.2f}%")
    with col4: st.metric("Samples", results[horizon]['samples'])
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(create_mape_by_horizon_chart(results[horizon]['mape_by_horizon_day']), use_container_width=True)
    with col2: st.plotly_chart(create_mape_by_dow_chart(results[horizon]['mape_by_dow']), use_container_width=True)
    st.plotly_chart(create_horizon_comparison_chart(results), use_container_width=True)

def render_shap():
    if not st.session_state.data_loaded:
        st.info("Click Load Data & Train")
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
        st.info("Click Load Data & Train")
        return
    outlier_detector = st.session_state.outlier_detector
    summary = outlier_detector.get_outlier_summary()
    outliers_df = outlier_detector.get_outliers()
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Days Analyzed", summary.get('total_days', 0))
    with col2: st.metric("Outliers Found", summary.get('outlier_count', 0))
    with col3: st.metric("High Severity", summary.get('by_severity', {}).get('High', 0))
    with col4: st.metric("Medium Severity", summary.get('by_severity', {}).get('Medium', 0))
    st.markdown("---")
    if outliers_df is not None and len(outliers_df) > 0:
        st.subheader("Actionable Items")
        for sev in ['High', 'Medium']:
            sev_df = outliers_df[outliers_df['severity'] == sev]
            if len(sev_df) > 0:
                st.markdown(f"#### {sev} Priority")
                for _, row in sev_df.iterrows():
                    with st.expander(f"{row['date'].strftime('%Y-%m-%d')} - {row['anomaly_type']}", expanded=(sev=='High')):
                        st.markdown(f"**{row['description']}**")
                        st.info(f"**Action:** {row['recommended_action']}")
                        c1, c2, c3 = st.columns(3)
                        with c1: st.metric("Actual", f"${row['value']:,.0f}")
                        with c2: st.metric("Expected", f"${row['expected']:,.0f}")
                        with c3: st.metric("Z-Score", f"{row['z_score']:.1f}")
    else:
        st.success("No actionable outliers detected.")

def main():
    init_session_state()
    render_sidebar()
    st.title("Cash Forecasting Intelligence")
    tabs = st.tabs(["Overview", "Daily Position", "Forecasts", "Accuracy", "SHAP", "Outliers"])
    with tabs[0]: render_overview()
    with tabs[1]: render_daily_position()
    with tabs[2]: render_forecasts()
    with tabs[3]: render_accuracy()
    with tabs[4]: render_shap()
    with tabs[5]: render_outliers()

if __name__ == "__main__":
    main()
