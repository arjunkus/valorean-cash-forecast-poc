"""
Cash Forecasting Intelligence Dashboard - Enhanced Version
==========================================================
Complete interactive dashboard with all features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import os
import io

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import local modules
from config import (
    TIME_HORIZONS, MAPE_THRESHOLDS, DASHBOARD_CONFIG,
    COMPANY_CODES, CASH_FLOW_CATEGORIES, LIQUIDITY
)
from data_simulator_realistic import SAPFQMSimulator, generate_sample_data
from models_prophet import CashFlowForecaster, run_backtest
from analysis import CashFlowAnalyzer, MAPEAnalyzer
from recommendations import RecommendationEngine, generate_recommendations, Severity

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Cash Forecasting Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1f77b4, #17becf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Recommendation cards */
    .recommendation-critical {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 5px solid #dc2626;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 4px rgba(220, 38, 38, 0.1);
    }
    .recommendation-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.1);
    }
    .recommendation-info {
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        border-left: 5px solid #0284c7;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 4px rgba(2, 132, 199, 0.1);
    }
    .recommendation-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border-left: 5px solid #16a34a;
        padding: 1.2rem;
        margin: 0.8rem 0;
        border-radius: 0 12px 12px 0;
        box-shadow: 0 2px 4px rgba(22, 163, 74, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1.5rem;
        background-color: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.5rem;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Dividers */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #dee2e6, transparent);
    }
    
    /* Info boxes */
    .info-box {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress indicator */
    .progress-step {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #1f77b4;
        color: white;
        text-align: center;
        line-height: 30px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'data_loaded': False,
        'models_trained': False,
        'data': None,
        'forecaster': None,
        'forecasts': None,
        'analysis_results': None,
        'mape_results': None,
        'recommendations': None,
        'rec_summary': None,
        'training_progress': 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_sample_data(periods: int = 730, random_seed: int = 42):
    """Load or generate sample SAP FQM data."""
    return generate_sample_data(periods=periods, random_seed=random_seed)


@st.cache_resource(show_spinner=False)
def train_all_models(_daily_cash: pd.DataFrame, test_size: int = 90):
    """Train all forecasting models with backtest."""
    return run_backtest(_daily_cash, test_size=test_size)


@st.cache_data(show_spinner=False)
def run_full_analysis(_daily_cash: pd.DataFrame):
    """Run comprehensive analysis."""
    analyzer = CashFlowAnalyzer()
    return analyzer.full_analysis(_daily_cash)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_kpi_card(value: str, label: str, delta: str = None, color: str = "#1f77b4"):
    """Create a styled KPI card."""
    delta_html = f'<div style="color: {"#16a34a" if delta and delta.startswith("+") else "#dc2626"}; font-size: 0.9rem;">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """


def create_cash_position_chart(df: pd.DataFrame, forecasts: dict = None):
    """Create main cash position chart with actuals and forecasts."""
    fig = go.Figure()
    
    # Actual cash position
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cash_position'],
        mode='lines',
        name='Actual Cash Position',
        line=dict(color='#1f77b4', width=2.5),
        hovertemplate='<b>Date:</b> %{x}<br><b>Cash Position:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add forecasts if available
    if forecasts:
        colors = {
            'RT+7': '#ff7f0e',
            'T+30': '#2ca02c', 
            'T+90': '#d62728',
            'NT+365': '#9467bd'
        }
        
        last_actual = df['cash_position'].iloc[-1]
        
        for horizon, forecast_df in forecasts.items():
            if len(forecast_df) > 0:
                cumulative = last_actual + forecast_df['forecast'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=cumulative,
                    mode='lines',
                    name=f'Forecast ({horizon})',
                    line=dict(color=colors.get(horizon, '#888'), width=2, dash='dash'),
                    hovertemplate=f'<b>{horizon} Forecast</b><br>Date: %{{x}}<br>Projected: $%{{y:,.0f}}<extra></extra>'
                ))
                
                # Confidence interval
                lower = last_actual + forecast_df['lower_bound'].cumsum()
                upper = last_actual + forecast_df['upper_bound'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({int(colors.get(horizon, "#888888")[1:3], 16)}, '
                             f'{int(colors.get(horizon, "#888888")[3:5], 16)}, '
                             f'{int(colors.get(horizon, "#888888")[5:7], 16)}, 0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
    
    # Add threshold lines
    if 'outflow' in df.columns:
        avg_outflow = df['outflow'].mean()
        min_cash = avg_outflow * LIQUIDITY.minimum_cash_days
        target_cash = avg_outflow * LIQUIDITY.target_cash_days
        
        fig.add_hline(y=min_cash, line_dash="dot", line_color="red", 
                     annotation_text="Minimum Cash", annotation_position="right")
        fig.add_hline(y=target_cash, line_dash="dot", line_color="green",
                     annotation_text="Target Cash", annotation_position="right")
    
    fig.update_layout(
        title=dict(text='Cash Position: Actual vs Forecast', font=dict(size=20)),
        xaxis_title='Date',
        yaxis_title='Cash Position (USD)',
        yaxis_tickformat='$,.0f',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    
    return fig


def create_waterfall_chart(df: pd.DataFrame, n_days: int = 7):
    """Create cash flow waterfall chart for recent days."""
    recent = df.tail(n_days).copy()
    
    fig = go.Figure(go.Waterfall(
        name="Cash Flow",
        orientation="v",
        x=recent['date'].dt.strftime('%m/%d'),
        y=recent['net_cash_flow'],
        textposition="outside",
        text=[f"${x:,.0f}" for x in recent['net_cash_flow']],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#1f77b4"}}
    ))
    
    fig.update_layout(
        title=f'Net Cash Flow - Last {n_days} Days',
        xaxis_title='Date',
        yaxis_title='Net Cash Flow (USD)',
        yaxis_tickformat='$,.0f',
        height=400,
        showlegend=False
    )
    
    return fig


def create_mape_gauge(mape: float, horizon: str):
    """Create a gauge chart for MAPE."""
    thresholds = MAPE_THRESHOLDS.get(horizon, MAPE_THRESHOLDS["T+30"])
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=mape,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': f'{horizon} MAPE', 'font': {'size': 16}},
        delta={'reference': thresholds['good'], 'relative': False},
        gauge={
            'axis': {'range': [0, thresholds['poor'] * 1.5], 'tickwidth': 1},
            'bar': {'color': "#1f77b4"},
            'steps': [
                {'range': [0, thresholds['excellent']], 'color': '#dcfce7'},
                {'range': [thresholds['excellent'], thresholds['good']], 'color': '#fef3c7'},
                {'range': [thresholds['good'], thresholds['acceptable']], 'color': '#fed7aa'},
                {'range': [thresholds['acceptable'], thresholds['poor'] * 1.5], 'color': '#fee2e2'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': thresholds['acceptable']
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_mape_heatmap(mape_results: dict):
    """Create MAPE heatmap by day of week and horizon."""
    if 'daily_analysis' not in mape_results:
        return None
    
    daily_data = mape_results['daily_analysis']
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    horizons = list(daily_data.keys())
    
    z_data = []
    for horizon in horizons:
        row = [daily_data[horizon].get(i, 0) for i in range(7)]
        z_data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=days,
        y=horizons,
        colorscale='RdYlGn_r',
        text=[[f'{v:.1f}%' for v in row] for row in z_data],
        texttemplate='%{text}',
        textfont={"size": 14, "color": "black"},
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.1f}% MAPE<extra></extra>',
        colorbar=dict(title='MAPE %')
    ))
    
    fig.update_layout(
        title='Daily MAPE Analysis (Not Weekly Averages)',
        xaxis_title='Day of Week',
        yaxis_title='Forecast Horizon',
        height=350
    )
    
    return fig


def create_forecast_comparison_chart(forecasts: dict, daily_cash: pd.DataFrame):
    """Create comparison of all forecast horizons."""
    fig = make_subplots(rows=2, cols=2, subplot_titles=list(forecasts.keys()))
    
    colors = {'RT+7': '#ff7f0e', 'T+30': '#2ca02c', 'T+90': '#d62728', 'NT+365': '#9467bd'}
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    for (horizon, forecast_df), (row, col) in zip(forecasts.items(), positions):
        fig.add_trace(
            go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name=horizon,
                line=dict(color=colors.get(horizon, '#888'), width=2),
                showlegend=True
            ),
            row=row, col=col
        )
        
        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor=f'rgba(128, 128, 128, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=600, title_text="Forecast Comparison by Horizon")
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig


def create_trend_decomposition_chart(df: pd.DataFrame, trend_result):
    """Create comprehensive trend decomposition visualization."""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Original Data + Trend', 'Weekly Seasonality Pattern', 
                       'Monthly Seasonality Pattern', 'Residuals (Noise)'),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Original + Trend
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['net_cash_flow'],
        mode='lines', name='Actual', line=dict(color='#1f77b4', width=1), opacity=0.5
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=trend_result.trend_values,
        mode='lines', name='Trend', line=dict(color='#ff7f0e', width=3)
    ), row=1, col=1)
    
    # Weekly seasonality
    if 'weekly' in trend_result.seasonality:
        weekly = trend_result.seasonality['weekly']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in weekly]
        fig.add_trace(go.Bar(
            x=days, y=weekly, name='Weekly Pattern', marker_color=colors
        ), row=2, col=1)
    
    # Monthly seasonality
    if 'monthly' in trend_result.seasonality:
        monthly = trend_result.seasonality['monthly']
        colors = ['#2ca02c' if v >= 0 else '#d62728' for v in monthly]
        fig.add_trace(go.Bar(
            x=list(range(1, len(monthly) + 1)), y=monthly,
            name='Day of Month Pattern', marker_color=colors
        ), row=3, col=1)
    
    # Residuals
    fig.add_trace(go.Scatter(
        x=df['date'], y=trend_result.residuals,
        mode='lines', name='Residuals', line=dict(color='#9467bd', width=1)
    ), row=4, col=1)
    
    fig.update_layout(height=800, showlegend=True)
    fig.update_yaxes(tickformat='$,.0f', row=1, col=1)
    
    return fig


def create_shap_waterfall(shap_result, sample_idx: int = 0):
    """Create SHAP waterfall chart for a single prediction."""
    if not shap_result.top_features:
        return None
    
    features = [f[0] for f in shap_result.top_features[:8]]
    values = [shap_result.shap_values[sample_idx, i] for i, f in enumerate(shap_result.feature_importance.keys()) 
              if f in features][:8]
    
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="h",
        y=features[::-1],
        x=values[::-1],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#2ca02c"}},
        decreasing={"marker": {"color": "#d62728"}},
    ))
    
    fig.update_layout(
        title="SHAP Feature Contributions",
        xaxis_title="Impact on Prediction",
        height=400
    )
    
    return fig


def create_outlier_timeline(df: pd.DataFrame, outlier_result):
    """Create timeline visualization of outliers."""
    fig = go.Figure()
    
    # Normal points (smaller, more transparent)
    normal_mask = ~df.index.isin(outlier_result.outlier_indices)
    fig.add_trace(go.Scatter(
        x=df.loc[normal_mask, 'date'],
        y=df.loc[normal_mask, 'net_cash_flow'],
        mode='markers',
        name='Normal',
        marker=dict(color='#1f77b4', size=4, opacity=0.3),
        hovertemplate='<b>Normal</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
    ))
    
    # Outliers (larger, prominent)
    if len(outlier_result.outlier_indices) > 0:
        outlier_df = df.loc[outlier_result.outlier_indices]
        fig.add_trace(go.Scatter(
            x=outlier_df['date'],
            y=outlier_df['net_cash_flow'],
            mode='markers',
            name=f'Outliers ({len(outlier_result.outlier_indices)})',
            marker=dict(
                color='#d62728',
                size=12,
                symbol='diamond',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>OUTLIER</b><br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ))
    
    # Add standard deviation bands
    mean_val = df['net_cash_flow'].mean()
    std_val = df['net_cash_flow'].std()
    
    fig.add_hline(y=mean_val, line_dash="solid", line_color="gray", annotation_text="Mean")
    fig.add_hline(y=mean_val + 2*std_val, line_dash="dot", line_color="orange", annotation_text="+2σ")
    fig.add_hline(y=mean_val - 2*std_val, line_dash="dot", line_color="orange", annotation_text="-2σ")
    
    fig.update_layout(
        title='Outlier Detection Results',
        xaxis_title='Date',
        yaxis_title='Net Cash Flow (USD)',
        yaxis_tickformat='$,.0f',
        height=450,
        hovermode='closest'
    )
    
    return fig


def display_recommendations(recommendations, max_display: int = None):
    """Display recommendations with proper styling."""
    display_list = recommendations[:max_display] if max_display else recommendations
    
    for rec in display_list:
        severity_class = f"recommendation-{rec.severity.value}"
        icon = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }.get(rec.severity.value, '📌')
        
        st.markdown(f"""
        <div class="{severity_class}">
            <strong>{icon} {rec.title}</strong>
            <span style="float: right; font-size: 0.8rem; color: #666;">Priority: {rec.priority}</span>
            <br><br>
            <em>{rec.description}</em>
            <br><br>
            <strong>📋 Action:</strong> {rec.action}
            <br>
            <strong>📈 Impact:</strong> {rec.impact}
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main():
    """Main dashboard entry point."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">💰 Cash Forecasting Intelligence Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Multi-Horizon Forecasting • ARIMA • Prophet • LSTM • Ensemble | SAP FQM Integration Ready</p>', 
                unsafe_allow_html=True)
    
    # ==========================================================================
    # SIDEBAR
    # ==========================================================================
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/cash-in-hand.png", width=80)
        st.header("⚙️ Configuration")
        
        with st.expander("📊 Data Settings", expanded=True):
            data_periods = st.slider(
                "Historical Days",
                min_value=365, max_value=1095, value=365, step=30,
                help="Number of days of historical data to generate"
            )
            random_seed = st.number_input(
                "Random Seed",
                min_value=1, max_value=100, value=42,
                help="Seed for reproducible data generation"
            )
            test_size = st.slider(
                "Test Period (Days)",
                min_value=30, max_value=180, value=90, step=15,
                help="Days reserved for backtesting"
            )
        
        with st.expander("🎯 Model Settings", expanded=False):
            st.markdown("**Active Horizons:**")
            horizons = {}
            for hz in TIME_HORIZONS.keys():
                horizons[hz] = st.checkbox(hz, value=True, key=f"hz_{hz}")
        
        st.divider()
        
        # Main action button
        if st.button("🚀 Load Data & Train Models", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Initializing...")
            
            try:
                # Step 1: Load data
                progress_bar.progress(10, text="📊 Generating SAP FQM data...")
                st.session_state.data = load_sample_data(periods=data_periods, random_seed=random_seed)
                st.session_state.data_loaded = True
                
                # Step 2: Train models
                progress_bar.progress(30, text="🤖 Training ARIMA model...")
                daily_cash = st.session_state.data['daily_cash_position']
                
                progress_bar.progress(50, text="🤖 Training Prophet & LSTM models...")
                mape_results, forecaster, forecasts = train_all_models(daily_cash, test_size=test_size)
                
                st.session_state.forecaster = forecaster
                st.session_state.forecasts = forecasts
                st.session_state.mape_results = mape_results
                st.session_state.models_trained = True
                
                # Step 3: Run analysis
                progress_bar.progress(75, text="📈 Running comprehensive analysis...")
                st.session_state.analysis_results = run_full_analysis(daily_cash)
                
                # Step 4: Generate recommendations
                progress_bar.progress(90, text="💡 Generating recommendations...")
                recommendations, summary = generate_recommendations(
                    daily_cash,
                    st.session_state.forecasts,
                    st.session_state.analysis_results,
                    st.session_state.mape_results
                )
                st.session_state.recommendations = recommendations
                st.session_state.rec_summary = summary
                
                progress_bar.progress(100, text="✅ Complete!")
                st.success("All models trained successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                progress_bar.empty()
        
        # Quick stats after training
        if st.session_state.models_trained:
            st.divider()
            st.subheader("📊 Quick Stats")
            
            summary = st.session_state.rec_summary
            col1, col2 = st.columns(2)
            
            with col1:
                critical_color = "#dc2626" if summary['critical_count'] > 0 else "#16a34a"
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: {critical_color}22; border-radius: 8px;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: {critical_color};">{summary['critical_count']}</div>
                    <div style="font-size: 0.8rem; color: #666;">Critical</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                warning_color = "#f59e0b" if summary['warning_count'] > 0 else "#16a34a"
                st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: {warning_color}22; border-radius: 8px;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: {warning_color};">{summary['warning_count']}</div>
                    <div style="font-size: 0.8rem; color: #666;">Warnings</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; font-size: 0.8rem; color: #888;">
            v1.0.0 | Enterprise Treasury
        </div>
        """, unsafe_allow_html=True)
    
    # ==========================================================================
    # MAIN CONTENT
    # ==========================================================================
    if not st.session_state.data_loaded:
        # Landing page
        st.info("👈 Click **'Load Data & Train Models'** in the sidebar to get started.")
        
        st.markdown("---")
        st.subheader("🏗️ System Architecture")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            **📊 Data Layer**
            - SAP FQM Integration
            - Multi-company Support
            - Multi-currency → USD
            - 2 Years Historical
            """)
        
        with col2:
            st.markdown("""
            **🤖 Forecasting**
            - ARIMA (RT+7)
            - Prophet (T+30)
            - LSTM (T+90)
            - Ensemble (NT+365)
            """)
        
        with col3:
            st.markdown("""
            **📈 Analytics**
            - **Daily** MAPE Analysis
            - Trend Decomposition
            - SHAP Explainability
            - Outlier Detection
            """)
        
        with col4:
            st.markdown("""
            **💡 Insights**
            - Liquidity Alerts
            - Accuracy Monitoring
            - Trend Warnings
            - Action Items
            """)
        
        st.markdown("---")
        st.subheader("📋 Forecast Horizons & Models")
        
        horizon_data = []
        for hz, config in TIME_HORIZONS.items():
            thresholds = MAPE_THRESHOLDS[hz]
            horizon_data.append({
                "Horizon": hz,
                "Days": config['days'],
                "Model": config['model'],
                "Description": config['description'],
                "Target MAPE": f"< {thresholds['good']}%",
            })
        
        st.dataframe(pd.DataFrame(horizon_data), use_container_width=True, hide_index=True)
        
        return
    
    # ==========================================================================
    # DATA LOADED - SHOW FULL DASHBOARD
    # ==========================================================================
    daily_cash = st.session_state.data['daily_cash_position']
    fqm_flow = st.session_state.data['fqm_flow']
    
    # KPI Row
    current_cash = daily_cash['cash_position'].iloc[-1]
    prev_week_cash = daily_cash['cash_position'].iloc[-8] if len(daily_cash) > 8 else current_cash
    cash_change = (current_cash - prev_week_cash) / prev_week_cash * 100
    
    avg_daily_outflow = daily_cash['outflow'].mean()
    days_of_cash = current_cash / avg_daily_outflow
    
    net_30d = daily_cash['net_cash_flow'].tail(30).sum()
    avg_inflow = daily_cash['inflow'].mean()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💵 Cash Position",
            f"${current_cash:,.0f}",
            f"{cash_change:+.1f}% vs last week"
        )
    
    with col2:
        days_color = "normal" if days_of_cash >= 30 else "inverse"
        st.metric(
            "📅 Days of Cash",
            f"{days_of_cash:.0f} days",
            "Healthy" if days_of_cash >= 30 else "Low",
            delta_color=days_color
        )
    
    with col3:
        st.metric("📥 Avg Daily Inflow", f"${avg_inflow:,.0f}")
    
    with col4:
        st.metric("📤 Avg Daily Outflow", f"${avg_daily_outflow:,.0f}")
    
    with col5:
        net_color = "normal" if net_30d >= 0 else "inverse"
        st.metric("📊 Net Cash (30d)", f"${net_30d:,.0f}", delta_color=net_color)
    
    st.divider()
    
    # ==========================================================================
    # MAIN TABS
    # ==========================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "🔮 Forecasts", 
        "📈 MAPE Analysis",
        "🔍 Trends & SHAP",
        "⚠️ Outliers",
        "💡 Recommendations",
        "📥 Export"
    ])
    
    # --------------------------------------------------------------------------
    # TAB 1: OVERVIEW
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("Cash Position Overview")
        
        if st.session_state.models_trained:
            fig = create_cash_position_chart(daily_cash, st.session_state.forecasts)
        else:
            fig = create_cash_position_chart(daily_cash)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Cash Flow Waterfall")
            waterfall_days = st.slider("Days to show", 5, 30, 14, key="waterfall_days")
            fig_waterfall = create_waterfall_chart(daily_cash, waterfall_days)
            st.plotly_chart(fig_waterfall, use_container_width=True)
        
        with col2:
            st.subheader("Category Distribution")
            
            category_data = st.session_state.data['category_breakdown']
            
            # Inflows pie
            inflow_cats = category_data[category_data['flow_type'] == 'INFLOW']
            inflow_summary = inflow_cats.groupby('category_name')['amount_usd'].sum().reset_index()
            fig_in = px.pie(
                inflow_summary, values='amount_usd', names='category_name',
                title='Inflows', color_discrete_sequence=px.colors.qualitative.G10,
                hole=0.4
            )
            fig_in.update_layout(height=300, showlegend=True, legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_in, use_container_width=True)
            
            # Outflows pie
            outflow_cats = category_data[category_data['flow_type'] == 'OUTFLOW']
            outflow_summary = outflow_cats.groupby('category_name')['amount_usd'].sum().reset_index()
            fig_out = px.pie(
                outflow_summary, values='amount_usd', names='category_name',
                title='Outflows', color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            fig_out.update_layout(height=300, showlegend=True, legend=dict(font=dict(size=10)))
            st.plotly_chart(fig_out, use_container_width=True)
    
    # --------------------------------------------------------------------------
    # TAB 2: FORECASTS
    # --------------------------------------------------------------------------
    with tab2:
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first using the sidebar button.")
        else:
            st.subheader("Multi-Horizon Forecast Results")
            
            forecasts = st.session_state.forecasts
            
            # Comparison chart
            fig_compare = create_forecast_comparison_chart(forecasts, daily_cash)
            st.plotly_chart(fig_compare, use_container_width=True)
            
            st.divider()
            
            # Individual horizon details
            horizon_select = st.selectbox(
                "Select Horizon for Details",
                list(forecasts.keys()),
                format_func=lambda x: f"{x} - {TIME_HORIZONS[x]['description']} ({TIME_HORIZONS[x]['model']})"
            )
            
            forecast_df = forecasts[horizon_select]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                # Recent actuals
                recent = daily_cash.tail(60)
                fig.add_trace(go.Scatter(
                    x=recent['date'], y=recent['net_cash_flow'],
                    mode='lines', name='Actual (Last 60 days)',
                    line=dict(color='#1f77b4', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'], y=forecast_df['forecast'],
                    mode='lines', name=f'Forecast ({horizon_select})',
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                # CI
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                    y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                    fill='toself', fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(0,0,0,0)'), name='95% CI'
                ))
                
                fig.update_layout(
                    title=f'{horizon_select} Forecast Detail - {TIME_HORIZONS[horizon_select]["model"]} Model',
                    xaxis_title='Date', yaxis_title='Net Cash Flow (USD)',
                    yaxis_tickformat='$,.0f', height=450, hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Forecast Statistics**")
                
                stats = {
                    "Mean Forecast": f"${forecast_df['forecast'].mean():,.0f}",
                    "Max Forecast": f"${forecast_df['forecast'].max():,.0f}",
                    "Min Forecast": f"${forecast_df['forecast'].min():,.0f}",
                    "Std Dev": f"${forecast_df['forecast'].std():,.0f}",
                    "Forecast Days": len(forecast_df),
                }
                
                for label, value in stats.items():
                    st.metric(label, value)
            
            # Forecast table
            with st.expander("📋 View Forecast Data Table"):
                display_df = forecast_df.copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                for col in ['forecast', 'lower_bound', 'upper_bound']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # --------------------------------------------------------------------------
    # TAB 3: MAPE ANALYSIS
    # --------------------------------------------------------------------------
    with tab3:
        if not st.session_state.models_trained:
            st.warning("⚠️ Please train models first.")
        else:
            st.subheader("📊 Daily MAPE Analysis (Not Weekly Averages)")
            st.info("💡 This analysis shows forecast accuracy at the **daily level** for each day of the week, providing granular insights into when forecasts perform best and worst.")
            
            mape_results = st.session_state.mape_results
            
            # Gauge charts for overall MAPE
            st.markdown("### Overall MAPE by Horizon")
            cols = st.columns(4)
            
            for i, (horizon, metrics) in enumerate(mape_results.items()):
                if horizon != "daily_analysis" and isinstance(metrics, dict):
                    with cols[i % 4]:
                        mape = metrics.get('mape', 0)
                        fig_gauge = create_mape_gauge(mape, horizon)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        rating = metrics.get('rating', 'N/A')
                        color = "#16a34a" if rating == "Excellent" else "#f59e0b" if rating in ["Good", "Acceptable"] else "#dc2626"
                        st.markdown(f"<div style='text-align: center; color: {color}; font-weight: bold;'>{rating}</div>", 
                                   unsafe_allow_html=True)
            
            st.divider()
            
            # Heatmap
            st.markdown("### Daily MAPE Heatmap")
            fig_heatmap = create_mape_heatmap(mape_results)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Detailed table
            st.markdown("### Detailed Daily MAPE Table")
            
            if 'daily_analysis' in mape_results:
                daily_data = mape_results['daily_analysis']
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                table_data = []
                for horizon, daily_mape in daily_data.items():
                    row = {"Horizon": horizon}
                    for i, day in enumerate(days):
                        val = daily_mape.get(i, 0)
                        row[day] = f"{val:.1f}%"
                    
                    # Calculate stats
                    values = [daily_mape.get(i, 0) for i in range(7)]
                    row["Best"] = days[np.argmin(values)]
                    row["Worst"] = days[np.argmax(values)]
                    table_data.append(row)
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            
            # Insights
            st.markdown("### 💡 Key Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Understanding MAPE Patterns:**
                - **Weekend dips**: Reduced business activity leads to more predictable (or less variable) cash flows
                - **Monday spikes**: Week-start payment processing creates volatility
                - **Month-end effects**: Settlement cycles cause predictable but large swings
                """)
            
            with col2:
                st.markdown("""
                **Improvement Actions:**
                - Focus model tuning on worst-performing days
                - Add day-of-week features if not already included
                - Consider separate models for weekday vs weekend
                - Review data quality for high-error periods
                """)
    
    # --------------------------------------------------------------------------
    # TAB 4: TRENDS & SHAP
    # --------------------------------------------------------------------------
    with tab4:
        if not st.session_state.analysis_results:
            st.warning("⚠️ Please train models first.")
        else:
            analysis = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Trend Analysis")
                
                trend_result = analysis.get('trend')
                if trend_result:
                    direction_emoji = {
                        "increasing": "📈",
                        "decreasing": "📉", 
                        "stable": "➡️"
                    }.get(trend_result.trend_direction, "❓")
                    
                    monthly_change = trend_result.trend_slope * 30
                    
                    st.markdown(f"""
                    <div class="info-box">
                        <h4>{direction_emoji} Trend: {trend_result.trend_direction.title()}</h4>
                        <p><strong>Monthly Slope:</strong> ${monthly_change:,.0f}</p>
                        <p><strong>Change Points:</strong> {len(trend_result.change_points)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for insight in trend_result.insights:
                        st.markdown(f"• {insight}")
            
            with col2:
                st.markdown("### 🎯 SHAP Feature Importance")
                
                shap_result = analysis.get('shap')
                if shap_result and shap_result.top_features:
                    # Bar chart
                    features = [f[0] for f in shap_result.top_features[:10]]
                    importance = [f[1] for f in shap_result.top_features[:10]]
                    effects = [shap_result.feature_effects.get(f, 'neutral') for f in features]
                    colors = ['#2ca02c' if e == 'positive' else '#d62728' if e == 'negative' else '#888' for e in effects]
                    
                    fig = go.Figure(go.Bar(
                        x=importance[::-1], y=features[::-1],
                        orientation='h', marker_color=colors[::-1],
                        text=[f'{v:.3f}' for v in importance[::-1]],
                        textposition='outside'
                    ))
                    fig.update_layout(
                        title='Feature Importance', height=400,
                        xaxis_title='Mean |SHAP Value|'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Full trend decomposition
            st.divider()
            st.markdown("### Trend Decomposition")
            
            if trend_result:
                fig_trend = create_trend_decomposition_chart(daily_cash, trend_result)
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # SHAP insights
            if shap_result:
                st.markdown("### SHAP Insights")
                for insight in shap_result.insights:
                    st.markdown(f"• {insight}")
    
    # --------------------------------------------------------------------------
    # TAB 5: OUTLIERS
    # --------------------------------------------------------------------------
    with tab5:
        if not st.session_state.analysis_results:
            st.warning("⚠️ Please train models first.")
        else:
            outlier_result = st.session_state.analysis_results.get('outliers')
            
            if outlier_result:
                st.subheader("⚠️ Outlier Detection Results")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Outliers", outlier_result.total_outliers)
                with col2:
                    st.metric("Outlier %", f"{outlier_result.outlier_pct:.1f}%")
                with col3:
                    st.metric("Method", outlier_result.detection_method.title())
                with col4:
                    status = "✅ Normal" if outlier_result.outlier_pct < 3 else "⚠️ High"
                    st.metric("Status", status)
                
                # Timeline chart
                fig_outliers = create_outlier_timeline(daily_cash, outlier_result)
                st.plotly_chart(fig_outliers, use_container_width=True)
                
                # Outlier table
                if outlier_result.total_outliers > 0:
                    st.markdown("### 📋 Detected Outliers")
                    
                    outlier_df = daily_cash.loc[outlier_result.outlier_indices].copy()
                    outlier_df['outlier_score'] = outlier_result.outlier_scores
                    outlier_df = outlier_df[['date', 'net_cash_flow', 'inflow', 'outflow', 'outlier_score']]
                    outlier_df = outlier_df.sort_values('outlier_score', ascending=False)
                    outlier_df['date'] = outlier_df['date'].dt.strftime('%Y-%m-%d')
                    
                    # Format currency
                    for col in ['net_cash_flow', 'inflow', 'outflow']:
                        outlier_df[col] = outlier_df[col].apply(lambda x: f"${x:,.0f}")
                    outlier_df['outlier_score'] = outlier_df['outlier_score'].apply(lambda x: f"{x:.2f}")
                    
                    st.dataframe(outlier_df.head(20), use_container_width=True, hide_index=True)
                
                # Insights
                st.markdown("### 💡 Insights")
                for insight in outlier_result.insights:
                    st.markdown(f"• {insight}")
    
    # --------------------------------------------------------------------------
    # TAB 6: RECOMMENDATIONS
    # --------------------------------------------------------------------------
    with tab6:
        if not st.session_state.recommendations:
            st.warning("⚠️ Please train models first.")
        else:
            recommendations = st.session_state.recommendations
            summary = st.session_state.rec_summary
            
            st.subheader("💡 Actionable Recommendations")
            
            # Summary cards
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total", summary['total'])
            with col2:
                st.metric("🚨 Critical", summary['critical_count'])
            with col3:
                st.metric("⚠️ Warnings", summary['warning_count'])
            with col4:
                st.metric("Categories", len(summary['by_category']))
            
            st.divider()
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=[s.value for s in Severity],
                    default=[s.value for s in Severity]
                )
            with col2:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=list(summary['by_category'].keys()),
                    default=list(summary['by_category'].keys())
                )
            
            # Display filtered recommendations
            filtered_recs = [
                r for r in recommendations
                if r.severity.value in severity_filter and r.category.value in category_filter
            ]
            
            if filtered_recs:
                display_recommendations(filtered_recs)
            else:
                st.info("No recommendations match the selected filters.")
    
    # --------------------------------------------------------------------------
    # TAB 7: EXPORT
    # --------------------------------------------------------------------------
    with tab7:
        st.subheader("📥 Export Data & Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Data Exports")
            
            # Daily cash position
            if st.button("📥 Download Daily Cash Position (CSV)", use_container_width=True):
                csv = daily_cash.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name=f"daily_cash_position_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # FQM Flow data
            if st.button("📥 Download FQM Flow Data (CSV)", use_container_width=True):
                csv = fqm_flow.to_csv(index=False)
                st.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name=f"fqm_flow_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            # Forecasts
            if st.session_state.models_trained:
                if st.button("📥 Download All Forecasts (CSV)", use_container_width=True):
                    all_forecasts = pd.concat([
                        df.assign(horizon=hz) for hz, df in st.session_state.forecasts.items()
                    ])
                    csv = all_forecasts.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name=f"forecasts_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        with col2:
            st.markdown("### 📋 Report Exports")
            
            # Recommendations
            if st.session_state.recommendations:
                if st.button("📥 Download Recommendations (CSV)", use_container_width=True):
                    rec_data = []
                    for rec in st.session_state.recommendations:
                        rec_data.append({
                            "ID": rec.id,
                            "Priority": rec.priority,
                            "Severity": rec.severity.value,
                            "Category": rec.category.value,
                            "Title": rec.title,
                            "Description": rec.description,
                            "Action": rec.action,
                            "Impact": rec.impact
                        })
                    csv = pd.DataFrame(rec_data).to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name=f"recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # MAPE Results
            if st.session_state.mape_results:
                if st.button("📥 Download MAPE Analysis (CSV)", use_container_width=True):
                    mape_data = []
                    for horizon, metrics in st.session_state.mape_results.items():
                        if horizon != "daily_analysis" and isinstance(metrics, dict):
                            mape_data.append({
                                "Horizon": horizon,
                                "MAPE": metrics.get('mape', 0),
                                "Rating": metrics.get('rating', 'N/A'),
                                "MAE": metrics.get('mae', 0),
                                "RMSE": metrics.get('rmse', 0)
                            })
                    csv = pd.DataFrame(mape_data).to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name=f"mape_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        
        st.divider()
        
        st.markdown("### 📧 Schedule Reports")
        st.info("📌 **Coming Soon**: Automated email reports, Slack integration, and scheduled data refreshes.")
    
    # ==========================================================================
    # FOOTER
    # ==========================================================================
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.85rem; padding: 1rem;">
        <strong>Cash Forecasting Intelligence Dashboard</strong> v1.0.0<br>
        Powered by ARIMA • Prophet • LSTM • Ensemble Models<br>
        SAP FQM Integration Ready | Enterprise Treasury Management
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    main()
