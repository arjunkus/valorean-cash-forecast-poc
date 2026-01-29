"""
Cash Forecasting Intelligence Dashboard
========================================
Interactive Streamlit dashboard for cash flow forecasting and analysis.

Features:
- Multi-horizon forecasting (RT+7, T+30, T+90, NT+365)
- Daily MAPE analysis (not weekly averages)
- Trend decomposition with STL
- SHAP feature importance
- Outlier detection
- Actionable recommendations
- SAP FQM integration ready
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

# Suppress warnings and TensorFlow logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import local modules
from config import (
    TIME_HORIZONS, MAPE_THRESHOLDS, DASHBOARD_CONFIG,
    COMPANY_CODES, CASH_FLOW_CATEGORIES
)
from data_simulator import SAPFQMSimulator, generate_sample_data
from models import CashFlowForecaster, run_backtest
from analysis import CashFlowAnalyzer, MAPEAnalyzer
from recommendations import RecommendationEngine, generate_recommendations, Severity

# Page configuration
st.set_page_config(
    page_title="Cash Forecasting Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-critical {
        background-color: #fee2e2;
        border-left: 4px solid #dc2626;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .recommendation-warning {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .recommendation-info {
        background-color: #e0f2fe;
        border-left: 4px solid #0284c7;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .recommendation-success {
        background-color: #dcfce7;
        border-left: 4px solid #16a34a;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'forecasts' not in st.session_state:
        st.session_state.forecasts = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'mape_results' not in st.session_state:
        st.session_state.mape_results = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None


# ============================================================================
# DATA LOADING AND MODEL TRAINING
# ============================================================================

@st.cache_data(ttl=3600)
def load_sample_data(periods: int = 730, random_seed: int = 42):
    """Load or generate sample data."""
    return generate_sample_data(periods=periods, random_seed=random_seed)


@st.cache_resource
def train_models(_daily_cash: pd.DataFrame):
    """Train all forecasting models."""
    return run_backtest(_daily_cash, test_size=90)


@st.cache_data
def run_analysis(_daily_cash: pd.DataFrame):
    """Run comprehensive analysis."""
    analyzer = CashFlowAnalyzer()
    return analyzer.full_analysis(_daily_cash)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_cash_position_chart(df: pd.DataFrame, forecasts: dict = None):
    """Create main cash position chart with actuals and forecasts."""
    fig = go.Figure()
    
    # Actual cash position
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['cash_position'],
        mode='lines',
        name='Actual Cash Position',
        line=dict(color=DASHBOARD_CONFIG['color_scheme']['actual'], width=2)
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
        last_date = df['date'].iloc[-1]
        
        for horizon, forecast_df in forecasts.items():
            if len(forecast_df) > 0:
                # Calculate cumulative cash position from forecast
                cumulative = last_actual + forecast_df['forecast'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=forecast_df['date'],
                    y=cumulative,
                    mode='lines',
                    name=f'Forecast ({horizon})',
                    line=dict(color=colors.get(horizon, '#888'), width=2, dash='dash')
                ))
                
                # Add confidence interval
                lower = last_actual + forecast_df['lower_bound'].cumsum()
                upper = last_actual + forecast_df['upper_bound'].cumsum()
                
                fig.add_trace(go.Scatter(
                    x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill='toself',
                    fillcolor=f'rgba({int(colors.get(horizon, "#888")[1:3], 16)}, '
                             f'{int(colors.get(horizon, "#888")[3:5], 16)}, '
                             f'{int(colors.get(horizon, "#888")[5:7], 16)}, 0.1)',
                    line=dict(color='rgba(0,0,0,0)'),
                    showlegend=False,
                    name=f'{horizon} CI'
                ))
    
    fig.update_layout(
        title='Cash Position: Actual vs Forecast',
        xaxis_title='Date',
        yaxis_title='Cash Position (USD)',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_inflow_outflow_chart(df: pd.DataFrame):
    """Create inflow vs outflow comparison chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Daily Cash Flows', 'Net Cash Flow')
    )
    
    # Inflows and Outflows
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['inflow'],
        name='Inflows',
        marker_color='#2ca02c'
    ), row=1, col=1)
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=-df['outflow'],
        name='Outflows',
        marker_color='#d62728'
    ), row=1, col=1)
    
    # Net Cash Flow
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df['net_cash_flow']]
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['net_cash_flow'],
        name='Net Cash Flow',
        marker_color=colors
    ), row=2, col=1)
    
    fig.update_layout(
        height=600,
        barmode='overlay',
        hovermode='x unified'
    )
    
    return fig


def create_mape_heatmap(mape_results: dict):
    """Create MAPE heatmap by day of week and horizon."""
    if 'daily_analysis' not in mape_results:
        return None
    
    daily_data = mape_results['daily_analysis']
    
    # Prepare data
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
        textfont={"size": 12},
        hovertemplate='%{y}<br>%{x}: %{z:.1f}% MAPE<extra></extra>'
    ))
    
    fig.update_layout(
        title='MAPE by Day of Week and Forecast Horizon',
        xaxis_title='Day of Week',
        yaxis_title='Forecast Horizon',
        height=350
    )
    
    return fig


def create_trend_chart(df: pd.DataFrame, trend_result):
    """Create trend decomposition chart."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Original + Trend', 'Weekly Seasonality', 'Residuals'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Original data
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['net_cash_flow'],
        mode='lines',
        name='Actual',
        line=dict(color='#1f77b4', width=1),
        opacity=0.5
    ), row=1, col=1)
    
    # Trend
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=trend_result.trend_values,
        mode='lines',
        name='Trend',
        line=dict(color='#ff7f0e', width=3)
    ), row=1, col=1)
    
    # Seasonality
    if 'weekly' in trend_result.seasonality:
        weekly = trend_result.seasonality['weekly']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        fig.add_trace(go.Bar(
            x=days,
            y=weekly,
            name='Weekly Pattern',
            marker_color='#2ca02c'
        ), row=2, col=1)
    
    # Residuals
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=trend_result.residuals,
        mode='lines',
        name='Residuals',
        line=dict(color='#9467bd', width=1)
    ), row=3, col=1)
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def create_shap_chart(shap_result):
    """Create SHAP feature importance chart."""
    if not shap_result.top_features:
        return None
    
    features = [f[0] for f in shap_result.top_features[:10]]
    importance = [f[1] for f in shap_result.top_features[:10]]
    effects = [shap_result.feature_effects.get(f, 'neutral') for f in features]
    
    colors = ['#2ca02c' if e == 'positive' else '#d62728' if e == 'negative' else '#888' 
              for e in effects]
    
    fig = go.Figure(go.Bar(
        x=importance[::-1],
        y=features[::-1],
        orientation='h',
        marker_color=colors[::-1],
        text=[f'{v:.3f}' for v in importance[::-1]],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='SHAP Feature Importance',
        xaxis_title='Mean |SHAP Value|',
        yaxis_title='Feature',
        height=400
    )
    
    return fig


def create_outlier_chart(df: pd.DataFrame, outlier_result):
    """Create outlier visualization chart."""
    fig = go.Figure()
    
    # Normal points
    normal_mask = ~df.index.isin(outlier_result.outlier_indices)
    fig.add_trace(go.Scatter(
        x=df.loc[normal_mask, 'date'],
        y=df.loc[normal_mask, 'net_cash_flow'],
        mode='markers',
        name='Normal',
        marker=dict(color='#1f77b4', size=5, opacity=0.5)
    ))
    
    # Outliers
    if len(outlier_result.outlier_indices) > 0:
        outlier_df = df.loc[outlier_result.outlier_indices]
        fig.add_trace(go.Scatter(
            x=outlier_df['date'],
            y=outlier_df['net_cash_flow'],
            mode='markers',
            name='Outliers',
            marker=dict(color='#d62728', size=12, symbol='x')
        ))
    
    fig.update_layout(
        title='Outlier Detection Results',
        xaxis_title='Date',
        yaxis_title='Net Cash Flow (USD)',
        height=400
    )
    
    return fig


def display_recommendations(recommendations):
    """Display recommendations with proper styling."""
    for rec in recommendations:
        severity_class = f"recommendation-{rec.severity.value}"
        icon = {
            'critical': '🚨',
            'warning': '⚠️',
            'info': 'ℹ️',
            'success': '✅'
        }.get(rec.severity.value, '📌')
        
        st.markdown(f"""
        <div class="{severity_class}">
            <strong>{icon} {rec.title}</strong><br>
            <em>{rec.description}</em><br><br>
            <strong>Action:</strong> {rec.action}<br>
            <strong>Impact:</strong> {rec.impact}
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    """Main dashboard function."""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">💰 Cash Forecasting Intelligence Dashboard</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Cash Flow Forecasting & Analysis | SAP FQM Integration Ready</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        st.subheader("Data Settings")
        data_periods = st.slider("Historical Days", 365, 1095, 730, 30)
        random_seed = st.number_input("Random Seed", 1, 100, 42)
        
        st.subheader("Forecast Horizons")
        selected_horizons = st.multiselect(
            "Select Horizons",
            options=list(TIME_HORIZONS.keys()),
            default=list(TIME_HORIZONS.keys())
        )
        
        st.divider()
        
        if st.button("🔄 Load Data & Train Models", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                st.session_state.data = load_sample_data(periods=data_periods, random_seed=random_seed)
                st.session_state.data_loaded = True
            
            with st.spinner("Training models (this may take a minute)..."):
                daily_cash = st.session_state.data['daily_cash_position']
                mape_results, forecaster, forecasts = train_models(daily_cash)
                
                st.session_state.forecaster = forecaster
                st.session_state.forecasts = forecasts
                st.session_state.mape_results = mape_results
                st.session_state.models_trained = True
            
            with st.spinner("Running analysis..."):
                st.session_state.analysis_results = run_analysis(daily_cash)
            
            with st.spinner("Generating recommendations..."):
                recommendations, summary = generate_recommendations(
                    daily_cash,
                    st.session_state.forecasts,
                    st.session_state.analysis_results,
                    st.session_state.mape_results
                )
                st.session_state.recommendations = recommendations
                st.session_state.rec_summary = summary
            
            st.success("✅ All models trained successfully!")
        
        if st.session_state.models_trained:
            st.divider()
            st.subheader("📊 Quick Stats")
            summary = st.session_state.rec_summary
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Critical", summary['critical_count'], 
                         delta_color="inverse" if summary['critical_count'] > 0 else "off")
            with col2:
                st.metric("Warnings", summary['warning_count'],
                         delta_color="inverse" if summary['warning_count'] > 0 else "off")
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train Models' in the sidebar to get started.")
        
        # Show sample architecture
        st.subheader("🏗️ System Architecture")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Data Layer**
            - SAP FQM Integration
            - Multi-company Support
            - Multi-currency (→ USD)
            - Category Breakdown
            """)
        
        with col2:
            st.markdown("""
            **Forecasting Models**
            - ARIMA (RT+7)
            - Prophet (T+30)
            - LSTM (T+90)
            - Ensemble (NT+365)
            """)
        
        with col3:
            st.markdown("""
            **Analytics**
            - Daily MAPE Analysis
            - Trend Decomposition
            - SHAP Explainability
            - Outlier Detection
            """)
        
        return
    
    # Data loaded - show dashboard
    daily_cash = st.session_state.data['daily_cash_position']
    fqm_flow = st.session_state.data['fqm_flow']
    
    # Key Metrics Row
    st.subheader("📈 Key Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_cash = daily_cash['cash_position'].iloc[-1]
    avg_daily_outflow = daily_cash['outflow'].mean()
    days_of_cash = current_cash / avg_daily_outflow
    
    with col1:
        st.metric(
            "Current Cash Position",
            f"${current_cash:,.0f}",
            f"{(daily_cash['cash_position'].iloc[-1] - daily_cash['cash_position'].iloc[-8]) / daily_cash['cash_position'].iloc[-8] * 100:.1f}% vs last week"
        )
    
    with col2:
        st.metric(
            "Days of Cash",
            f"{days_of_cash:.0f} days",
            "Healthy" if days_of_cash > 30 else "Low"
        )
    
    with col3:
        avg_inflow = daily_cash['inflow'].mean()
        st.metric(
            "Avg Daily Inflow",
            f"${avg_inflow:,.0f}"
        )
    
    with col4:
        st.metric(
            "Avg Daily Outflow",
            f"${avg_daily_outflow:,.0f}"
        )
    
    with col5:
        net_monthly = daily_cash['net_cash_flow'].tail(30).sum()
        st.metric(
            "Net Cash (30d)",
            f"${net_monthly:,.0f}",
            delta_color="normal"
        )
    
    st.divider()
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", 
        "🔮 Forecasts", 
        "📈 MAPE Analysis",
        "🔍 Trend & SHAP",
        "⚠️ Outliers",
        "💡 Recommendations"
    ])
    
    with tab1:
        st.subheader("Cash Position Overview")
        
        if st.session_state.models_trained:
            fig = create_cash_position_chart(daily_cash, st.session_state.forecasts)
        else:
            fig = create_cash_position_chart(daily_cash)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Inflows vs Outflows")
        
        # Date filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=daily_cash['date'].max() - timedelta(days=90)
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=daily_cash['date'].max()
            )
        
        filtered_df = daily_cash[
            (daily_cash['date'] >= pd.Timestamp(start_date)) &
            (daily_cash['date'] <= pd.Timestamp(end_date))
        ]
        
        fig2 = create_inflow_outflow_chart(filtered_df)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Category breakdown
        st.subheader("Cash Flow by Category")
        category_data = st.session_state.data['category_breakdown']
        
        col1, col2 = st.columns(2)
        
        with col1:
            inflow_cats = category_data[category_data['flow_type'] == 'INFLOW']
            inflow_summary = inflow_cats.groupby('category_name')['amount_usd'].sum().reset_index()
            fig_in = px.pie(inflow_summary, values='amount_usd', names='category_name',
                           title='Inflow Distribution', color_discrete_sequence=px.colors.qualitative.G10)
            st.plotly_chart(fig_in, use_container_width=True)
        
        with col2:
            outflow_cats = category_data[category_data['flow_type'] == 'OUTFLOW']
            outflow_summary = outflow_cats.groupby('category_name')['amount_usd'].sum().reset_index()
            fig_out = px.pie(outflow_summary, values='amount_usd', names='category_name',
                            title='Outflow Distribution', color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_out, use_container_width=True)
    
    with tab2:
        st.subheader("Forecast Results by Horizon")
        
        if not st.session_state.models_trained:
            st.warning("Please train models first by clicking the button in the sidebar.")
        else:
            forecasts = st.session_state.forecasts
            
            # Forecast selection
            horizon_select = st.selectbox("Select Horizon", list(forecasts.keys()))
            
            forecast_df = forecasts[horizon_select]
            
            # Display forecast chart
            fig = go.Figure()
            
            # Last 30 days of actual
            recent_actual = daily_cash.tail(30)
            fig.add_trace(go.Scatter(
                x=recent_actual['date'],
                y=recent_actual['net_cash_flow'],
                mode='lines',
                name='Actual (Last 30 days)',
                line=dict(color='#1f77b4', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['forecast'],
                mode='lines',
                name=f'Forecast ({horizon_select})',
                line=dict(color='#ff7f0e', width=2, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='95% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f'{horizon_select} Forecast: {TIME_HORIZONS[horizon_select]["model"]} Model',
                xaxis_title='Date',
                yaxis_title='Net Cash Flow (USD)',
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast data table
            st.subheader("Forecast Data")
            display_df = forecast_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df = display_df.round(2)
            st.dataframe(display_df, use_container_width=True)
    
    with tab3:
        st.subheader("MAPE Analysis by Day")
        
        if not st.session_state.models_trained:
            st.warning("Please train models first.")
        else:
            mape_results = st.session_state.mape_results
            
            # Summary metrics
            st.markdown("### Overall MAPE by Horizon")
            
            cols = st.columns(4)
            for i, (horizon, metrics) in enumerate(mape_results.items()):
                if horizon != "daily_analysis" and isinstance(metrics, dict):
                    with cols[i % 4]:
                        mape = metrics.get('mape', 0)
                        rating = metrics.get('rating', 'N/A')
                        
                        color = "#2ca02c" if rating == "Excellent" else "#f59e0b" if rating in ["Good", "Acceptable"] else "#d62728"
                        
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, {color}22, {color}11); border-radius: 10px;">
                            <h3 style="margin: 0; color: {color};">{mape:.1f}%</h3>
                            <p style="margin: 0; font-weight: bold;">{horizon}</p>
                            <p style="margin: 0; font-size: 0.9rem; color: {color};">{rating}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # MAPE Heatmap
            st.markdown("### MAPE Heatmap (by Day of Week)")
            fig_heatmap = create_mape_heatmap(mape_results)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Daily breakdown table
            st.markdown("### Detailed Daily MAPE")
            
            if 'daily_analysis' in mape_results:
                daily_data = mape_results['daily_analysis']
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                table_data = []
                for horizon, daily_mape in daily_data.items():
                    row = {"Horizon": horizon}
                    for i, day in enumerate(days):
                        row[day] = f"{daily_mape.get(i, 0):.1f}%"
                    table_data.append(row)
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                
                st.info("""
                💡 **Insights:**
                - Lower MAPE = Better accuracy
                - Weekend predictions often have higher errors due to reduced business activity
                - Monday/Friday may show higher errors due to week-start/end payment patterns
                """)
    
    with tab4:
        st.subheader("Trend Decomposition & SHAP Analysis")
        
        if not st.session_state.analysis_results:
            st.warning("Please train models first.")
        else:
            analysis = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 Trend Analysis")
                
                trend_result = analysis.get('trend')
                if trend_result:
                    # Trend summary
                    direction_emoji = "📈" if trend_result.trend_direction == "increasing" else "📉" if trend_result.trend_direction == "decreasing" else "➡️"
                    
                    st.markdown(f"""
                    **Trend Direction:** {direction_emoji} {trend_result.trend_direction.title()}
                    
                    **Monthly Slope:** ${trend_result.trend_slope * 30:,.0f}
                    
                    **Change Points Detected:** {len(trend_result.change_points)}
                    """)
                    
                    # Trend insights
                    for insight in trend_result.insights:
                        st.markdown(f"- {insight}")
            
            with col2:
                st.markdown("### 🎯 SHAP Feature Importance")
                
                shap_result = analysis.get('shap')
                if shap_result and shap_result.top_features:
                    fig_shap = create_shap_chart(shap_result)
                    if fig_shap:
                        st.plotly_chart(fig_shap, use_container_width=True)
            
            # Full trend chart
            st.markdown("### Trend Decomposition")
            if trend_result:
                fig_trend = create_trend_chart(daily_cash, trend_result)
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # SHAP insights
            if shap_result:
                st.markdown("### SHAP Insights")
                for insight in shap_result.insights:
                    st.markdown(f"- {insight}")
    
    with tab5:
        st.subheader("Outlier Detection")
        
        if not st.session_state.analysis_results:
            st.warning("Please train models first.")
        else:
            outlier_result = st.session_state.analysis_results.get('outliers')
            
            if outlier_result:
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Outliers", outlier_result.total_outliers)
                with col2:
                    st.metric("Outlier %", f"{outlier_result.outlier_pct:.1f}%")
                with col3:
                    st.metric("Detection Method", outlier_result.detection_method.title())
                
                # Outlier chart
                fig_outliers = create_outlier_chart(daily_cash, outlier_result)
                st.plotly_chart(fig_outliers, use_container_width=True)
                
                # Outlier details
                if outlier_result.total_outliers > 0:
                    st.markdown("### Detected Outliers")
                    
                    outlier_df = daily_cash.loc[outlier_result.outlier_indices].copy()
                    outlier_df['outlier_score'] = outlier_result.outlier_scores
                    outlier_df = outlier_df[['date', 'net_cash_flow', 'inflow', 'outflow', 'outlier_score']]
                    outlier_df['date'] = outlier_df['date'].dt.strftime('%Y-%m-%d')
                    outlier_df = outlier_df.sort_values('outlier_score', ascending=False)
                    
                    st.dataframe(outlier_df.head(20), use_container_width=True)
                
                # Insights
                st.markdown("### Insights")
                for insight in outlier_result.insights:
                    st.markdown(f"- {insight}")
    
    with tab6:
        st.subheader("💡 Actionable Recommendations")
        
        if not st.session_state.recommendations:
            st.warning("Please train models first.")
        else:
            recommendations = st.session_state.recommendations
            summary = st.session_state.rec_summary
            
            # Summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Recommendations", summary['total'])
            with col2:
                st.metric("Critical", summary['critical_count'])
            with col3:
                st.metric("Warnings", summary['warning_count'])
            with col4:
                st.metric("Categories", len(summary['by_category']))
            
            st.divider()
            
            # Filter
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
            
            # Display recommendations
            filtered_recs = [
                r for r in recommendations 
                if r.severity.value in severity_filter and r.category.value in category_filter
            ]
            
            if filtered_recs:
                display_recommendations(filtered_recs)
            else:
                st.info("No recommendations match the selected filters.")
            
            # Export option
            st.divider()
            if st.button("📥 Export Recommendations to CSV"):
                engine = RecommendationEngine()
                engine.recommendations = recommendations
                df = engine.to_dataframe()
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="recommendations.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.9rem;">
        Cash Forecasting Intelligence Dashboard | Powered by ARIMA, Prophet, LSTM & Ensemble Models<br>
        SAP FQM Integration Ready | Built for Enterprise Treasury Management
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
