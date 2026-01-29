# Fix the combined view chart to properly align dates with horizon

fix_code = '''
def create_combined_view_chart(
    historical_df: pd.DataFrame,
    accuracy_trail: pd.DataFrame,
    forecasts: dict,
    horizon: str = 'T+7'
) -> go.Figure:
    """Create the combined view chart showing historical, trail, and forecast."""
    
    fig = go.Figure()
    
    # Determine how much history to show based on horizon
    horizon_days = {'T+1': 1, 'T+7': 7, 'T+30': 30, 'T+90': 90}.get(horizon, 7)
    
    # Show 2x the horizon days of history (or at least 14 days)
    history_days = max(horizon_days * 2, 14)
    
    # Get T0 date
    t0_date = historical_df['date'].iloc[-1]
    
    # 1. Historical actuals (limited to relevant period)
    hist = historical_df.tail(history_days)
    fig.add_trace(go.Scatter(
        x=hist['date'],
        y=hist['closing_balance'],
        mode='lines',
        name='Actual Balance',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='%{x|%Y-%m-%d}<br>Balance: $%{y:,.0f}<extra>Actual</extra>'
    ))
    
    # 2. Accuracy trail (if available)
    if accuracy_trail is not None and len(accuracy_trail) > 0:
        # Actual line in trail
        fig.add_trace(go.Scatter(
            x=accuracy_trail['date'],
            y=accuracy_trail['actual_balance'],
            mode='lines+markers',
            name='Actual (Trail)',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8),
            hovertemplate='%{x|%Y-%m-%d}<br>Actual: $%{y:,.0f}<extra>Trail Actual</extra>'
        ))
        
        # Forecast line in trail
        fig.add_trace(go.Scatter(
            x=accuracy_trail['date'],
            y=accuracy_trail['forecast_balance'],
            mode='lines+markers',
            name='Forecast (Trail)',
            line=dict(color='#ff7f0e', width=2, dash='dot'),
            marker=dict(size=6, symbol='diamond'),
            hovertemplate='%{x|%Y-%m-%d}<br>Forecast: $%{y:,.0f}<extra>Trail Forecast</extra>'
        ))
    
    # 3. Future forecast
    if horizon in forecasts:
        fcast = forecasts[horizon]
        
        # Main forecast line
        fig.add_trace(go.Scatter(
            x=fcast['date'],
            y=fcast['closing_balance'],
            mode='lines+markers',
            name=f'{horizon} Forecast',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=6),
            hovertemplate='%{x|%Y-%m-%d}<br>Forecast: $%{y:,.0f}<extra>Forecast</extra>'
        ))
        
        # Confidence interval
        if 'net_lower' in fcast.columns and 'net_upper' in fcast.columns:
            # Get starting balance
            start_balance = historical_df['closing_balance'].iloc[-1]
            
            # Calculate cumulative bounds
            lower_balances = []
            upper_balances = []
            cum_lower = 0
            cum_upper = 0
            
            for i in range(len(fcast)):
                cum_lower += fcast.iloc[i]['net_lower']
                cum_upper += fcast.iloc[i]['net_upper']
                lower_balances.append(start_balance + cum_lower)
                upper_balances.append(start_balance + cum_upper)
            
            # Add confidence band
            fig.add_trace(go.Scatter(
                x=pd.concat([fcast['date'], fcast['date'][::-1]]),
                y=lower_balances + upper_balances[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence',
                showlegend=True,
                hoverinfo='skip'
            ))
        
        # Set x-axis range to show history + forecast
        x_min = hist['date'].min()
        x_max = fcast['date'].max()
    else:
        x_min = hist['date'].min()
        x_max = t0_date
    
    # Add vertical line for T0 using shapes
    fig.add_shape(
        type="line",
        x0=t0_date, x1=t0_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    
    # Add T0 annotation
    fig.add_annotation(
        x=t0_date,
        y=1.05,
        yref="paper",
        text="T0 (Today)",
        showarrow=False,
        font=dict(size=12, color="gray")
    )
    
    fig.update_layout(
        title=f"Cash Position: Historical + {horizon} Forecast",
        xaxis_title="Date",
        yaxis_title="Cash Balance ($)",
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
        yaxis_tickformat='$,.0f',
        xaxis=dict(
            range=[x_min, x_max],
            tickformat='%Y-%m-%d'
        )
    )
    
    return fig
'''

# Read the dashboard file
with open('dashboard_v2.py', 'r') as f:
    content = f.read()

# Find and replace the create_combined_view_chart function
import re

# Pattern to match the entire function
pattern = r'def create_combined_view_chart\([^)]*\)[^}]*?(?=\ndef |\nclass |\Z)'

# Find the function
match = re.search(r'def create_combined_view_chart\(', content)
if match:
    start = match.start()
    
    # Find the next function definition
    next_func = re.search(r'\ndef create_mape_by_horizon_chart', content[start:])
    if next_func:
        end = start + next_func.start()
        
        # Replace the function
        content = content[:start] + fix_code + '\n\n' + content[end:]
        
        with open('dashboard_v2.py', 'w') as f:
            f.write(content)
        
        print("Fixed chart date alignment!")
    else:
        print("Could not find end of function")
else:
    print("Could not find function to replace")
