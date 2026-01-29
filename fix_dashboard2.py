# Fix the vline issue by using add_shape instead

with open('dashboard_v2.py', 'r') as f:
    content = f.read()

# Replace the problematic add_vline with add_shape
old_code = '''    # Add vertical line for T0
    t0_date = historical_df['date'].iloc[-1]
    # Convert to string for Plotly compatibility
    t0_str = t0_date.strftime('%Y-%m-%d') if hasattr(t0_date, 'strftime') else str(t0_date)
    fig.add_vline(
        x=t0_str,
        line_dash="dash",
        line_color="gray",
        annotation_text="T0 (Today)",
        annotation_position="top"
    )'''

new_code = '''    # Add vertical line for T0 using shapes (more compatible)
    t0_date = historical_df['date'].iloc[-1]
    fig.add_shape(
        type="line",
        x0=t0_date, x1=t0_date,
        y0=0, y1=1,
        yref="paper",
        line=dict(color="gray", width=2, dash="dash")
    )
    # Add T0 annotation separately
    fig.add_annotation(
        x=t0_date,
        y=1.05,
        yref="paper",
        text="T0 (Today)",
        showarrow=False,
        font=dict(size=12, color="gray")
    )'''

content = content.replace(old_code, new_code)

# Also fix the original version if the first fix didn't apply
old_code2 = '''    # Add vertical line for T0
    t0_date = historical_df['date'].iloc[-1]
    fig.add_vline(
        x=t0_date,
        line_dash="dash",
        line_color="gray",
        annotation_text="T0 (Today)",
        annotation_position="top"
    )'''

content = content.replace(old_code2, new_code)

with open('dashboard_v2.py', 'w') as f:
    f.write(content)

print("Fixed!")
