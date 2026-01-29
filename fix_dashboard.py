# Fix the Timestamp issue in dashboard_v2.py

with open('dashboard_v2.py', 'r') as f:
    content = f.read()

# Fix the add_vline issue - convert Timestamp to string
old_code = '''    # Add vertical line for T0
    t0_date = historical_df['date'].iloc[-1]
    fig.add_vline(
        x=t0_date,
        line_dash="dash",
        line_color="gray",
        annotation_text="T0 (Today)",
        annotation_position="top"
    )'''

new_code = '''    # Add vertical line for T0
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

content = content.replace(old_code, new_code)

with open('dashboard_v2.py', 'w') as f:
    f.write(content)

print("Fixed!")
