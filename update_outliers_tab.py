import re

with open('dashboard_v5.py', 'r') as f:
    content = f.read()

# Find and replace the render_outliers function
old_func = '''def render_outliers():
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Load Data & Train'")
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
    st.plotly_chart(fig, use_container_width=True)'''

new_func = '''def render_outliers():
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
        st.metric("🔴 High Severity", summary.get('by_severity', {}).get('High', 0))
    with col4:
        st.metric("🟡 Medium Severity", summary.get('by_severity', {}).get('Medium', 0))
    
    st.markdown("---")
    
    # Actionable outliers table
    if outliers_df is not None and len(outliers_df) > 0:
        st.subheader("🎯 Actionable Items")
        
        # High severity first
        high_outliers = outliers_df[outliers_df['severity'] == 'High']
        if len(high_outliers) > 0:
            st.markdown("#### 🔴 High Priority")
            for _, row in high_outliers.iterrows():
                with st.expander(f"{row['date'].strftime('%Y-%m-%d')} - {row['anomaly_type']}", expanded=True):
                    st.markdown(f"**{row['description']}**")
                    st.info(f"**Recommended Action:** {row['recommended_action']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual", f"${row['value']:,.0f}")
                    with col2:
                        st.metric("Expected", f"${row['expected']:,.0f}")
                    with col3:
                        st.metric("Z-Score", f"{row['z_score']:.1f}σ")
        
        # Medium severity
        med_outliers = outliers_df[outliers_df['severity'] == 'Medium']
        if len(med_outliers) > 0:
            st.markdown("#### 🟡 Medium Priority")
            for _, row in med_outliers.iterrows():
                with st.expander(f"{row['date'].strftime('%Y-%m-%d')} - {row['anomaly_type']}"):
                    st.markdown(f"**{row['description']}**")
                    st.info(f"**Recommended Action:** {row['recommended_action']}")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Actual", f"${row['value']:,.0f}")
                    with col2:
                        st.metric("Expected", f"${row['expected']:,.0f}")
                    with col3:
                        st.metric("Z-Score", f"{row['z_score']:.1f}σ")
    else:
        st.success("✅ No actionable outliers detected. Cash flows are within normal ranges.")
    
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
        """)'''

content = content.replace(old_func, new_func)

# Also update the sidebar to store the detector
old_sidebar = '''            outlier_detector = OutlierDetector()
            st.session_state.outlier_results = outlier_detector.detect(st.session_state.daily_cash.copy())
            st.session_state.outlier_summary = outlier_detector.get_outlier_summary()'''

new_sidebar = '''            outlier_detector = OutlierDetector()
            outlier_detector.detect(st.session_state.daily_cash.copy(), st.session_state.category_df)
            st.session_state.outlier_detector = outlier_detector
            st.session_state.outlier_results = outlier_detector.results
            st.session_state.outlier_summary = outlier_detector.get_outlier_summary()'''

content = content.replace(old_sidebar, new_sidebar)

with open('dashboard_v5.py', 'w') as f:
    f.write(content)

print("Dashboard outliers tab updated!")
