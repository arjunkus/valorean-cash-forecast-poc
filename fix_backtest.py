# Fix the run_detailed_backtest function to include day_of_week

import re

with open('dashboard_v5.py', 'r') as f:
    content = f.read()

# Find and replace the problematic section
old_code = '''    # Filter test to banking days
    test_df['is_banking_day'] = test_df['date'].apply(
        lambda x: USBankingCalendar.is_banking_day(x, holidays)
    )
    test_banking = test_df[test_df['is_banking_day']].copy()'''

new_code = '''    # Filter test to banking days
    test_df['is_banking_day'] = test_df['date'].apply(
        lambda x: USBankingCalendar.is_banking_day(x, holidays)
    )
    test_df['day_of_week'] = test_df['date'].dt.dayofweek
    test_banking = test_df[test_df['is_banking_day']].copy()'''

content = content.replace(old_code, new_code)

# Also fix the merge to include day_of_week
old_merge = '''        merged = forecast_df.merge(
            test_banking[['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week'] + 
                        (['outflow_ex_capex'] if 'outflow_ex_capex' in test_banking.columns else [])],
            on='date', how='inner'
        )'''

new_merge = '''        # Ensure day_of_week is in test_banking
        if 'day_of_week' not in test_banking.columns:
            test_banking['day_of_week'] = test_banking['date'].dt.dayofweek
        
        merge_cols = ['date', 'inflow', 'outflow', 'closing_balance', 'day_of_week']
        if 'outflow_ex_capex' in test_banking.columns:
            merge_cols.append('outflow_ex_capex')
        
        merged = forecast_df.merge(
            test_banking[merge_cols],
            on='date', how='inner'
        )'''

content = content.replace(old_merge, new_merge)

with open('dashboard_v5.py', 'w') as f:
    f.write(content)

print("Fixed!")
