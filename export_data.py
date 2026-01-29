"""Export all simulated data to Excel for analysis"""

import pandas as pd
from data_simulator_realistic import generate_sample_data

print("Generating data...")
data = generate_sample_data(periods=730, random_seed=42)

# Create Excel writer
output_file = "cash_forecast_data.xlsx"
print(f"Exporting to {output_file}...")

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    
    # 1. FQM Flow (raw transactions)
    print("  - FQM Flow transactions...")
    data['fqm_flow'].to_excel(writer, sheet_name='FQM_Flow_Transactions', index=False)
    
    # 2. Daily Cash Position
    print("  - Daily cash position...")
    data['daily_cash_position'].to_excel(writer, sheet_name='Daily_Cash_Position', index=False)
    
    # 3. Category Breakdown
    print("  - Category breakdown...")
    data['category_breakdown'].to_excel(writer, sheet_name='Category_Breakdown', index=False)
    
    # 4. Company Breakdown
    print("  - Company breakdown...")
    data['company_breakdown'].to_excel(writer, sheet_name='Company_Breakdown', index=False)
    
    # 5. Summary stats
    print("  - Summary statistics...")
    daily = data['daily_cash_position']
    summary = pd.DataFrame({
        'Metric': [
            'Total Days',
            'Start Date',
            'End Date',
            'Starting Cash Position',
            'Ending Cash Position',
            'Avg Daily Inflow',
            'Avg Daily Outflow',
            'Avg Net Cash Flow',
            'Total Inflows',
            'Total Outflows',
            'Net Change',
            'Min Cash Position',
            'Max Cash Position',
            'Std Dev (Net Cash Flow)'
        ],
        'Value': [
            len(daily),
            daily['date'].min().strftime('%Y-%m-%d'),
            daily['date'].max().strftime('%Y-%m-%d'),
            f"${daily['cash_position'].iloc[0]:,.0f}",
            f"${daily['cash_position'].iloc[-1]:,.0f}",
            f"${daily['inflow'].mean():,.0f}",
            f"${daily['outflow'].mean():,.0f}",
            f"${daily['net_cash_flow'].mean():,.0f}",
            f"${daily['inflow'].sum():,.0f}",
            f"${daily['outflow'].sum():,.0f}",
            f"${daily['net_cash_flow'].sum():,.0f}",
            f"${daily['cash_position'].min():,.0f}",
            f"${daily['cash_position'].max():,.0f}",
            f"${daily['net_cash_flow'].std():,.0f}"
        ]
    })
    summary.to_excel(writer, sheet_name='Summary', index=False)

print(f"\n✅ Done! File saved: {output_file}")
print(f"\nSheets included:")
print("  1. FQM_Flow_Transactions - Raw transaction data")
print("  2. Daily_Cash_Position - Aggregated daily view")
print("  3. Category_Breakdown - By cash flow category")
print("  4. Company_Breakdown - By company code")
print("  5. Summary - Key statistics")
