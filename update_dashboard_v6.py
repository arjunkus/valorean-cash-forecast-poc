# Update dashboard to use v6 models and v3 data simulator

with open('dashboard_v5.py', 'r') as f:
    content = f.read()

# Update imports
content = content.replace(
    'from models_prophet_v5 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar',
    'from models_prophet_v6 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar'
)

content = content.replace(
    'from data_simulator_v2 import generate_category_data',
    'from data_simulator_v3 import generate_category_data'
)

with open('dashboard_v5.py', 'w') as f:
    f.write(content)

print("Dashboard updated to use v6 models and v3 data simulator!")
