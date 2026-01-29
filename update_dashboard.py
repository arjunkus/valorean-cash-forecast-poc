# Update dashboard to use models_prophet_v5

with open('dashboard_v5.py', 'r') as f:
    content = f.read()

# Update import
content = content.replace(
    'from models_prophet_v4 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar',
    'from models_prophet_v5 import ProphetCashForecaster, ForecastAnalyzer, USBankingCalendar'
)

with open('dashboard_v5.py', 'w') as f:
    f.write(content)

print("Dashboard updated to use models_prophet_v5!")
