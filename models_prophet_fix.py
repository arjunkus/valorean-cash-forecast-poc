# Quick fix for the column naming issue in analyze()
import pandas as pd

# Read the file
with open('models_prophet.py', 'r') as f:
    content = f.read()

# Fix the merge issue - closing_balance becomes closing_balance_actual after merge
content = content.replace(
    "merged['closing_balance'].values,",
    "merged['closing_balance_actual'].values,"
)

# Also need to fix the forecast column reference
content = content.replace(
    "merged['closing_balance_forecast'].values",
    "merged['closing_balance'].values"
)

# Write back
with open('models_prophet.py', 'w') as f:
    f.write(content)

print("Fixed!")
