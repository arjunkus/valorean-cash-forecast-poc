# Add training_data attribute to ProphetCashForecaster

with open('models_prophet_v3.py', 'r') as f:
    content = f.read()

# Find and replace the fit method to add training_data
old_line = "        self.last_actual_date = df['date'].iloc[-1]"
new_line = """        self.training_data = df
        self.last_actual_date = df['date'].iloc[-1]"""

content = content.replace(old_line, new_line)

with open('models_prophet_v3.py', 'w') as f:
    f.write(content)

print("Fixed! Added training_data attribute.")
