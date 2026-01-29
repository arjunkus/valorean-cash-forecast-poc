# Fix the day_of_week issue by adding it after merge

with open('dashboard_v5.py', 'r') as f:
    content = f.read()

# Find the section after merge and add day_of_week
old_code = '''        if len(merged) == 0:
            continue
        
        # Handle column naming after merge'''

new_code = '''        if len(merged) == 0:
            continue
        
        # Add day_of_week from date (most reliable)
        merged['day_of_week'] = merged['date'].dt.dayofweek
        
        # Handle column naming after merge'''

content = content.replace(old_code, new_code)

with open('dashboard_v5.py', 'w') as f:
    f.write(content)

print("Fixed!")
