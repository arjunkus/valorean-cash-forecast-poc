"""
Realistic Cash Flow Data Simulator v2
=====================================
Generates cash flow data WITH CATEGORY BREAKDOWN.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

def get_us_holidays(year: int) -> List[datetime]:
    holidays = [
        datetime(year, 1, 1),
        datetime(year, 7, 4),
        datetime(year, 12, 25),
        datetime(year, 11, 11),
    ]
    nov1 = datetime(year, 11, 1)
    days_until_thu = (3 - nov1.weekday()) % 7
    thanksgiving = nov1 + timedelta(days=days_until_thu + 21)
    holidays.append(thanksgiving)
    return holidays

def is_banking_day(date: datetime, holidays: List[datetime]) -> bool:
    if date.weekday() >= 5:
        return False
    if date.date() in [h.date() for h in holidays]:
        return False
    return True

def generate_category_data(
    start_date: datetime = None,
    periods: int = 730,
    monthly_revenue: float = 30_000_000,
    starting_cash_balance: float = 50_000_000,
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    
    np.random.seed(seed)
    
    if start_date is None:
        start_date = datetime.now() - timedelta(days=periods)
    
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    
    years = set(d.year for d in dates)
    holidays = []
    for year in years:
        holidays.extend(get_us_holidays(year))
    
    category_data = []
    
    daily_ar = monthly_revenue / 22
    weekly_ap = (monthly_revenue * 0.4) / 4.33
    biweekly_payroll = monthly_revenue * 0.30 / 2
    monthly_debt = 500_000
    monthly_inv_income = 50_000
    weekly_ic_in = 200_000
    weekly_ic_out = 150_000
    
    for date in dates:
        date_dt = date.to_pydatetime()
        is_banking = is_banking_day(date_dt, holidays)
        day_of_week = date.dayofweek
        day_of_month = date.day
        month = date.month
        
        daily_cats = {
            'date': date,
            'is_banking_day': is_banking,
            'day_of_week': day_of_week,
            'day_name': date.day_name(),
            'day_of_month': day_of_month,
            'month': month,
            'AR': 0, 'INV_INC': 0, 'IC_IN': 0,
            'PAYROLL': 0, 'AP': 0, 'TAX': 0, 'CAPEX': 0, 'DEBT': 0, 'IC_OUT': 0,
        }
        
        if is_banking:
            dow_factor = {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.9}.get(day_of_week, 0)
            month_end_factor = 1.3 if day_of_month >= 25 else 1.0
            quarterly_factor = 1.1 if month in [3, 6, 9, 12] else 1.0
            noise = np.random.uniform(0.85, 1.15)
            daily_cats['AR'] = daily_ar * dow_factor * month_end_factor * quarterly_factor * noise
            
            if day_of_month == 1:
                daily_cats['INV_INC'] = monthly_inv_income * np.random.uniform(0.95, 1.05)
            
            if day_of_week == 0:
                daily_cats['IC_IN'] = weekly_ic_in * np.random.uniform(0.9, 1.1)
            
            is_month_end = (date + timedelta(days=1)).month != month
            if day_of_month == 15 or is_month_end:
                daily_cats['PAYROLL'] = biweekly_payroll * np.random.uniform(0.98, 1.02)
            
            if day_of_week == 4:
                daily_cats['AP'] = weekly_ap * np.random.uniform(0.85, 1.15)
            
            if day_of_month == 15 and month in [4, 6, 9, 12]:
                quarterly_revenue = monthly_revenue * 3
                daily_cats['TAX'] = quarterly_revenue * 0.08 * np.random.uniform(0.95, 1.05)
            
            if day_of_month == 20 and month in [1, 5, 9]:
                four_month_revenue = monthly_revenue * 4
                daily_cats['CAPEX'] = four_month_revenue * 0.05 * np.random.uniform(0.8, 1.2)
            
            if day_of_month == 1:
                daily_cats['DEBT'] = monthly_debt
            
            if day_of_week == 2:
                daily_cats['IC_OUT'] = weekly_ic_out * np.random.uniform(0.9, 1.1)
        
        category_data.append(daily_cats)
    
    category_df = pd.DataFrame(category_data)
    
    category_df['total_inflow'] = category_df['AR'] + category_df['INV_INC'] + category_df['IC_IN']
    category_df['total_outflow'] = (category_df['PAYROLL'] + category_df['AP'] + 
                                     category_df['TAX'] + category_df['CAPEX'] + 
                                     category_df['DEBT'] + category_df['IC_OUT'])
    category_df['net_cash_flow'] = category_df['total_inflow'] - category_df['total_outflow']
    
    opening_balances = [starting_cash_balance]
    closing_balances = []
    
    for i in range(len(category_df)):
        opening = opening_balances[-1]
        net = category_df.iloc[i]['net_cash_flow']
        closing = opening + net
        closing_balances.append(closing)
        if i < len(category_df) - 1:
            opening_balances.append(closing)
    
    category_df['opening_balance'] = opening_balances
    category_df['closing_balance'] = closing_balances
    
    daily_cash = category_df[[
        'date', 'is_banking_day', 'day_of_week', 'day_name', 'day_of_month',
        'opening_balance', 'total_inflow', 'total_outflow', 'net_cash_flow', 'closing_balance'
    ]].copy()
    daily_cash = daily_cash.rename(columns={'total_inflow': 'inflow', 'total_outflow': 'outflow'})
    
    return {
        'daily_cash_position': daily_cash,
        'category_details': category_df
    }

def generate_sample_data(periods: int = 730, seed: int = 42) -> Dict:
    return generate_category_data(periods=periods, seed=seed)

if __name__ == "__main__":
    data = generate_category_data(periods=30)
    print(data['category_details'][['date', 'AR', 'PAYROLL', 'AP']].head(10))
