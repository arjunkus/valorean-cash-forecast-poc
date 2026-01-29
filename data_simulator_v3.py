"""Cash Flow Data Simulator v3 - Biweekly Payroll"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

def get_us_holidays(year):
    holidays = [datetime(year, 1, 1), datetime(year, 7, 4), datetime(year, 12, 25), datetime(year, 11, 11)]
    nov1 = datetime(year, 11, 1)
    holidays.append(nov1 + timedelta(days=(3 - nov1.weekday()) % 7 + 21))
    return holidays

def is_banking_day(date, holidays):
    return date.weekday() < 5 and date.date() not in [h.date() for h in holidays]

def generate_category_data(start_date=None, periods=730, monthly_revenue=30_000_000, starting_cash_balance=50_000_000, seed=42):
    np.random.seed(seed)
    if start_date is None: start_date = datetime.now() - timedelta(days=periods)
    dates = pd.date_range(start=start_date, periods=periods, freq='D')
    holidays = [h for year in set(d.year for d in dates) for h in get_us_holidays(year)]
    
    daily_ar, weekly_ap, biweekly_payroll = monthly_revenue / 22, (monthly_revenue * 0.4) / 4.33, monthly_revenue * 0.30 / 2
    monthly_debt, monthly_inv_income, weekly_ic_in, weekly_ic_out = 500_000, 50_000, 200_000, 150_000
    
    first_date = dates[0].to_pydatetime()
    first_wed = first_date + timedelta(days=(2 - first_date.weekday()) % 7)
    payroll_dates = set()
    current = first_wed
    while current <= dates[-1].to_pydatetime():
        payroll_dates.add(current.date())
        current += timedelta(days=14)
    
    category_data = []
    for date in dates:
        date_dt, is_banking = date.to_pydatetime(), is_banking_day(date.to_pydatetime(), holidays)
        dow, dom, month = date.dayofweek, date.day, date.month
        row = {'date': date, 'is_banking_day': is_banking, 'day_of_week': dow, 'day_name': date.day_name(), 'day_of_month': dom, 'month': month, 'AR': 0, 'INV_INC': 0, 'IC_IN': 0, 'PAYROLL': 0, 'AP': 0, 'TAX': 0, 'CAPEX': 0, 'DEBT': 0, 'IC_OUT': 0, 'is_payroll_day': False}
        if is_banking:
            row['AR'] = daily_ar * {0: 1.2, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.9}.get(dow, 0) * (1.3 if dom >= 25 else 1.0) * (1.1 if month in [3,6,9,12] else 1.0) * np.random.uniform(0.85, 1.15)
            if dom == 1: row['INV_INC'] = monthly_inv_income * np.random.uniform(0.95, 1.05)
            if dow == 0: row['IC_IN'] = weekly_ic_in * np.random.uniform(0.9, 1.1)
            if date_dt.date() in payroll_dates: row['PAYROLL'], row['is_payroll_day'] = biweekly_payroll * np.random.uniform(0.98, 1.02), True
            if dow == 4: row['AP'] = weekly_ap * np.random.uniform(0.85, 1.15)
            if dom == 15 and month in [4,6,9,12]: row['TAX'] = monthly_revenue * 3 * 0.08 * np.random.uniform(0.95, 1.05)
            if dom == 20 and month in [1,5,9]: row['CAPEX'] = monthly_revenue * 4 * 0.05 * np.random.uniform(0.8, 1.2)
            if dom == 1: row['DEBT'] = monthly_debt
            if dow == 2: row['IC_OUT'] = weekly_ic_out * np.random.uniform(0.9, 1.1)
        category_data.append(row)
    
    df = pd.DataFrame(category_data)
    df['total_inflow'] = df['AR'] + df['INV_INC'] + df['IC_IN']
    df['total_outflow'] = df['PAYROLL'] + df['AP'] + df['TAX'] + df['CAPEX'] + df['DEBT'] + df['IC_OUT']
    df['net_cash_flow'] = df['total_inflow'] - df['total_outflow']
    balances = [starting_cash_balance]
    for net in df['net_cash_flow'].iloc[:-1]: balances.append(balances[-1] + net)
    df['opening_balance'] = balances
    df['closing_balance'] = df['opening_balance'] + df['net_cash_flow']
    
    daily_cash = df[['date', 'is_banking_day', 'day_of_week', 'day_name', 'day_of_month', 'opening_balance', 'total_inflow', 'total_outflow', 'net_cash_flow', 'closing_balance']].copy()
    daily_cash = daily_cash.rename(columns={'total_inflow': 'inflow', 'total_outflow': 'outflow'})
    return {'daily_cash_position': daily_cash, 'category_details': df}
