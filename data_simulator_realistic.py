"""
Realistic SAP FQM Data Simulator
================================
Proper treasury logic for historical data:
  Closing Balance = Opening Balance + Receipts - Payments
  Next Day Opening = Previous Day Closing

Payment patterns based on actual business rules:

INFLOWS:
- Accounts Receivable: Daily on banking days (Mon-Fri, excl holidays)
- Investment Income: Monthly on 1st
- Intercompany In: Weekly on Mondays

OUTFLOWS:
- Payroll: Bi-weekly on 15th and last day of month
- Accounts Payable: Weekly on Fridays
- Tax Payments: Quarterly on 15th (Apr, Jun, Sep, Dec)
- Capital Expenditures: Every 4 months on 20th (Jan, May, Sep)
- Debt Service: Monthly on 1st
- Intercompany Out: Weekly on Wednesdays
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from config import (
    COMPANY_CODES, BANK_ACCOUNTS, EXCHANGE_RATES
)


class RealisticDataSimulator:
    """
    Generates realistic cash flow data with proper treasury logic.
    """
    
    def __init__(
        self,
        start_date: str = None,
        periods: int = 730,
        starting_cash_balance: float = 50_000_000,  # $50M starting balance
        base_monthly_revenue: float = 30_000_000,   # $30M monthly revenue
        random_seed: int = 42
    ):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=periods)
        else:
            self.start_date = pd.to_datetime(start_date)
        
        self.periods = periods
        self.starting_cash_balance = starting_cash_balance
        self.base_monthly_revenue = base_monthly_revenue
        
        # Generate date range
        self.dates = pd.date_range(start=self.start_date, periods=periods, freq='D')
        
        # US Federal Holidays
        self.holidays = self._generate_holidays()
    
    def _generate_holidays(self) -> List[datetime]:
        """Generate US federal holidays."""
        holidays = []
        for year in range(self.start_date.year, self.start_date.year + 3):
            holidays.extend([
                datetime(year, 1, 1),   # New Year
                datetime(year, 7, 4),   # Independence Day
                datetime(year, 12, 25), # Christmas
                datetime(year, 11, 24) if year == 2022 else datetime(year, 11, 23),  # Thanksgiving approx
            ])
        return holidays
    
    def _is_banking_day(self, date: pd.Timestamp) -> bool:
        """Check if date is a banking day (Mon-Fri, not holiday)."""
        if date.dayofweek >= 5:  # Weekend
            return False
        if date.date() in [h.date() for h in self.holidays]:
            return False
        return True
    
    def _is_last_day_of_month(self, date: pd.Timestamp) -> bool:
        """Check if date is last day of month."""
        next_day = date + timedelta(days=1)
        return next_day.month != date.month
    
    def _add_variance(self, base: float, variance_pct: float = 0.1) -> float:
        """Add random variance to a base amount."""
        return base * (1 + np.random.uniform(-variance_pct, variance_pct))
    
    def generate_inflows(self) -> pd.DataFrame:
        """Generate realistic inflow transactions."""
        records = []
        
        for date in self.dates:
            # -----------------------------------------------------------------
            # ACCOUNTS RECEIVABLE - Daily on banking days
            # -----------------------------------------------------------------
            if self._is_banking_day(date):
                base_daily_ar = self.base_monthly_revenue / 22
                
                # Day-of-week pattern (more collections Mon/Tue)
                dow_factor = {0: 1.2, 1: 1.15, 2: 1.0, 3: 0.95, 4: 0.9}.get(date.dayofweek, 1.0)
                
                # Month-end spike
                month_end_factor = 1.3 if date.day >= 25 else 1.0
                
                # Quarterly seasonality
                quarter_factor = {1: 0.9, 2: 1.0, 3: 0.95, 4: 1.15}.get(date.quarter, 1.0)
                
                ar_amount = self._add_variance(
                    base_daily_ar * dow_factor * month_end_factor * quarter_factor,
                    variance_pct=0.15
                )
                
                records.append({
                    'date': date,
                    'category': 'AR',
                    'category_name': 'Accounts Receivable',
                    'amount': ar_amount,
                    'flow_type': 'INFLOW',
                    'schedule': 'Daily (Banking Days)'
                })
            
            # -----------------------------------------------------------------
            # INVESTMENT INCOME - Monthly on 1st
            # -----------------------------------------------------------------
            if date.day == 1:
                inv_income = self._add_variance(50_000, variance_pct=0.05)
                records.append({
                    'date': date,
                    'category': 'INV_INC',
                    'category_name': 'Investment Income',
                    'amount': inv_income,
                    'flow_type': 'INFLOW',
                    'schedule': 'Monthly (1st)'
                })
            
            # -----------------------------------------------------------------
            # INTERCOMPANY IN - Weekly on Mondays
            # -----------------------------------------------------------------
            if date.dayofweek == 0:  # Monday
                ic_in = self._add_variance(200_000, variance_pct=0.2)
                records.append({
                    'date': date,
                    'category': 'IC_IN',
                    'category_name': 'Intercompany Transfers In',
                    'amount': ic_in,
                    'flow_type': 'INFLOW',
                    'schedule': 'Weekly (Monday)'
                })
        
        return pd.DataFrame(records)
    
    def generate_outflows(self) -> pd.DataFrame:
        """Generate realistic outflow transactions."""
        records = []
        
        # Calculate base amounts from revenue
        monthly_payroll = self.base_monthly_revenue * 0.30
        monthly_ap = self.base_monthly_revenue * 0.40
        quarterly_tax = self.base_monthly_revenue * 0.08 * 3
        capex_4months = self.base_monthly_revenue * 0.05 * 4
        monthly_debt = 500_000
        
        for date in self.dates:
            # -----------------------------------------------------------------
            # PAYROLL - Bi-weekly on 15th and last day of month
            # -----------------------------------------------------------------
            if date.day == 15 or self._is_last_day_of_month(date):
                payroll = self._add_variance(monthly_payroll / 2, variance_pct=0.02)
                records.append({
                    'date': date,
                    'category': 'PAYROLL',
                    'category_name': 'Payroll',
                    'amount': payroll,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Bi-weekly (15th & Month End)'
                })
            
            # -----------------------------------------------------------------
            # ACCOUNTS PAYABLE - Weekly on Fridays
            # -----------------------------------------------------------------
            if date.dayofweek == 4:  # Friday
                weekly_ap = self._add_variance(monthly_ap / 4.33, variance_pct=0.15)
                records.append({
                    'date': date,
                    'category': 'AP',
                    'category_name': 'Accounts Payable',
                    'amount': weekly_ap,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Weekly (Friday)'
                })
            
            # -----------------------------------------------------------------
            # TAX PAYMENTS - Quarterly on 15th (Apr, Jun, Sep, Dec)
            # -----------------------------------------------------------------
            if date.day == 15 and date.month in [4, 6, 9, 12]:
                tax = self._add_variance(quarterly_tax, variance_pct=0.1)
                records.append({
                    'date': date,
                    'category': 'TAX',
                    'category_name': 'Tax Payments',
                    'amount': tax,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Quarterly (15th: Apr, Jun, Sep, Dec)'
                })
            
            # -----------------------------------------------------------------
            # CAPITAL EXPENDITURES - Every 4 months on 20th (Jan, May, Sep)
            # -----------------------------------------------------------------
            if date.day == 20 and date.month in [1, 5, 9]:
                capex = self._add_variance(capex_4months, variance_pct=0.25)
                records.append({
                    'date': date,
                    'category': 'CAPEX',
                    'category_name': 'Capital Expenditures',
                    'amount': capex,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Every 4 Months (20th: Jan, May, Sep)'
                })
            
            # -----------------------------------------------------------------
            # DEBT SERVICE - Monthly on 1st
            # -----------------------------------------------------------------
            if date.day == 1:
                debt = self._add_variance(monthly_debt, variance_pct=0.0)
                records.append({
                    'date': date,
                    'category': 'DEBT',
                    'category_name': 'Debt Service',
                    'amount': debt,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Monthly (1st)'
                })
            
            # -----------------------------------------------------------------
            # INTERCOMPANY OUT - Weekly on Wednesdays
            # -----------------------------------------------------------------
            if date.dayofweek == 2:  # Wednesday
                ic_out = self._add_variance(150_000, variance_pct=0.2)
                records.append({
                    'date': date,
                    'category': 'IC_OUT',
                    'category_name': 'Intercompany Transfers Out',
                    'amount': ic_out,
                    'flow_type': 'OUTFLOW',
                    'schedule': 'Weekly (Wednesday)'
                })
        
        return pd.DataFrame(records)
    
    def generate_fqm_flow(self) -> pd.DataFrame:
        """Generate complete FQM_FLOW table."""
        inflows = self.generate_inflows()
        outflows = self.generate_outflows()
        
        all_flows = pd.concat([inflows, outflows], ignore_index=True)
        
        all_flows['transaction_id'] = [f'TXN{i:08d}' for i in range(len(all_flows))]
        all_flows['posting_date'] = all_flows['date']
        all_flows['value_date'] = all_flows['date']
        
        all_flows['company_code'] = np.random.choice(
            [c['code'] for c in COMPANY_CODES],
            size=len(all_flows),
            p=[0.5, 0.2, 0.15, 0.1, 0.05]
        )
        
        company_map = {c['code']: c for c in COMPANY_CODES}
        all_flows['company_name'] = all_flows['company_code'].map(lambda x: company_map[x]['name'])
        all_flows['currency'] = all_flows['company_code'].map(lambda x: company_map[x]['currency'])
        all_flows['region'] = all_flows['company_code'].map(lambda x: company_map[x]['region'])
        
        all_flows['exchange_rate'] = all_flows['currency'].map(EXCHANGE_RATES)
        all_flows['amount_local'] = all_flows['amount'] / all_flows['exchange_rate']
        all_flows['amount_usd'] = all_flows['amount']
        
        all_flows = all_flows.sort_values('date').reset_index(drop=True)
        
        return all_flows
    
    def generate_daily_cash_position(self, fqm_flow: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate daily cash position with proper treasury logic:
        
        For each day:
          Opening Balance = Previous day's Closing Balance
          + Actual Receipts (Inflows)
          - Actual Payments (Outflows)
          = Closing Balance
        
        The last Closing Balance becomes T0 for forecasting.
        """
        if fqm_flow is None:
            fqm_flow = self.generate_fqm_flow()
        
        # Aggregate inflows and outflows by date
        daily_inflows = fqm_flow[fqm_flow['flow_type'] == 'INFLOW'].groupby('date')['amount_usd'].sum()
        daily_outflows = fqm_flow[fqm_flow['flow_type'] == 'OUTFLOW'].groupby('date')['amount_usd'].sum()
        
        # Build daily DataFrame
        daily = pd.DataFrame({'date': self.dates})
        daily['inflow'] = daily['date'].map(daily_inflows).fillna(0)
        daily['outflow'] = daily['date'].map(daily_outflows).fillna(0)
        daily['net_cash_flow'] = daily['inflow'] - daily['outflow']
        
        # Calculate opening and closing balances iteratively
        opening_balances = []
        closing_balances = []
        
        current_balance = self.starting_cash_balance
        
        for i in range(len(daily)):
            # Opening balance for this day
            opening_balances.append(current_balance)
            
            # Closing = Opening + Receipts - Payments
            closing = current_balance + daily.iloc[i]['inflow'] - daily.iloc[i]['outflow']
            closing_balances.append(closing)
            
            # Next day's opening = Today's closing
            current_balance = closing
        
        daily['opening_balance'] = opening_balances
        daily['closing_balance'] = closing_balances
        
        # For backward compatibility, also keep cash_position (= closing_balance)
        daily['cash_position'] = daily['closing_balance']
        
        # Add time features
        daily['day_of_week'] = daily['date'].dt.dayofweek
        daily['day_name'] = daily['date'].dt.day_name()
        daily['day_of_month'] = daily['date'].dt.day
        daily['month'] = daily['date'].dt.month
        daily['quarter'] = daily['date'].dt.quarter
        daily['year'] = daily['date'].dt.year
        daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
        daily['is_month_end'] = (daily['date'] + pd.Timedelta(days=1)).dt.month != daily['date'].dt.month
        daily['is_quarter_end'] = (daily['date'] + pd.Timedelta(days=1)).dt.quarter != daily['date'].dt.quarter
        daily['is_banking_day'] = (~daily['is_weekend'].astype(bool)).astype(int)
        
        # Reorder columns for clarity
        column_order = [
            'date', 'day_name', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
            'is_weekend', 'is_banking_day', 'is_month_end', 'is_quarter_end',
            'opening_balance', 'inflow', 'outflow', 'net_cash_flow', 'closing_balance',
            'cash_position'
        ]
        daily = daily[column_order]
        
        return daily
    
    def generate_category_breakdown(self, fqm_flow: pd.DataFrame = None) -> pd.DataFrame:
        """Generate daily breakdown by category."""
        if fqm_flow is None:
            fqm_flow = self.generate_fqm_flow()
        
        return fqm_flow.groupby(
            ['date', 'flow_type', 'category', 'category_name', 'schedule']
        ).agg({'amount_usd': 'sum'}).reset_index()
    
    def generate_payment_schedule(self) -> pd.DataFrame:
        """Generate summary of payment schedules for reference."""
        schedules = [
            {'Category': 'Accounts Receivable', 'Type': 'INFLOW', 'Schedule': 'Daily (Banking Days)', 
             'Description': 'Customer payments received Mon-Fri, excluding holidays'},
            {'Category': 'Investment Income', 'Type': 'INFLOW', 'Schedule': 'Monthly (1st)',
             'Description': 'Interest and dividend income'},
            {'Category': 'Intercompany In', 'Type': 'INFLOW', 'Schedule': 'Weekly (Monday)',
             'Description': 'Transfers from subsidiaries'},
            {'Category': 'Payroll', 'Type': 'OUTFLOW', 'Schedule': 'Bi-weekly (15th & Month End)',
             'Description': 'Employee salaries and wages'},
            {'Category': 'Accounts Payable', 'Type': 'OUTFLOW', 'Schedule': 'Weekly (Friday)',
             'Description': 'Vendor payments'},
            {'Category': 'Tax Payments', 'Type': 'OUTFLOW', 'Schedule': 'Quarterly (15th: Apr, Jun, Sep, Dec)',
             'Description': 'Federal and state tax payments'},
            {'Category': 'Capital Expenditures', 'Type': 'OUTFLOW', 'Schedule': 'Every 4 Months (20th: Jan, May, Sep)',
             'Description': 'Equipment and infrastructure investments'},
            {'Category': 'Debt Service', 'Type': 'OUTFLOW', 'Schedule': 'Monthly (1st)',
             'Description': 'Loan principal and interest'},
            {'Category': 'Intercompany Out', 'Type': 'OUTFLOW', 'Schedule': 'Weekly (Wednesday)',
             'Description': 'Transfers to subsidiaries'},
        ]
        return pd.DataFrame(schedules)


def generate_sample_data(periods: int = 730, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Generate all sample datasets with realistic payment patterns.
    """
    simulator = RealisticDataSimulator(periods=periods, random_seed=random_seed)
    
    fqm_flow = simulator.generate_fqm_flow()
    daily_cash = simulator.generate_daily_cash_position(fqm_flow)
    category_breakdown = simulator.generate_category_breakdown(fqm_flow)
    payment_schedule = simulator.generate_payment_schedule()
    
    return {
        'fqm_flow': fqm_flow,
        'daily_cash_position': daily_cash,
        'category_breakdown': category_breakdown,
        'company_breakdown': fqm_flow.groupby(['date', 'company_code', 'company_name', 'flow_type']).agg({'amount_usd': 'sum'}).reset_index(),
        'payment_schedule': payment_schedule,
    }


if __name__ == "__main__":
    print("Generating realistic cash flow data...")
    print("=" * 70)
    
    data = generate_sample_data(periods=365)
    daily = data['daily_cash_position']
    
    print(f"\nTotal Days: {len(daily)}")
    print(f"Date Range: {daily['date'].min().strftime('%Y-%m-%d')} to {daily['date'].max().strftime('%Y-%m-%d')}")
    
    print("\n" + "=" * 70)
    print("PAYMENT SCHEDULES")
    print("=" * 70)
    print(data['payment_schedule'].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("FIRST 14 DAYS - DAILY CASH POSITION")
    print("=" * 70)
    print("Logic: Closing Balance = Opening Balance + Inflows - Outflows")
    print("       Next Day Opening = Previous Day Closing")
    print("-" * 70)
    
    sample = daily.head(14)[['date', 'day_name', 'opening_balance', 'inflow', 'outflow', 'net_cash_flow', 'closing_balance']]
    
    # Format for display
    sample_display = sample.copy()
    sample_display['date'] = sample_display['date'].dt.strftime('%Y-%m-%d')
    for col in ['opening_balance', 'inflow', 'outflow', 'net_cash_flow', 'closing_balance']:
        sample_display[col] = sample_display[col].apply(lambda x: f"${x:,.0f}")
    
    print(sample_display.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("VALIDATION: Opening/Closing Balance Continuity")
    print("=" * 70)
    
    # Verify the math
    print(f"Starting Balance (T0 Opening): ${daily.iloc[0]['opening_balance']:,.0f}")
    print(f"Final Closing Balance (T{len(daily)-1}): ${daily.iloc[-1]['closing_balance']:,.0f}")
    print(f"\nThis final closing balance becomes T0 for forecasting.")
    
    # Verify continuity
    errors = 0
    for i in range(1, len(daily)):
        if abs(daily.iloc[i]['opening_balance'] - daily.iloc[i-1]['closing_balance']) > 0.01:
            errors += 1
    
    if errors == 0:
        print("✅ Continuity Check PASSED: Each day's opening = previous day's closing")
    else:
        print(f"❌ Continuity Check FAILED: {errors} mismatches found")
    
    # Verify closing balance formula
    sample_row = daily.iloc[5]
    calculated = sample_row['opening_balance'] + sample_row['inflow'] - sample_row['outflow']
    if abs(calculated - sample_row['closing_balance']) < 0.01:
        print("✅ Formula Check PASSED: Closing = Opening + Inflows - Outflows")
    else:
        print("❌ Formula Check FAILED")
