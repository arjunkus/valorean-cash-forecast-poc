"""
SAP FQM Data Simulator
======================
Generates realistic cash flow data mimicking SAP FQM_FLOW and related tables.
Includes seasonality, trends, holidays, and realistic business patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    COMPANY_CODES, BANK_ACCOUNTS, CASH_FLOW_CATEGORIES,
    EXCHANGE_RATES, TIME_HORIZONS
)


class SAPFQMSimulator:
    """
    Simulates SAP FQM (Financial Quotation Management) cash flow data.
    
    Generates realistic daily cash flows with:
    - Multiple company codes and currencies
    - Multiple bank accounts
    - Inflow and outflow categories
    - Seasonality patterns (weekly, monthly, quarterly, yearly)
    - Trend components
    - Random noise and outliers
    - Holiday effects
    """
    
    def __init__(
        self,
        start_date: str = None,
        periods: int = 730,  # 2 years
        base_daily_inflow: float = 1_000_000,
        base_daily_outflow: float = 850_000,
        random_seed: int = 42
    ):
        """
        Initialize the simulator.
        
        Args:
            start_date: Start date for simulation (default: 2 years ago)
            periods: Number of days to simulate
            base_daily_inflow: Base daily inflow amount in USD
            base_daily_outflow: Base daily outflow amount in USD
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        if start_date is None:
            self.start_date = datetime.now() - timedelta(days=periods)
        else:
            self.start_date = pd.to_datetime(start_date)
        
        self.periods = periods
        self.base_daily_inflow = base_daily_inflow
        self.base_daily_outflow = base_daily_outflow
        
        # Generate date range
        self.dates = pd.date_range(start=self.start_date, periods=periods, freq='D')
        
        # US Federal Holidays (simplified)
        self.holidays = self._generate_holidays()
    
    def _generate_holidays(self) -> List[datetime]:
        """Generate list of US federal holidays for the simulation period."""
        holidays = []
        for year in range(self.start_date.year, self.start_date.year + 3):
            holidays.extend([
                datetime(year, 1, 1),   # New Year's Day
                datetime(year, 1, 15),  # MLK Day (approx)
                datetime(year, 2, 19),  # Presidents Day (approx)
                datetime(year, 5, 28),  # Memorial Day (approx)
                datetime(year, 7, 4),   # Independence Day
                datetime(year, 9, 4),   # Labor Day (approx)
                datetime(year, 10, 9),  # Columbus Day (approx)
                datetime(year, 11, 11), # Veterans Day
                datetime(year, 11, 23), # Thanksgiving (approx)
                datetime(year, 12, 25), # Christmas
            ])
        return holidays
    
    def _add_seasonality(self, dates: pd.DatetimeIndex) -> np.ndarray:
        """
        Add multiple seasonality components.
        
        Returns multiplier array for seasonal effects.
        """
        n = len(dates)
        
        # Weekly seasonality (lower on weekends)
        day_of_week = dates.dayofweek
        weekly = np.where(day_of_week < 5, 1.0, 0.3)  # Weekdays vs weekends
        
        # Monthly seasonality (higher at month end for payments)
        day_of_month = dates.day
        monthly = 1.0 + 0.3 * np.sin(2 * np.pi * day_of_month / 30)
        
        # End of month spike
        month_end = (dates + pd.Timedelta(days=1)).month != dates.month
        monthly = np.where(month_end, monthly * 1.5, monthly)
        
        # Quarterly seasonality (Q4 typically higher)
        quarter = dates.quarter
        quarterly = np.where(quarter == 4, 1.2, 
                    np.where(quarter == 1, 0.9, 1.0))
        
        # Yearly seasonality
        day_of_year = dates.dayofyear
        yearly = 1.0 + 0.15 * np.sin(2 * np.pi * (day_of_year - 90) / 365)
        
        return weekly * monthly * quarterly * yearly
    
    def _add_trend(self, n: int, annual_growth_rate: float = 0.05) -> np.ndarray:
        """
        Add trend component.
        
        Args:
            n: Number of periods
            annual_growth_rate: Annual growth rate (0.05 = 5%)
        """
        daily_growth = (1 + annual_growth_rate) ** (1/365) - 1
        return np.cumprod(1 + np.full(n, daily_growth))
    
    def _add_noise(self, n: int, volatility: str = "medium") -> np.ndarray:
        """
        Add random noise based on volatility level.
        
        Args:
            n: Number of periods
            volatility: 'low', 'medium', or 'high'
        """
        vol_map = {"low": 0.05, "medium": 0.15, "high": 0.30}
        std = vol_map.get(volatility, 0.15)
        return 1.0 + np.random.normal(0, std, n)
    
    def _add_outliers(self, data: np.ndarray, outlier_prob: float = 0.02) -> np.ndarray:
        """
        Add occasional outliers to simulate exceptional transactions.
        
        Args:
            data: Input array
            outlier_prob: Probability of outlier on any given day
        """
        outliers = np.random.random(len(data)) < outlier_prob
        multipliers = np.where(outliers, np.random.choice([0.3, 2.0, 3.0], len(data)), 1.0)
        return data * multipliers
    
    def _apply_holiday_effect(self, dates: pd.DatetimeIndex, data: np.ndarray) -> np.ndarray:
        """Reduce cash flow on holidays."""
        is_holiday = np.array([d.date() in [h.date() for h in self.holidays] for d in dates])
        return np.where(is_holiday, data * 0.2, data)
    
    def generate_fqm_flow(self) -> pd.DataFrame:
        """
        Generate the main FQM_FLOW table with all cash flow transactions.
        
        Returns:
            DataFrame mimicking SAP FQM_FLOW structure
        """
        all_records = []
        
        for company in COMPANY_CODES:
            company_code = company["code"]
            currency = company["currency"]
            exchange_rate = EXCHANGE_RATES.get(currency, 1.0)
            
            # Company-specific scaling factor
            company_scale = {
                "1000": 1.0,    # US - largest
                "2000": 0.6,   # EU
                "3000": 0.3,   # UK
                "4000": 0.4,   # APAC
                "5000": 0.2,   # LATAM
            }.get(company_code, 0.5)
            
            # Get bank accounts for this company
            company_accounts = [ba for ba in BANK_ACCOUNTS if ba["company_code"] == company_code]
            
            # Generate INFLOWS
            for category in CASH_FLOW_CATEGORIES["inflows"]:
                base_amount = self.base_daily_inflow * category["typical_pct"] * company_scale
                
                # Apply all components
                seasonality = self._add_seasonality(self.dates)
                trend = self._add_trend(self.periods, annual_growth_rate=0.05)
                noise = self._add_noise(self.periods, category["volatility"])
                
                amounts = base_amount * seasonality * trend * noise
                amounts = self._add_outliers(amounts)
                amounts = self._apply_holiday_effect(self.dates, amounts)
                
                # Convert to local currency
                amounts_local = amounts / exchange_rate
                
                for i, date in enumerate(self.dates):
                    if amounts[i] > 0:  # Skip zero amounts
                        bank_account = np.random.choice(company_accounts) if company_accounts else None
                        
                        all_records.append({
                            "transaction_id": f"TXN{len(all_records):08d}",
                            "posting_date": date,
                            "value_date": date + timedelta(days=np.random.randint(0, 3)),
                            "company_code": company_code,
                            "company_name": company["name"],
                            "bank_account_id": bank_account["account_id"] if bank_account else None,
                            "bank_name": bank_account["bank_name"] if bank_account else None,
                            "flow_type": "INFLOW",
                            "category_id": category["category_id"],
                            "category_name": category["name"],
                            "amount_local": round(amounts_local[i], 2),
                            "currency": currency,
                            "exchange_rate": exchange_rate,
                            "amount_usd": round(amounts[i], 2),
                            "region": company["region"],
                        })
            
            # Generate OUTFLOWS
            for category in CASH_FLOW_CATEGORIES["outflows"]:
                base_amount = self.base_daily_outflow * category["typical_pct"] * company_scale
                
                # Special patterns for certain categories
                if category["category_id"] == "PAYROLL":
                    # Payroll typically bi-weekly or monthly
                    amounts = np.zeros(self.periods)
                    for i, date in enumerate(self.dates):
                        if date.day in [1, 15]:  # Semi-monthly payroll
                            amounts[i] = base_amount * 15 * self._add_noise(1, "low")[0]
                
                elif category["category_id"] == "TAX":
                    # Quarterly tax payments
                    amounts = np.zeros(self.periods)
                    for i, date in enumerate(self.dates):
                        if date.month in [4, 6, 9, 12] and date.day == 15:
                            amounts[i] = base_amount * 90 * self._add_noise(1, "medium")[0]
                
                elif category["category_id"] == "DEBT":
                    # Monthly debt service
                    amounts = np.zeros(self.periods)
                    for i, date in enumerate(self.dates):
                        if date.day == 1:
                            amounts[i] = base_amount * 30 * self._add_noise(1, "low")[0]
                
                else:
                    # Regular daily patterns
                    seasonality = self._add_seasonality(self.dates)
                    trend = self._add_trend(self.periods, annual_growth_rate=0.03)
                    noise = self._add_noise(self.periods, category["volatility"])
                    
                    amounts = base_amount * seasonality * trend * noise
                    amounts = self._add_outliers(amounts, outlier_prob=0.01)
                    amounts = self._apply_holiday_effect(self.dates, amounts)
                
                # Convert to local currency
                amounts_local = amounts / exchange_rate
                
                for i, date in enumerate(self.dates):
                    if amounts[i] > 0:
                        bank_account = np.random.choice(company_accounts) if company_accounts else None
                        
                        all_records.append({
                            "transaction_id": f"TXN{len(all_records):08d}",
                            "posting_date": date,
                            "value_date": date + timedelta(days=np.random.randint(0, 3)),
                            "company_code": company_code,
                            "company_name": company["name"],
                            "bank_account_id": bank_account["account_id"] if bank_account else None,
                            "bank_name": bank_account["bank_name"] if bank_account else None,
                            "flow_type": "OUTFLOW",
                            "category_id": category["category_id"],
                            "category_name": category["name"],
                            "amount_local": round(amounts_local[i], 2),
                            "currency": currency,
                            "exchange_rate": exchange_rate,
                            "amount_usd": round(amounts[i], 2),
                            "region": company["region"],
                        })
        
        df = pd.DataFrame(all_records)
        df["posting_date"] = pd.to_datetime(df["posting_date"])
        df["value_date"] = pd.to_datetime(df["value_date"])
        
        return df.sort_values("posting_date").reset_index(drop=True)
    
    def generate_daily_cash_position(self, fqm_flow: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate daily aggregated cash position from FQM_FLOW data.
        
        Args:
            fqm_flow: FQM_FLOW DataFrame (will generate if not provided)
        
        Returns:
            DataFrame with daily cash positions
        """
        if fqm_flow is None:
            fqm_flow = self.generate_fqm_flow()
        
        # Aggregate by date
        daily = fqm_flow.groupby(["posting_date", "flow_type"]).agg({
            "amount_usd": "sum"
        }).reset_index()
        
        # Pivot to get inflows and outflows as columns
        daily_pivot = daily.pivot(
            index="posting_date",
            columns="flow_type",
            values="amount_usd"
        ).fillna(0).reset_index()
        
        daily_pivot.columns = ["date", "inflow", "outflow"]
        daily_pivot["net_cash_flow"] = daily_pivot["inflow"] - daily_pivot["outflow"]
        
        # Calculate cumulative cash position
        starting_balance = 10_000_000  # $10M starting balance
        daily_pivot["cash_position"] = starting_balance + daily_pivot["net_cash_flow"].cumsum()
        
        # Add time features for modeling
        daily_pivot["day_of_week"] = daily_pivot["date"].dt.dayofweek
        daily_pivot["day_of_month"] = daily_pivot["date"].dt.day
        daily_pivot["month"] = daily_pivot["date"].dt.month
        daily_pivot["quarter"] = daily_pivot["date"].dt.quarter
        daily_pivot["year"] = daily_pivot["date"].dt.year
        daily_pivot["is_weekend"] = daily_pivot["day_of_week"].isin([5, 6]).astype(int)
        daily_pivot["is_month_end"] = (daily_pivot["date"] + pd.Timedelta(days=1)).dt.month != daily_pivot["date"].dt.month
        daily_pivot["is_quarter_end"] = (daily_pivot["date"] + pd.Timedelta(days=1)).dt.quarter != daily_pivot["date"].dt.quarter
        
        return daily_pivot
    
    def generate_category_breakdown(self, fqm_flow: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate daily breakdown by category.
        
        Returns:
            DataFrame with daily amounts by category
        """
        if fqm_flow is None:
            fqm_flow = self.generate_fqm_flow()
        
        category_daily = fqm_flow.groupby(
            ["posting_date", "flow_type", "category_id", "category_name"]
        ).agg({
            "amount_usd": "sum"
        }).reset_index()
        
        category_daily.columns = ["date", "flow_type", "category_id", "category_name", "amount_usd"]
        
        return category_daily
    
    def generate_company_breakdown(self, fqm_flow: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generate daily breakdown by company code.
        
        Returns:
            DataFrame with daily amounts by company
        """
        if fqm_flow is None:
            fqm_flow = self.generate_fqm_flow()
        
        company_daily = fqm_flow.groupby(
            ["posting_date", "company_code", "company_name", "flow_type"]
        ).agg({
            "amount_usd": "sum"
        }).reset_index()
        
        return company_daily


def generate_sample_data(periods: int = 730, random_seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to generate all sample datasets.
    
    Args:
        periods: Number of days to simulate
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary containing all generated DataFrames
    """
    simulator = SAPFQMSimulator(periods=periods, random_seed=random_seed)
    
    fqm_flow = simulator.generate_fqm_flow()
    daily_cash = simulator.generate_daily_cash_position(fqm_flow)
    category_breakdown = simulator.generate_category_breakdown(fqm_flow)
    company_breakdown = simulator.generate_company_breakdown(fqm_flow)
    
    return {
        "fqm_flow": fqm_flow,
        "daily_cash_position": daily_cash,
        "category_breakdown": category_breakdown,
        "company_breakdown": company_breakdown,
    }


if __name__ == "__main__":
    # Test the simulator
    print("Generating sample SAP FQM data...")
    data = generate_sample_data()
    
    print(f"\nFQM_FLOW shape: {data['fqm_flow'].shape}")
    print(f"Daily Cash Position shape: {data['daily_cash_position'].shape}")
    print(f"Category Breakdown shape: {data['category_breakdown'].shape}")
    print(f"Company Breakdown shape: {data['company_breakdown'].shape}")
    
    print("\nSample FQM_FLOW records:")
    print(data['fqm_flow'].head())
    
    print("\nDaily Cash Position sample:")
    print(data['daily_cash_position'].head())
