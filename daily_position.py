"""T+1 Daily Cash Position Module"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class IntradayTransaction:
    time: datetime
    category: str
    description: str
    amount: float
    source: str
    status: str

class DailyPositionManager:
    INFLOW_CATEGORIES = [
        ('AR_WIRE', 'AR - Wire Receipts'),
        ('AR_ACH', 'AR - ACH Credits'),
        ('AR_LOCKBOX', 'AR - Lockbox'),
        ('INV_INC', 'Investment Income'),
        ('IC_IN', 'Intercompany In'),
        ('OTHER_IN', 'Other Receipts'),
    ]
    
    OUTFLOW_CATEGORIES = [
        ('PAYROLL', 'Payroll'),
        ('AP_WIRE', 'AP - Wires'),
        ('AP_ACH', 'AP - ACH Payments'),
        ('AP_CHECK', 'AP - Check Clearings'),
        ('TAX', 'Tax Payments'),
        ('DEBT', 'Debt Service'),
        ('IC_OUT', 'Intercompany Out'),
        ('OTHER_OUT', 'Other Payments'),
    ]
    
    def __init__(self):
        self.position_date = None
        self.opening_balance_actual = None
        self.opening_balance_forecast = None
        self.forecast_data = {}
        self.intraday_transactions = []
        self.sap_scheduled_payments = []
        self.position_df = None
    
    def initialize_position(self, position_date, opening_balance_actual, forecast_df=None):
        self.position_date = position_date
        self.opening_balance_actual = opening_balance_actual
        self.intraday_transactions = []
        self.sap_scheduled_payments = []
        
        if forecast_df is not None and len(forecast_df) > 0:
            day_forecast = forecast_df[forecast_df['date'].dt.date == position_date.date()]
            if len(day_forecast) > 0:
                row = day_forecast.iloc[0]
                self.opening_balance_forecast = row.get('opening_balance', opening_balance_actual)
                self.forecast_data = {k: row.get(k, 0) for k in ['AR', 'INV_INC', 'IC_IN', 'PAYROLL', 'AP', 'TAX', 'DEBT', 'IC_OUT', 'CAPEX', 'forecast_inflow', 'forecast_outflow', 'closing_balance']}
            else:
                self.opening_balance_forecast = opening_balance_actual
                self.forecast_data = {}
        else:
            self.opening_balance_forecast = opening_balance_actual
            self.forecast_data = {}
        return self
    
    def add_intraday_transaction(self, time, category, description, amount, source='BANK', status='POSTED'):
        self.intraday_transactions.append(IntradayTransaction(time=time, category=category, description=description, amount=amount, source=source, status=status))
    
    def add_sap_scheduled_payment(self, category, description, amount, status='SCHEDULED'):
        self.sap_scheduled_payments.append({'category': category, 'description': description, 'amount': amount, 'status': status, 'source': 'SAP'})
    
    def load_intraday_from_bank(self, bank_transactions):
        for txn in bank_transactions:
            if txn.get('type') == 'CREDIT':
                if 'WIRE' in txn.get('description', '').upper(): category = 'AR_WIRE'
                elif 'ACH' in txn.get('description', '').upper(): category = 'AR_ACH'
                elif 'LOCKBOX' in txn.get('description', '').upper(): category = 'AR_LOCKBOX'
                else: category = 'OTHER_IN'
            else:
                if 'WIRE' in txn.get('description', '').upper(): category = 'AP_WIRE'
                elif 'ACH' in txn.get('description', '').upper(): category = 'AP_ACH'
                elif 'CHECK' in txn.get('description', '').upper(): category = 'AP_CHECK'
                else: category = 'OTHER_OUT'
            self.add_intraday_transaction(time=txn.get('time', datetime.now()), category=category, description=txn.get('description', ''), amount=txn.get('amount', 0), source='BANK', status='POSTED')
    
    def load_sap_payment_queue(self, sap_payments):
        for pmt in sap_payments:
            method = pmt.get('payment_method', 'OTHER').upper()
            if method == 'WIRE': category = 'AP_WIRE'
            elif method == 'ACH': category = 'AP_ACH'
            elif method == 'CHECK': category = 'AP_CHECK'
            else: category = 'OTHER_OUT'
            self.add_sap_scheduled_payment(category=category, description=f"{pmt.get('vendor', 'Unknown')} - {method}", amount=pmt.get('amount', 0), status=pmt.get('status', 'SCHEDULED'))
    
    def build_position(self):
        rows = []
        rows.append({'category': 'OPENING_BALANCE', 'display_name': 'Opening Balance', 'forecast': self.opening_balance_forecast, 'estimated_actual': self.opening_balance_actual, 'variance': self.opening_balance_actual - self.opening_balance_forecast, 'status': 'Confirmed from bank', 'section': 'BALANCE', 'sort_order': 0})
        
        intraday_by_cat = {}
        for txn in self.intraday_transactions:
            if txn.category not in intraday_by_cat: intraday_by_cat[txn.category] = {'amount': 0, 'count': 0}
            intraday_by_cat[txn.category]['amount'] += txn.amount
            intraday_by_cat[txn.category]['count'] += 1
        
        sap_by_cat = {}
        for pmt in self.sap_scheduled_payments:
            cat = pmt['category']
            if cat not in sap_by_cat: sap_by_cat[cat] = {'amount': 0, 'count': 0, 'status': pmt['status']}
            sap_by_cat[cat]['amount'] += pmt['amount']
            sap_by_cat[cat]['count'] += 1
        
        total_inflow_forecast, total_inflow_actual, sort_order = 0, 0, 10
        for cat_code, cat_name in self.INFLOW_CATEGORIES:
            if cat_code.startswith('AR'): forecast = self.forecast_data.get('AR', 0) / 3
            elif cat_code == 'INV_INC': forecast = self.forecast_data.get('INV_INC', 0)
            elif cat_code == 'IC_IN': forecast = self.forecast_data.get('IC_IN', 0)
            else: forecast = 0
            actual = intraday_by_cat.get(cat_code, {}).get('amount', 0)
            if actual == 0 and forecast > 0: actual, status = forecast * 0.8, 'Estimated'
            elif actual > 0: status = f"Posted ({intraday_by_cat[cat_code]['count']} txns)"
            else: status = '-'
            total_inflow_forecast += forecast
            total_inflow_actual += actual
            rows.append({'category': cat_code, 'display_name': cat_name, 'forecast': forecast, 'estimated_actual': actual, 'variance': actual - forecast, 'status': status, 'section': 'INFLOW', 'sort_order': sort_order})
            sort_order += 1
        
        rows.append({'category': 'TOTAL_INFLOW', 'display_name': 'TOTAL RECEIPTS', 'forecast': total_inflow_forecast, 'estimated_actual': total_inflow_actual, 'variance': total_inflow_actual - total_inflow_forecast, 'status': '', 'section': 'SUBTOTAL', 'sort_order': 20})
        
        total_outflow_forecast, total_outflow_actual, sort_order = 0, 0, 30
        for cat_code, cat_name in self.OUTFLOW_CATEGORIES:
            if cat_code == 'PAYROLL': forecast = self.forecast_data.get('PAYROLL', 0)
            elif cat_code.startswith('AP'): forecast = self.forecast_data.get('AP', 0) / 3
            elif cat_code == 'TAX': forecast = self.forecast_data.get('TAX', 0)
            elif cat_code == 'DEBT': forecast = self.forecast_data.get('DEBT', 0)
            elif cat_code == 'IC_OUT': forecast = self.forecast_data.get('IC_OUT', 0)
            else: forecast = 0
            sap_amount = sap_by_cat.get(cat_code, {}).get('amount', 0)
            intraday_amount = intraday_by_cat.get(cat_code, {}).get('amount', 0)
            actual = sap_amount + intraday_amount
            status_parts = []
            if sap_amount > 0: status_parts.append(f"SAP: ${sap_amount:,.0f}")
            if intraday_amount > 0: status_parts.append(f"Cleared: ${intraday_amount:,.0f}")
            if not status_parts and forecast > 0: actual, status_parts = forecast, ['Forecast']
            status = ' | '.join(status_parts) if status_parts else '-'
            total_outflow_forecast += forecast
            total_outflow_actual += actual
            rows.append({'category': cat_code, 'display_name': cat_name, 'forecast': forecast, 'estimated_actual': actual, 'variance': actual - forecast, 'status': status, 'section': 'OUTFLOW', 'sort_order': sort_order})
            sort_order += 1
        
        rows.append({'category': 'TOTAL_OUTFLOW', 'display_name': 'TOTAL PAYMENTS', 'forecast': total_outflow_forecast, 'estimated_actual': total_outflow_actual, 'variance': total_outflow_actual - total_outflow_forecast, 'status': '', 'section': 'SUBTOTAL', 'sort_order': 40})
        
        net_forecast, net_actual = total_inflow_forecast - total_outflow_forecast, total_inflow_actual - total_outflow_actual
        rows.append({'category': 'NET_CASH_FLOW', 'display_name': 'NET CASH FLOW', 'forecast': net_forecast, 'estimated_actual': net_actual, 'variance': net_actual - net_forecast, 'status': '', 'section': 'SUBTOTAL', 'sort_order': 50})
        
        closing_forecast, closing_actual = self.opening_balance_forecast + net_forecast, self.opening_balance_actual + net_actual
        rows.append({'category': 'CLOSING_BALANCE', 'display_name': 'CLOSING BALANCE', 'forecast': closing_forecast, 'estimated_actual': closing_actual, 'variance': closing_actual - closing_forecast, 'status': 'Projected', 'section': 'BALANCE', 'sort_order': 60})
        
        self.position_df = pd.DataFrame(rows).sort_values('sort_order')
        self.position_df['variance_pct'] = self.position_df.apply(lambda row: (row['variance'] / row['forecast'] * 100) if row['forecast'] != 0 else 0, axis=1)
        return self.position_df
    
    def get_investment_borrowing_recommendation(self):
        if self.position_df is None: self.build_position()
        closing = self.position_df[self.position_df['category'] == 'CLOSING_BALANCE'].iloc[0]
        closing_actual, variance = closing['estimated_actual'], closing['variance']
        target_balance, min_balance, invest_threshold = 10_000_000, 5_000_000, 20_000_000
        rec = {'closing_forecast': closing['forecast'], 'closing_estimated': closing_actual, 'variance': variance, 'target_balance': target_balance, 'confidence': 'HIGH' if abs(variance) < 1_000_000 else 'MEDIUM' if abs(variance) < 5_000_000 else 'LOW'}
        if closing_actual > invest_threshold: rec.update({'action': 'INVEST', 'amount': closing_actual - target_balance, 'reasoning': f"Excess cash ${closing_actual:,.0f}. Consider overnight investment."})
        elif closing_actual < min_balance: rec.update({'action': 'BORROW', 'amount': target_balance - closing_actual, 'reasoning': f"Shortfall projected. Consider borrowing."})
        else: rec.update({'action': 'HOLD', 'amount': 0, 'reasoning': f"Position ${closing_actual:,.0f} within target range."})
        return rec
    
    def get_position_summary(self):
        if self.position_df is None: self.build_position()
        df = self.position_df
        return {'position_date': self.position_date, 'opening_balance': self.opening_balance_actual, 'closing_forecast': df[df['category'] == 'CLOSING_BALANCE'].iloc[0]['forecast'], 'closing_estimated': df[df['category'] == 'CLOSING_BALANCE'].iloc[0]['estimated_actual'], 'variance': df[df['category'] == 'CLOSING_BALANCE'].iloc[0]['variance'], 'posted_transactions': len([t for t in self.intraday_transactions if t.status == 'POSTED']), 'scheduled_payments': len(self.sap_scheduled_payments), 'last_updated': datetime.now()}

    def get_accuracy_metrics(self):
        """Calculate RMSE-based accuracy metrics for T+1 position."""
        if self.position_df is None:
            self.build_position()
        
        closing = self.position_df[self.position_df['category'] == 'CLOSING_BALANCE'].iloc[0]
        forecast = closing['forecast']
        actual = closing['estimated_actual']
        
        # Absolute Error (RMSE for single day = absolute error)
        absolute_error = abs(actual - forecast)
        
        # Directional Bias: positive = overforecast, negative = underforecast
        bias = forecast - actual
        
        return {
            'date': self.position_date,
            'forecast': forecast,
            'actual': actual,
            'absolute_error': absolute_error,
            'bias': bias,
            'bias_direction': 'OVER' if bias > 0 else 'UNDER' if bias < 0 else 'ACCURATE'
        }
    def archive_position(self, storage_path='t1_position_history.parquet'):
        """Archive the current day's position for historical RMSE calculation."""
        if self.position_df is None:
            self.build_position()
        
        metrics = self.get_accuracy_metrics()
        summary = self.get_position_summary()
        
        archive_record = {
            'position_date': self.position_date,
            'opening_balance': self.opening_balance_actual,
            'closing_forecast': metrics['forecast'],
            'closing_actual': metrics['actual'],
            'absolute_error': metrics['absolute_error'],
            'bias': metrics['bias'],
            'bias_direction': metrics['bias_direction'],
            'posted_transactions': summary['posted_transactions'],
            'scheduled_payments': summary['scheduled_payments'],
            'archived_at': datetime.now()
        }
        
        # Load existing history or create new
        try:
            history_df = pd.read_parquet(storage_path)
            # Remove any existing record for this date (allow re-archiving)
            history_df = history_df[history_df['position_date'].dt.date != self.position_date.date()]
            history_df = pd.concat([history_df, pd.DataFrame([archive_record])], ignore_index=True)
        except (FileNotFoundError, Exception):
            history_df = pd.DataFrame([archive_record])
        
        history_df.to_parquet(storage_path, index=False)
        return archive_record


def get_historical_accuracy(storage_path='t1_position_history.parquet', days=30):
    """Calculate RMSE and bias from historical T+1 positions."""
    try:
        history_df = pd.read_parquet(storage_path)
    except (FileNotFoundError, Exception):
        return None
    
    if len(history_df) == 0:
        return None
    
    # Filter to recent days if specified
    if days and len(history_df) > 0:
        cutoff = datetime.now() - timedelta(days=days)
        history_df = history_df[history_df['position_date'] >= cutoff]
    
    if len(history_df) == 0:
        return None
    
    # Calculate RMSE: sqrt(mean(error^2))
    rmse = np.sqrt((history_df['absolute_error'] ** 2).mean())
    mean_bias = history_df['bias'].mean()
    
    return {
        'days_analyzed': len(history_df),
        'rmse': rmse,
        'mean_absolute_error': history_df['absolute_error'].mean(),
        'mean_bias': mean_bias,
        'bias_direction': 'OVER' if mean_bias > 0 else 'UNDER' if mean_bias < 0 else 'NEUTRAL',
        'best_day_error': history_df['absolute_error'].min(),
        'worst_day_error': history_df['absolute_error'].max(),
        'history': history_df
    }

    def archive_position(self, storage_path='t1_position_history.parquet'):
        """Archive the current day's position for historical RMSE calculation."""
        if self.position_df is None:
            self.build_position()
        
        metrics = self.get_accuracy_metrics()
        summary = self.get_position_summary()
        
        archive_record = {
            'position_date': self.position_date,
            'opening_balance': self.opening_balance_actual,
            'closing_forecast': metrics['forecast'],
            'closing_actual': metrics['actual'],
            'absolute_error': metrics['absolute_error'],
            'bias': metrics['bias'],
            'bias_direction': metrics['bias_direction'],
            'posted_transactions': summary['posted_transactions'],
            'scheduled_payments': summary['scheduled_payments'],
            'archived_at': datetime.now()
        }
        
        # Load existing history or create new
        try:
            history_df = pd.read_parquet(storage_path)
            # Remove any existing record for this date (allow re-archiving)
            history_df = history_df[history_df['position_date'].dt.date != self.position_date.date()]
            history_df = pd.concat([history_df, pd.DataFrame([archive_record])], ignore_index=True)
        except (FileNotFoundError, Exception):
            history_df = pd.DataFrame([archive_record])
        
        history_df.to_parquet(storage_path, index=False)
        return archive_record


def get_historical_accuracy(storage_path='t1_position_history.parquet', days=30):
    """Calculate RMSE and bias from historical T+1 positions."""
    try:
        history_df = pd.read_parquet(storage_path)
    except (FileNotFoundError, Exception):
        return None
    
    if len(history_df) == 0:
        return None
    
    # Filter to recent days if specified
    if days and len(history_df) > 0:
        cutoff = datetime.now() - timedelta(days=days)
        history_df = history_df[history_df['position_date'] >= cutoff]
    
    if len(history_df) == 0:
        return None
    
    # Calculate RMSE: sqrt(mean(error^2))
    rmse = np.sqrt((history_df['absolute_error'] ** 2).mean())
    mean_bias = history_df['bias'].mean()
    
    return {
        'days_analyzed': len(history_df),
        'rmse': rmse,
        'mean_absolute_error': history_df['absolute_error'].mean(),
        'mean_bias': mean_bias,
        'bias_direction': 'OVER' if mean_bias > 0 else 'UNDER' if mean_bias < 0 else 'NEUTRAL',
        'best_day_error': history_df['absolute_error'].min(),
        'worst_day_error': history_df['absolute_error'].max(),
        'history': history_df
    }



def simulate_intraday_data(position_date, forecast_df=None):
    np.random.seed(position_date.day)
    bank_transactions = []
    for i in range(np.random.randint(3, 8)):
        txn_type = np.random.choice(['WIRE', 'ACH', 'LOCKBOX'], p=[0.3, 0.5, 0.2])
        bank_transactions.append({'time': position_date.replace(hour=np.random.randint(6, 10)), 'type': 'CREDIT', 'amount': np.random.uniform(50000, 500000), 'description': f'{txn_type} RECEIPT - CUSTOMER {1000 + i}'})
    for i in range(np.random.randint(1, 4)):
        bank_transactions.append({'time': position_date.replace(hour=np.random.randint(6, 9)), 'type': 'DEBIT', 'amount': np.random.uniform(10000, 100000), 'description': f'CHECK CLEARING #{10000 + i}'})
    sap_payments = []
    for i in range(np.random.randint(1, 4)):
        sap_payments.append({'payment_method': 'WIRE', 'amount': np.random.uniform(100000, 1000000), 'vendor': f'VENDOR-{2000 + i}', 'status': np.random.choice(['SCHEDULED', 'APPROVED', 'RELEASED'])})
    for i in range(np.random.randint(5, 15)):
        sap_payments.append({'payment_method': 'ACH', 'amount': np.random.uniform(20000, 200000), 'vendor': f'VENDOR-{3000 + i}', 'status': np.random.choice(['SCHEDULED', 'APPROVED'])})
    return {'bank_transactions': bank_transactions, 'sap_payments': sap_payments}
