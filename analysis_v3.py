"""
Analysis Module v3 - Focused Outlier Detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any


class OutlierDetector:
    """Focused outlier detection for actionable treasury insights."""
    
    HIGH_THRESHOLD = 3.0
    MEDIUM_THRESHOLD = 2.5
    
    CATEGORY_NAMES = {
        'AR': 'Accounts Receivable',
        'AP': 'Accounts Payable', 
        'PAYROLL': 'Payroll',
        'IC_IN': 'Intercompany In',
        'IC_OUT': 'Intercompany Out',
        'TAX': 'Tax Payment',
        'DEBT': 'Debt Service',
        'INV_INC': 'Investment Income',
    }
    
    def __init__(self):
        self.category_stats = {}
        self.net_flow_stats = {}
        self.results = None
        self.outliers_df = None
    
    def detect(self, daily_cash, category_df=None):
        df = daily_cash.copy()
        if 'is_banking_day' in df.columns:
            df = df[df['is_banking_day']].copy()
        df = df.sort_values('date').reset_index(drop=True)
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['is_outlier'] = False
        
        outliers_list = []
        
        # Method 1: Net Cash Flow outliers
        net_outliers = self._detect_net_flow_outliers(df)
        outliers_list.extend(net_outliers)
        
        # Method 2: Category-specific outliers
        if category_df is not None:
            cat_outliers = self._detect_category_outliers(df, category_df)
            outliers_list.extend(cat_outliers)
        
        if outliers_list:
            self.outliers_df = pd.DataFrame(outliers_list)
            self.outliers_df = self.outliers_df.sort_values('date').reset_index(drop=True)
            outlier_dates = self.outliers_df['date'].unique()
            df.loc[df['date'].isin(outlier_dates), 'is_outlier'] = True
        else:
            self.outliers_df = pd.DataFrame(columns=[
                'date', 'day_name', 'severity', 'anomaly_type', 
                'description', 'recommended_action', 'z_score', 'value', 'expected'
            ])
        
        self.results = df
        return df
    
    def _detect_net_flow_outliers(self, df):
        outliers = []
        net_flow = df['net_cash_flow']
        mean_val = net_flow.mean()
        std_val = net_flow.std()
        
        self.net_flow_stats = {
            'mean': mean_val,
            'std': std_val,
            'high_threshold': mean_val + self.HIGH_THRESHOLD * std_val,
            'low_threshold': mean_val - self.HIGH_THRESHOLD * std_val,
        }
        
        if std_val == 0:
            return outliers
        
        for idx, row in df.iterrows():
            z = (row['net_cash_flow'] - mean_val) / std_val
            
            if abs(z) >= self.MEDIUM_THRESHOLD:
                severity = 'High' if abs(z) >= self.HIGH_THRESHOLD else 'Medium'
                actual_val = row['net_cash_flow']
                pct_diff = ((actual_val - mean_val) / abs(mean_val) * 100) if mean_val != 0 else 0

                if z > 0:
                    anomaly_type = 'Unexpected Cash Surplus'
                    description = f"Actual: ${actual_val:,.0f} vs Average: ${mean_val:,.0f} ({abs(pct_diff):.0f}% higher)"
                    action = "Verify receipt source. Consider short-term investment if excess persists."
                else:
                    anomaly_type = 'Unexpected Cash Deficit'
                    description = f"Actual: ${actual_val:,.0f} vs Average: ${mean_val:,.0f} ({abs(pct_diff):.0f}% lower)"
                    action = "Verify payment validity. Check liquidity buffer."

                outliers.append({
                    'date': row['date'],
                    'day_name': row['day_name'],
                    'severity': severity,
                    'anomaly_type': anomaly_type,
                    'description': description,
                    'recommended_action': action,
                    'z_score': round(z, 2),
                    'value': actual_val,
                    'expected': mean_val,
                })
        
        return outliers
    
    def _detect_category_outliers(self, df, category_df):
        outliers = []
        cat_df = category_df.copy()
        
        if 'is_banking_day' in cat_df.columns:
            cat_df = cat_df[cat_df['is_banking_day']].copy()
        cat_df = cat_df.sort_values('date').reset_index(drop=True)
        
        categories = ['AR', 'AP', 'PAYROLL', 'TAX', 'DEBT']
        
        for cat in categories:
            if cat not in cat_df.columns:
                continue
            
            non_zero_mask = cat_df[cat] > 0
            non_zero_vals = cat_df.loc[non_zero_mask, cat]
            
            if len(non_zero_vals) < 5:
                continue
            
            mean_val = non_zero_vals.mean()
            std_val = non_zero_vals.std()
            
            self.category_stats[cat] = {'mean': mean_val, 'std': std_val, 'count': len(non_zero_vals)}
            
            if std_val == 0:
                continue
            
            for idx in cat_df[non_zero_mask].index:
                value = cat_df.loc[idx, cat]
                z = (value - mean_val) / std_val
                
                if abs(z) >= self.MEDIUM_THRESHOLD:
                    date = cat_df.loc[idx, 'date']
                    day_name = date.day_name()
                    severity = 'High' if abs(z) >= self.HIGH_THRESHOLD else 'Medium'
                    cat_name = self.CATEGORY_NAMES.get(cat, cat)
                    pct_diff = ((value - mean_val) / mean_val * 100) if mean_val != 0 else 0

                    if z > 0:
                        anomaly_type = f'{cat_name} Spike'
                        description = f"{cat_name} — Actual: ${value:,.0f} vs Average: ${mean_val:,.0f} ({abs(pct_diff):.0f}% higher)"
                        if cat == 'AR':
                            action = "Verify large receipt. Update forecast if recurring."
                        elif cat == 'AP':
                            action = "Verify payment authorization. Check for duplicates."
                        elif cat == 'PAYROLL':
                            action = "Verify payroll calculation. Check for bonuses."
                        else:
                            action = "Investigate unusual amount."
                    else:
                        anomaly_type = f'{cat_name} Shortfall'
                        description = f"{cat_name} — Actual: ${value:,.0f} vs Average: ${mean_val:,.0f} ({abs(pct_diff):.0f}% lower)"
                        if cat == 'AR':
                            action = "Follow up on delayed collections."
                        elif cat == 'AP':
                            action = "Verify all invoices processed."
                        else:
                            action = "Investigate shortfall."

                    outliers.append({
                        'date': date,
                        'day_name': day_name,
                        'severity': severity,
                        'anomaly_type': anomaly_type,
                        'description': description,
                        'recommended_action': action,
                        'z_score': round(z, 2),
                        'value': value,
                        'expected': mean_val,
                    })
        
        return outliers
    
    def get_outliers(self):
        return self.outliers_df
    
    def get_outlier_summary(self):
        if self.results is None:
            return {}
        
        total = len(self.results)
        outlier_count = len(self.outliers_df) if self.outliers_df is not None else 0
        
        severity_counts = {}
        type_counts = {}
        if self.outliers_df is not None and len(self.outliers_df) > 0:
            severity_counts = self.outliers_df['severity'].value_counts().to_dict()
            type_counts = self.outliers_df['anomaly_type'].value_counts().to_dict()
        
        return {
            'total_days': total,
            'outlier_count': outlier_count,
            'outlier_rate': outlier_count / total * 100 if total > 0 else 0,
            'by_severity': severity_counts,
            'by_type': type_counts,
            'net_flow_stats': self.net_flow_stats,
            'category_stats': self.category_stats,
        }
    
    def get_actionable_report(self):
        if self.outliers_df is None or len(self.outliers_df) == 0:
            return "No actionable outliers detected. Cash flows are within normal ranges."
        
        summary = self.get_outlier_summary()
        
        report = f"""
ACTIONABLE OUTLIER REPORT
{'='*50}

Summary:
- Total Banking Days Analyzed: {summary['total_days']}
- Outliers Requiring Attention: {summary['outlier_count']}
- High Severity: {summary['by_severity'].get('High', 0)}
- Medium Severity: {summary['by_severity'].get('Medium', 0)}

"""
        
        high_outliers = self.outliers_df[self.outliers_df['severity'] == 'High']
        if len(high_outliers) > 0:
            report += "HIGH PRIORITY ITEMS:\n"
            report += "-" * 50 + "\n"
            for _, row in high_outliers.iterrows():
                report += f"\n{row['date'].strftime('%Y-%m-%d')} ({row['day_name']})\n"
                report += f"  Type: {row['anomaly_type']}\n"
                report += f"  {row['description']}\n"
                report += f"  Action: {row['recommended_action']}\n"
        
        med_outliers = self.outliers_df[self.outliers_df['severity'] == 'Medium']
        if len(med_outliers) > 0:
            report += "\nMEDIUM PRIORITY ITEMS:\n"
            report += "-" * 50 + "\n"
            for _, row in med_outliers.iterrows():
                report += f"\n{row['date'].strftime('%Y-%m-%d')} ({row['day_name']})\n"
                report += f"  Type: {row['anomaly_type']}\n"
                report += f"  {row['description']}\n"
                report += f"  Action: {row['recommended_action']}\n"
        
        return report


class SHAPAnalyzer:
    """SHAP analysis for forecast interpretability."""
    
    def __init__(self, forecaster):
        self.forecaster = forecaster
        self.results = {}
    
    def analyze(self):
        if not self.forecaster.is_fitted:
            return {}
        
        results = {}
        if self.forecaster.inflow_model is not None:
            results['inflow'] = self._analyze_model(self.forecaster.inflow_model, 'inflow')
        if self.forecaster.outflow_model is not None:
            results['outflow'] = self._analyze_model(self.forecaster.outflow_model, 'outflow')
        
        self.results = results
        return results
    
    def _analyze_model(self, model, name):
        import pandas as pd
        training_dates = self.forecaster.training_data['date'].tail(30)
        future_df = pd.DataFrame({'ds': training_dates})
        forecast = model.predict(future_df)
        
        skip_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                    'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                    'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper']
        component_cols = [col for col in forecast.columns if col not in skip_cols]
        
        importance = {}
        for col in component_cols:
            if col in forecast.columns:
                importance[col] = abs(forecast[col]).mean()
        
        total = sum(importance.values()) or 1
        importance_pct = {k: v/total * 100 for k, v in importance.items()}
        
        importance_df = pd.DataFrame([
            {'component': k, 'importance': v, 'importance_pct': importance_pct[k]}
            for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True)
        ])
        
        return {'importance': importance_df, 'raw_importance': importance}
