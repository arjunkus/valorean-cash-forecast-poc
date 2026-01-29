"""
SAP Field Mapping for Cash Forecast POC
========================================
Maps POC synthetic data fields to SAP ERP tables/fields for BTP OData integration.

Usage:
    This file is a configuration/reference module. It will be imported by the
    BTP integration layer when real SAP endpoints replace synthetic data.

SAP Tables Referenced:
    - BSEG       GL Line Items (postings)
    - BKPF       Accounting Document Header
    - REGUH      Payment Program Header (scheduled payments)
    - REGUP      Payment Program Item
    - FAGLFLEXA  New GL Line Items (S/4HANA)
    - T012K      Bank Account Master
    - ACDOCA     Universal Journal (S/4HANA)
    - PA0008     HR Payroll (basic pay)
    - FEBEP      Electronic Bank Statement Line Items
    - FEBKO      Electronic Bank Statement Header
"""

import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. POC Category -> SAP Table / Field Mapping
# ---------------------------------------------------------------------------

# Each entry maps a POC category code to:
#   sap_table        - Primary SAP table to query via OData
#   sap_fields       - Key fields to SELECT
#   filter_logic     - OData $filter expression template
#   amount_field     - Field containing the monetary value
#   flow_direction   - INFLOW or OUTFLOW (for sign convention)
#   description      - Human-readable explanation of the mapping

CATEGORY_TO_SAP = {
    # --- INFLOWS ---
    'AR_WIRE': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUZEI', 'BUDAT', 'WRBTR', 'DMBTR', 'KOART', 'ZLSCH'],
        'filter_logic': "KOART eq 'D' and ZLSCH eq 'T'",  # Debtors, wire transfer
        'amount_field': 'DMBTR',  # Amount in local currency
        'flow_direction': 'INFLOW',
        'description': 'Customer wire receipts from AR subledger. '
                       'KOART=D (debtor), ZLSCH=T (bank transfer).',
    },
    'AR_ACH': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUZEI', 'BUDAT', 'WRBTR', 'DMBTR', 'KOART', 'ZLSCH'],
        'filter_logic': "KOART eq 'D' and ZLSCH eq 'E'",  # Debtors, ACH/electronic
        'amount_field': 'DMBTR',
        'flow_direction': 'INFLOW',
        'description': 'Customer ACH credits. ZLSCH=E (ACH/electronic payment).',
    },
    'AR_LOCKBOX': {
        'sap_table': 'FEBEP',
        'sap_fields': ['KUESSION', 'AZESSION', 'KWBTR', 'VALUT', 'VGSART'],
        'filter_logic': "VGSART eq 'LBX'",  # Lockbox transaction type
        'amount_field': 'KWBTR',
        'flow_direction': 'INFLOW',
        'description': 'Lockbox receipts posted via electronic bank statement (FEBEP). '
                       'VGSART identifies the bank statement transaction type.',
    },
    'INV_INC': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUDAT', 'DMBTR', 'HKONT', 'KOART'],
        'filter_logic': "KOART eq 'S' and HKONT ge '4100' and HKONT le '4199'",
        'amount_field': 'DMBTR',
        'flow_direction': 'INFLOW',
        'description': 'Investment income postings to GL range 4100-4199 '
                       '(interest income, dividends). KOART=S (GL account).',
    },
    'IC_IN': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUDAT', 'DMBTR', 'HKONT', 'KOART', 'VBUND'],
        'filter_logic': "VBUND ne '' and DMBTR gt 0",  # Trading partner populated = IC
        'amount_field': 'DMBTR',
        'flow_direction': 'INFLOW',
        'description': 'Intercompany inflows. VBUND (trading partner) populated '
                       'indicates intercompany. Positive DMBTR = receipt.',
    },
    'OTHER_IN': {
        'sap_table': 'FEBEP',
        'sap_fields': ['KUESSION', 'KWBTR', 'VALUT', 'VGSART'],
        'filter_logic': "KWBTR gt 0",  # Credits not matched to above categories
        'amount_field': 'KWBTR',
        'flow_direction': 'INFLOW',
        'description': 'Unclassified bank statement credits. Catchall for receipts '
                       'not matched to AR/INV/IC categories.',
    },

    # --- OUTFLOWS ---
    'PAYROLL': {
        'sap_table': 'REGUH',
        'sap_fields': ['LAUFD', 'LAUFI', 'ZBUKR', 'RWBTR', 'ZLSCH', 'XVORL'],
        'filter_logic': "ZLSCH eq 'P'",  # Payment method = Payroll
        'amount_field': 'RWBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Payroll disbursements from payment program. '
                       'Alternatively sourced from PA0008/PC_PAYRESULT cluster.',
    },
    'AP_WIRE': {
        'sap_table': 'REGUH',
        'sap_fields': ['LAUFD', 'LAUFI', 'LIFNR', 'ZBUKR', 'RWBTR', 'ZLSCH', 'XVORL'],
        'filter_logic': "ZLSCH eq 'T' and XVORL eq ''",  # Wire, not proposal
        'amount_field': 'RWBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Vendor wire payments from payment program (F110). '
                       'ZLSCH=T (bank transfer), XVORL=blank (not just a proposal).',
    },
    'AP_ACH': {
        'sap_table': 'REGUH',
        'sap_fields': ['LAUFD', 'LAUFI', 'LIFNR', 'ZBUKR', 'RWBTR', 'ZLSCH', 'XVORL'],
        'filter_logic': "ZLSCH eq 'E' and XVORL eq ''",
        'amount_field': 'RWBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Vendor ACH payments. ZLSCH=E (electronic/ACH).',
    },
    'AP_CHECK': {
        'sap_table': 'REGUH',
        'sap_fields': ['LAUFD', 'LAUFI', 'LIFNR', 'ZBUKR', 'RWBTR', 'ZLSCH', 'XVORL'],
        'filter_logic': "ZLSCH eq 'C' and XVORL eq ''",
        'amount_field': 'RWBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Vendor check payments. ZLSCH=C (check). '
                       'Check clearing date from PAYR table.',
    },
    'TAX': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUDAT', 'DMBTR', 'HKONT', 'KOART'],
        'filter_logic': "HKONT ge '2100' and HKONT le '2199'",
        'amount_field': 'DMBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Tax payments posted to GL range 2100-2199 (tax payable accounts).',
    },
    'DEBT': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUDAT', 'DMBTR', 'HKONT', 'KOART'],
        'filter_logic': "HKONT ge '2300' and HKONT le '2399'",
        'amount_field': 'DMBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Debt service payments (principal + interest) '
                       'posted to GL range 2300-2399 (loans payable).',
    },
    'IC_OUT': {
        'sap_table': 'FAGLFLEXA',
        'sap_fields': ['BELNR', 'BUDAT', 'DMBTR', 'HKONT', 'KOART', 'VBUND'],
        'filter_logic': "VBUND ne '' and DMBTR lt 0",
        'amount_field': 'DMBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Intercompany outflows. VBUND populated + negative amount.',
    },
    'OTHER_OUT': {
        'sap_table': 'FEBEP',
        'sap_fields': ['KUESSION', 'KWBTR', 'VALUT', 'VGSART'],
        'filter_logic': "KWBTR lt 0",
        'amount_field': 'KWBTR',
        'flow_direction': 'OUTFLOW',
        'description': 'Unclassified bank statement debits not matched to AP/TAX/DEBT/IC.',
    },
}

# Aggregate forecast categories used by Prophet (data_simulator_v3 / models_prophet_v6).
# These map to one or more granular T+1 categories above.
FORECAST_CATEGORY_TO_GRANULAR = {
    'AR':      ['AR_WIRE', 'AR_ACH', 'AR_LOCKBOX'],  # Forecast splits evenly (/3)
    'INV_INC': ['INV_INC'],
    'IC_IN':   ['IC_IN'],
    'PAYROLL': ['PAYROLL'],
    'AP':      ['AP_WIRE', 'AP_ACH', 'AP_CHECK'],    # Forecast splits evenly (/3)
    'TAX':     ['TAX'],
    'CAPEX':   [],  # No T+1 granular equivalent; tracked separately
    'DEBT':    ['DEBT'],
    'IC_OUT':  ['IC_OUT'],
}


# ---------------------------------------------------------------------------
# 2. POC Transaction Status -> SAP Status Mapping
# ---------------------------------------------------------------------------

# POC statuses used in DailyPositionManager and simulate_intraday_data()
# mapped to SAP document/payment statuses.

STATUS_TO_SAP = {
    # --- Intraday (bank-sourced) statuses ---
    'POSTED': {
        'sap_status_field': 'FEBEP-XBLNR',   # External document reference present
        'sap_status_value': 'Posted',
        'description': 'Transaction cleared on bank statement and posted in SAP. '
                       'FEBEP record exists with matched GL posting (BELNR populated).',
    },

    # --- SAP payment program (F110) statuses ---
    'SCHEDULED': {
        'sap_status_field': 'REGUH-XVORL',
        'sap_status_value': 'X',              # Proposal flag set
        'description': 'Payment is in proposal run (F110 step 1). '
                       'REGUH.XVORL = X means proposal not yet approved.',
    },
    'APPROVED': {
        'sap_status_field': 'REGUH-XVORL',
        'sap_status_value': '',               # Proposal flag cleared after approval
        'description': 'Payment proposal approved (F110 step 2). '
                       'REGUH.XVORL cleared, but payment not yet executed.',
    },
    'RELEASED': {
        'sap_status_field': 'REGUH-XVORL',
        'sap_status_value': '',
        'description': 'Payment executed and sent to bank (F110 step 3). '
                       'REGUH record has VBLNR (payment document number) populated.',
    },

    # --- POC display-only statuses (no direct SAP field) ---
    'Estimated': {
        'sap_status_field': None,
        'sap_status_value': None,
        'description': 'No bank or SAP data yet; POC uses forecast * 0.8 as estimate.',
    },
    'Forecast': {
        'sap_status_field': None,
        'sap_status_value': None,
        'description': 'No actual data; using raw forecast value as placeholder.',
    },
}

# Bank statement transaction type codes (FEBEP-VGSART) for classifying
# incoming bank transactions into POC categories.
BANK_TXN_TYPE_TO_CATEGORY = {
    'WIR': 'AR_WIRE',      # Wire transfer
    'ACH': 'AR_ACH',       # ACH credit
    'LBX': 'AR_LOCKBOX',   # Lockbox deposit
    'CHK': 'AP_CHECK',     # Check clearing (debit)
    'TRF': 'IC_IN',        # Internal transfer (credit side)
}


# ---------------------------------------------------------------------------
# 3. Transformation Functions: SAP OData -> POC DataFrame
# ---------------------------------------------------------------------------

def odata_bank_statement_to_transactions(odata_results: list[dict]) -> list[dict]:
    """Convert SAP FEBEP OData response to POC bank_transactions format.

    Input (SAP OData):
        Each dict has FEBEP fields: KUESSION, KWBTR, VALUT, VGSART, BELNR, etc.

    Output (POC format for DailyPositionManager.load_intraday_from_bank):
        List of dicts with: time, type (CREDIT/DEBIT), amount, description
    """
    transactions = []
    for item in odata_results:
        amount = float(item.get('KWBTR', 0))
        txn_type = 'CREDIT' if amount > 0 else 'DEBIT'
        # VGSART = bank statement transaction type
        vgsart = item.get('VGSART', '')
        description = _build_description(vgsart, item)
        valut = item.get('VALUT', '')
        transactions.append({
            'time': _parse_sap_datetime(valut),
            'type': txn_type,
            'amount': abs(amount),
            'description': description,
        })
    return transactions


def odata_payment_program_to_sap_payments(odata_results: list[dict]) -> list[dict]:
    """Convert SAP REGUH/REGUP OData response to POC sap_payments format.

    Input (SAP OData):
        Each dict has REGUH fields: LAUFD, LAUFI, LIFNR, RWBTR, ZLSCH, XVORL, VBLNR

    Output (POC format for DailyPositionManager.load_sap_payment_queue):
        List of dicts with: payment_method, amount, vendor, status
    """
    payments = []
    for item in odata_results:
        zlsch = item.get('ZLSCH', '')
        method = SAP_ZLSCH_TO_METHOD.get(zlsch, 'OTHER')
        status = _resolve_payment_status(item)
        payments.append({
            'payment_method': method,
            'amount': abs(float(item.get('RWBTR', 0))),
            'vendor': item.get('LIFNR', 'Unknown'),
            'status': status,
        })
    return payments


def odata_gl_to_category_df(odata_results: list[dict], date_col: str = 'BUDAT') -> pd.DataFrame:
    """Convert SAP FAGLFLEXA OData response to POC category_details format.

    Input (SAP OData):
        Each dict has: BUDAT, DMBTR, HKONT, KOART, VBUND, ZLSCH, etc.

    Output:
        DataFrame with columns matching data_simulator_v3 category_details:
        date, AR, INV_INC, IC_IN, PAYROLL, AP, TAX, CAPEX, DEBT, IC_OUT,
        total_inflow, total_outflow, opening_balance, closing_balance, etc.
    """
    if not odata_results:
        return pd.DataFrame()

    rows = []
    for item in odata_results:
        rows.append({
            'date': _parse_sap_date(item.get(date_col, '')),
            'amount': float(item.get('DMBTR', 0)),
            'gl_account': item.get('HKONT', ''),
            'account_type': item.get('KOART', ''),
            'trading_partner': item.get('VBUND', ''),
            'payment_method': item.get('ZLSCH', ''),
        })
    raw = pd.DataFrame(rows)

    # Classify each line into a POC category
    raw['poc_category'] = raw.apply(_classify_gl_line, axis=1)

    # Pivot to one row per date, one column per category
    daily = raw.groupby(['date', 'poc_category'])['amount'].sum().unstack(fill_value=0).reset_index()

    # Ensure all expected columns exist
    for cat in ['AR', 'INV_INC', 'IC_IN', 'PAYROLL', 'AP', 'TAX', 'CAPEX', 'DEBT', 'IC_OUT']:
        if cat not in daily.columns:
            daily[cat] = 0.0

    daily['total_inflow'] = daily['AR'] + daily['INV_INC'] + daily['IC_IN']
    daily['total_outflow'] = daily['PAYROLL'] + daily['AP'] + daily['TAX'] + daily['CAPEX'] + daily['DEBT'] + daily['IC_OUT']
    daily['net_cash_flow'] = daily['total_inflow'] - daily['total_outflow']
    daily['inflow'] = daily['total_inflow']
    daily['outflow'] = daily['total_outflow']

    # Balance columns require a starting balance from T012K or treasury
    # Placeholder: caller must set opening_balance and compute closing_balance
    daily['opening_balance'] = np.nan
    daily['closing_balance'] = np.nan

    daily = daily.sort_values('date').reset_index(drop=True)
    return daily


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

# SAP payment method codes (ZLSCH) -> POC method strings
SAP_ZLSCH_TO_METHOD = {
    'T': 'WIRE',    # Bank transfer
    'E': 'ACH',     # Electronic / ACH
    'C': 'CHECK',   # Check
    'P': 'PAYROLL', # Payroll run
    'S': 'WIRE',    # SWIFT (treat as wire)
}


def _parse_sap_date(value: str) -> datetime:
    """Parse SAP date formats: 'YYYYMMDD' or ISO '2025-01-15'."""
    if not value:
        return datetime.now()
    value = value.replace('-', '').replace('/', '')
    try:
        return datetime.strptime(value[:8], '%Y%m%d')
    except ValueError:
        return datetime.now()


def _parse_sap_datetime(value: str) -> datetime:
    """Parse SAP datetime. Falls back to date-only parsing."""
    if not value:
        return datetime.now()
    # OData often returns /Date(timestamp)/ or ISO format
    if '/Date(' in str(value):
        try:
            ms = int(str(value).split('(')[1].split(')')[0].split('+')[0])
            return datetime.fromtimestamp(ms / 1000)
        except (ValueError, IndexError):
            pass
    return _parse_sap_date(str(value))


def _build_description(vgsart: str, item: dict) -> str:
    """Build a human-readable description from bank statement fields."""
    belnr = item.get('BELNR', '')
    parts = []
    if vgsart:
        parts.append(vgsart)
    if belnr:
        parts.append(f'Doc#{belnr}')
    return ' '.join(parts) if parts else 'Bank Transaction'


def _resolve_payment_status(item: dict) -> str:
    """Determine POC status from REGUH fields."""
    if item.get('VBLNR'):       # Payment document exists -> executed
        return 'RELEASED'
    if item.get('XVORL') == 'X':  # Proposal flag set
        return 'SCHEDULED'
    return 'APPROVED'


def _classify_gl_line(row: pd.Series) -> str:
    """Classify a FAGLFLEXA line item into a POC aggregate forecast category.

    This maps individual GL postings back to the 9 categories used by
    data_simulator_v3 and models_prophet_v6 (AR, AP, PAYROLL, etc.).
    """
    koart = row.get('account_type', '')
    hkont = str(row.get('gl_account', ''))
    vbund = row.get('trading_partner', '')
    amount = row.get('amount', 0)

    # Intercompany takes precedence
    if vbund:
        return 'IC_IN' if amount > 0 else 'IC_OUT'

    # Debtor postings = AR
    if koart == 'D':
        return 'AR'

    # Creditor postings = AP
    if koart == 'K':
        return 'AP'

    # GL-based classification by account range
    if hkont[:4] >= '4100' and hkont[:4] <= '4199':
        return 'INV_INC'
    if hkont[:4] >= '2100' and hkont[:4] <= '2199':
        return 'TAX'
    if hkont[:4] >= '2300' and hkont[:4] <= '2399':
        return 'DEBT'

    # Fallback: classify by sign
    return 'AR' if amount > 0 else 'AP'
