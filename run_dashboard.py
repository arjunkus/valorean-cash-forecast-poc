#!/usr/bin/env python3
"""
Cash Forecasting Dashboard Launcher
====================================
Run this script to start the Streamlit dashboard.
"""

import subprocess
import sys
import os

def main():
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "dashboard_enhanced.py")
    
    # Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        dashboard_path,
        "--server.port=8501",
        "--server.address=0.0.0.0",
        "--browser.gatherUsageStats=false"
    ]
    
    print("🚀 Starting Cash Forecasting Intelligence Dashboard...")
    print(f"   Dashboard URL: http://localhost:8501")
    print("-" * 50)
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
