import yfinance as yf
import pandas as pd
from datetime import datetime

def trade_log(data: pd.DataFrame, format: str = None) -> None:
    if format == None:
        data.to_csv("trade_log.csv")
    elif format == "db":
        import sqlite3
        con = sqlite3.connect("RT2.db")
        data.to_sql("TradeLog", con)
    return None