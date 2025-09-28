#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Collect daily prices, intraday (optional), company overview, and financials for a list of tickers.

import argparse
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd

try:
    from vnstock import Vnstock, Trading
except Exception as e:
    print("[ERROR] Import vnstock failed. Install with: pip install -U vnstock", file=sys.stderr)
    raise

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def vietnamese_to_ascii(text: str) -> str:
    """Convert Vietnamese text to ASCII, removing diacritics."""
    if not isinstance(text, str):
        return str(text)
    # Normalize to NFD (decomposed form)
    normalized = unicodedata.normalize('NFD', text)
    # Remove diacritics by filtering out combining characters
    ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    return ascii_text

def is_index_symbol(symbol: str) -> bool:
    """Check if symbol is an index (like VNINDEX, HNX-INDEX, etc.)"""
    index_patterns = ['INDEX', 'HNX', 'UPCOM']
    symbol_upper = symbol.upper()
    return any(pattern in symbol_upper for pattern in index_patterns)

def save_df(df: Optional[pd.DataFrame], path: Path, desc: str) -> None:
    if df is None or (hasattr(df, "empty") and getattr(df, "empty", False)):
        print(f"[WARN] No data for {desc}. Skipped: {path}")
        return
    df = df.copy()
    
    # Convert column names: Vietnamese to ASCII, clean up
    df.columns = [vietnamese_to_ascii(str(c)).strip().lower().replace(" ", "_") for c in df.columns]
    
    # Convert string data to ASCII to avoid encoding issues
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: vietnamese_to_ascii(str(x)) if pd.notnull(x) else x)
    
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"[OK] Saved {desc} -> {path} ({len(df):,} rows)")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect VN stock data using vnstock")
    p.add_argument("--tickers", nargs="+", required=True, help="Symbols, e.g., FPT HPG VNM VNINDEX")
    p.add_argument("--start", type=str, default="2010-01-01")
    p.add_argument("--end", type=str, default=str(date.today()))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--source", type=str, default="VCI")
    p.add_argument("--intraday", action="store_true")
    p.add_argument("--no-financials", action="store_true")
    p.add_argument("--sleep", type=float, default=0.5)
    return p.parse_args()

def collect_for_symbol(symbol: str, source: str, start: str, end: str, out_dir: Path,
                       do_intraday: bool, fetch_financials: bool, sleep_sec: float) -> None:
    print("=" * 80)
    print(f"[INFO] Collecting {symbol} | {start} -> {end} | source={source}")
    symbol_dir = out_dir / symbol.upper()
    ensure_dir(symbol_dir)

    # Check if this is an index (VNINDEX, etc.) - indices don't have company/financial data
    is_index = is_index_symbol(symbol)
    if is_index:
        print(f"[INFO] {symbol} is an index - skipping company/financial data")

    stock = Vnstock().stock(symbol=symbol.upper(), source=source)

    # Daily OHLCV
    try:
        daily = stock.quote.history(start=start, end=end, interval="1D")
        save_df(daily, symbol_dir / "prices_daily.csv", f"{symbol} daily history")
    except Exception as e:
        print(f"[ERROR] daily history {symbol}: {e}", file=sys.stderr)

    time.sleep(sleep_sec)

    # Intraday (optional)
    if do_intraday:
        try:
            intraday = stock.quote.intraday(symbol=symbol.upper(), page_size=100_000, show_log=False)
            save_df(intraday, symbol_dir / "intraday_ticks.csv", f"{symbol} intraday")
        except Exception as e:
            print(f"[ERROR] intraday {symbol}: {e}", file=sys.stderr)
        time.sleep(sleep_sec)

    # Skip company and financial data for indices
    if is_index:
        print(f"[DONE] {symbol} (index - no company/financial data)")
        return

    # Company overview
    try:
        company = stock.company.overview()
        save_df(company, symbol_dir / "company_overview.csv", f"{symbol} company overview")
    except Exception as e:
        print(f"[ERROR] company overview {symbol}: {e}", file=sys.stderr)

    time.sleep(sleep_sec)

    if fetch_financials:
        for period in ("year", "quarter"):
            try:
                bs = stock.finance.balance_sheet(period=period, lang="vi", dropna=True)
                save_df(bs, symbol_dir / f"finance_balance_sheet_{period}.csv", f"{symbol} balance sheet ({period})")
            except Exception as e:
                print(f"[ERROR] balance_sheet {period} {symbol}: {e}", file=sys.stderr)
            time.sleep(sleep_sec)
            try:
                is_df = stock.finance.income_statement(period=period, lang="vi", dropna=True)
                save_df(is_df, symbol_dir / f"finance_income_statement_{period}.csv", f"{symbol} income statement ({period})")
            except Exception as e:
                print(f"[ERROR] income_statement {period} {symbol}: {e}", file=sys.stderr)
            time.sleep(sleep_sec)
            try:
                cf = stock.finance.cash_flow(period=period, dropna=True)
                save_df(cf, symbol_dir / f"finance_cash_flow_{period}.csv", f"{symbol} cash flow ({period})")
            except Exception as e:
                print(f"[ERROR] cash_flow {period} {symbol}: {e}", file=sys.stderr)
            time.sleep(sleep_sec)

        try:
            ratios = stock.finance.ratio(period="year", lang="vi", dropna=True)
            save_df(ratios, symbol_dir / "finance_ratios_year.csv", f"{symbol} financial ratios (year)")
        except Exception as e:
            print(f"[ERROR] ratios {symbol}: {e}", file=sys.stderr)

    print(f"[DONE] {symbol}")

def main():
    args = parse_args()
    out_dir = args.out
    ensure_dir(out_dir)
    tickers = [t.strip().upper() for t in args.tickers]
    print(f"[RUN] {tickers} -> {out_dir.resolve()}")

    # Price board snapshot
    try:
        pb = Trading(source=args.source).price_board(tickers)
        save_df(pb, out_dir / "price_board_snapshot.csv", "price board snapshot")
    except Exception as e:
        print(f"[WARN] price_board snapshot: {e}", file=sys.stderr)

    for sym in tickers:
        collect_for_symbol(sym, args.source, args.start, args.end, out_dir,
                           args.intraday, not args.no_financials, args.sleep)
    print("[ALL DONE]")

if __name__ == "__main__":
    main()