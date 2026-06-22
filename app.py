# -*- coding: utf-8 -*-
"""
모바일용 주식 전략 판단기 V1.3
- 자동매매가 아니라 매수/매도 의사결정 보조 앱입니다.
- 현금/거래 기록은 사용자가 입력한 값 기준으로 계산합니다.
- 무료 시세 데이터는 지연/누락될 수 있으므로, 실시간 매매 전에는 증권앱 현재가로 확인하세요.
- 관심종목 다중 스캔과 캡쳐 OCR 자동입력을 지원합니다.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception:  # pragma: no cover
    Image = None
    ImageFilter = None
    ImageOps = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

APP_TITLE = "주식 전략 판단기"
DB_DIR = "data"
DB_PATH = os.path.join(DB_DIR, "trader_state.sqlite")
MIN_BUDGET_KRW = 1_000_000
MAX_BUDGET_KRW = 1_100_000
DEFAULT_BUDGET_KRW = 1_050_000


# -----------------------------
# 기본 유틸
# -----------------------------

def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_krw(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.0f}원"


def fmt_num(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{value:,.{digits}f}"


def fmt_price(value: float, market: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if market == "KRX":
        return f"{value:,.0f}원"
    return f"${value:,.2f}"


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def clean_number_text(text: str) -> Optional[float]:
    if not text:
        return None
    cleaned = text.replace(",", "").replace(" ", "").replace("원", "").replace("$", "")
    cleaned = re.sub(r"[^0-9.\-]", "", cleaned)
    if cleaned in ["", ".", "-", "-."]:
        return None
    try:
        return float(cleaned)
    except Exception:
        return None


def round_price(value: float, market: str) -> float:
    """대략적인 호가 단위 반영. 정확한 호가 규칙은 증권앱 기준으로 최종 확인."""
    if value <= 0:
        return 0.0
    if market != "KRX":
        return round(value, 2)
    if value < 2_000:
        unit = 1
    elif value < 5_000:
        unit = 5
    elif value < 20_000:
        unit = 10
    elif value < 50_000:
        unit = 50
    elif value < 200_000:
        unit = 100
    elif value < 500_000:
        unit = 500
    else:
        unit = 1_000
    return round(value / unit) * unit


def normalize_symbol(raw_symbol: str, market: str) -> str:
    symbol = raw_symbol.strip().upper()
    if market == "KRX":
        digits = "".join(ch for ch in symbol if ch.isdigit())
        if len(digits) == 6:
            return f"{digits}.KS"
        return symbol
    return symbol.replace(" ", "")


def display_symbol(raw_symbol: str, market: str) -> str:
    symbol = raw_symbol.strip().upper()
    if market == "KRX":
        digits = "".join(ch for ch in symbol if ch.isdigit())
        return digits if digits else symbol
    return symbol.replace(" ", "")


def infer_market_from_symbol(symbol: str, default_market: str = "KRX") -> str:
    s = symbol.strip().upper().replace(".KS", "").replace(".KQ", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) == 6:
        return "KRX"
    if re.fullmatch(r"[A-Z]{1,6}", s):
        return "US"
    return default_market


def parse_symbol_lines(text: str, default_market: str = "KRX", auto_detect: bool = True) -> List[Tuple[str, str]]:
    """관심종목 입력: KRX:005930, US:NVDA, 005930, NVDA 등을 처리."""
    if not text:
        return []
    chunks = re.split(r"[\n,;\t]+", text)
    rows: List[Tuple[str, str]] = []
    seen = set()
    for chunk in chunks:
        item = chunk.strip()
        if not item:
            continue
        market = default_market
        raw = item
        if ":" in item:
            prefix, raw = item.split(":", 1)
            prefix = prefix.strip().upper()
            if prefix in ["KRX", "KOR", "국장", "K"]:
                market = "KRX"
            elif prefix in ["US", "USA", "미장", "U"]:
                market = "US"
        elif auto_detect:
            market = infer_market_from_symbol(item, default_market)
        symbol = display_symbol(raw, market)
        if not symbol:
            continue
        key = (symbol, market)
        if key not in seen:
            rows.append(key)
            seen.add(key)
    return rows


# -----------------------------
# DB
# -----------------------------

def get_conn() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cash_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            event_type TEXT NOT NULL,
            amount_krw REAL NOT NULL,
            memo TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL NOT NULL,
            price REAL,
            total_krw REAL NOT NULL,
            memo TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            baseline_price REAL NOT NULL,
            current_price REAL NOT NULL,
            market_mode TEXT,
            decision TEXT,
            buy_low REAL,
            buy_high REAL,
            chase_limit REAL,
            stop_price REAL,
            target1 REAL,
            target2 REAL,
            target3 REAL,
            risk_reward REAL,
            basis_json TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            symbol TEXT NOT NULL,
            market TEXT NOT NULL,
            name TEXT,
            memo TEXT,
            is_active INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    conn.commit()
    conn.close()


def cash_balance_krw() -> float:
    conn = get_conn()
    cash_events = pd.read_sql_query("SELECT * FROM cash_events", conn)
    trades = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()

    balance = 0.0
    if not cash_events.empty:
        for _, row in cash_events.iterrows():
            amount = safe_float(row["amount_krw"])
            event_type = row["event_type"]
            if event_type in ["입금", "현금보정+"]:
                balance += amount
            elif event_type in ["출금", "현금보정-"]:
                balance -= amount
    if not trades.empty:
        for _, row in trades.iterrows():
            total = safe_float(row["total_krw"])
            if row["side"] == "매수":
                balance -= total
            elif row["side"] == "매도":
                balance += total
    return balance


def add_cash_event(event_type: str, amount_krw: float, memo: str = "") -> None:
    conn = get_conn()
    conn.execute(
        "INSERT INTO cash_events(created_at, event_type, amount_krw, memo) VALUES (?, ?, ?, ?)",
        (now_text(), event_type, float(amount_krw), memo),
    )
    conn.commit()
    conn.close()


def add_trade(symbol: str, market: str, side: str, qty: float, price: float, total_krw: float, memo: str = "") -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO trades(created_at, symbol, market, side, qty, price, total_krw, memo)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (now_text(), symbol, market, side, float(qty), float(price), float(total_krw), memo),
    )
    conn.commit()
    conn.close()


def add_watchlist_item(symbol: str, market: str, name: str = "", memo: str = "") -> None:
    conn = get_conn()
    symbol = display_symbol(symbol, market)
    existing = conn.execute(
        "SELECT id FROM watchlist WHERE symbol = ? AND market = ? AND is_active = 1",
        (symbol, market),
    ).fetchone()
    if existing:
        conn.execute(
            "UPDATE watchlist SET name = ?, memo = ? WHERE id = ?",
            (name, memo, existing["id"]),
        )
    else:
        conn.execute(
            "INSERT INTO watchlist(created_at, symbol, market, name, memo, is_active) VALUES (?, ?, ?, ?, ?, 1)",
            (now_text(), symbol, market, name, memo),
        )
    conn.commit()
    conn.close()


def deactivate_watchlist_item(item_id: int) -> None:
    conn = get_conn()
    conn.execute("UPDATE watchlist SET is_active = 0 WHERE id = ?", (int(item_id),))
    conn.commit()
    conn.close()


def load_watchlist(active_only: bool = True) -> pd.DataFrame:
    conn = get_conn()
    where = "WHERE is_active = 1" if active_only else ""
    df = pd.read_sql_query(f"SELECT * FROM watchlist {where} ORDER BY market, symbol", conn)
    conn.close()
    return df


def load_table(table: str) -> pd.DataFrame:
    conn = get_conn()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY id DESC", conn)
    finally:
        conn.close()
    return df


def calculate_positions() -> pd.DataFrame:
    conn = get_conn()
    trades = pd.read_sql_query("SELECT * FROM trades ORDER BY created_at, id", conn)
    conn.close()
    if trades.empty:
        return pd.DataFrame(columns=["symbol", "market", "qty", "cost_krw", "avg_cost_krw_per_share"])

    positions: Dict[Tuple[str, str], Dict[str, float]] = {}
    for _, row in trades.iterrows():
        key = (row["symbol"], row["market"])
        positions.setdefault(key, {"qty": 0.0, "cost_krw": 0.0})
        qty = safe_float(row["qty"])
        total = safe_float(row["total_krw"])
        if row["side"] == "매수":
            positions[key]["qty"] += qty
            positions[key]["cost_krw"] += total
        elif row["side"] == "매도":
            old_qty = positions[key]["qty"]
            old_cost = positions[key]["cost_krw"]
            if old_qty > 0:
                avg_cost = old_cost / old_qty
                reduce_cost = min(qty, old_qty) * avg_cost
                positions[key]["qty"] = max(0.0, old_qty - qty)
                positions[key]["cost_krw"] = max(0.0, old_cost - reduce_cost)

    rows = []
    for (symbol, market), pos in positions.items():
        if pos["qty"] > 1e-9:
            rows.append(
                {
                    "symbol": symbol,
                    "market": market,
                    "qty": pos["qty"],
                    "cost_krw": pos["cost_krw"],
                    "avg_cost_krw_per_share": pos["cost_krw"] / pos["qty"] if pos["qty"] else 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_strategy(strategy: Dict) -> None:
    conn = get_conn()
    conn.execute(
        "UPDATE strategies SET is_active = 0 WHERE symbol = ? AND market = ? AND is_active = 1",
        (strategy["symbol"], strategy["market"]),
    )
    conn.execute(
        """
        INSERT INTO strategies(
            created_at, updated_at, symbol, market, baseline_price, current_price, market_mode, decision,
            buy_low, buy_high, chase_limit, stop_price, target1, target2, target3, risk_reward, basis_json, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            now_text(),
            now_text(),
            strategy["symbol"],
            strategy["market"],
            strategy["current_price"],
            strategy["current_price"],
            strategy.get("market_mode"),
            strategy.get("decision"),
            strategy.get("buy_low"),
            strategy.get("buy_high"),
            strategy.get("chase_limit"),
            strategy.get("stop_price"),
            strategy.get("target1"),
            strategy.get("target2"),
            strategy.get("target3"),
            strategy.get("risk_reward"),
            json.dumps(strategy.get("basis", {}), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def get_active_strategy(symbol: str, market: str) -> Optional[sqlite3.Row]:
    conn = get_conn()
    row = conn.execute(
        """
        SELECT * FROM strategies
        WHERE symbol = ? AND market = ? AND is_active = 1
        ORDER BY id DESC LIMIT 1
        """,
        (symbol, market),
    ).fetchone()
    conn.close()
    return row


# -----------------------------
# 시세/지표
# -----------------------------

@st.cache_data(ttl=60)
def fetch_ohlcv(yf_symbol: str, interval: str, period: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance가 설치되어 있지 않습니다. requirements.txt 설치를 확인하세요.")
    df = yf.download(yf_symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    rename_map = {}
    for col in df.columns:
        low = str(col).lower()
        if "date" in low or "datetime" in low:
            rename_map[col] = "Date"
    df = df.rename(columns=rename_map)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    for col in needed:
        if col not in df.columns:
            return pd.DataFrame()
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"].astype(float)
    high = out["High"].astype(float)
    low = out["Low"].astype(float)
    volume = out["Volume"].astype(float).replace(0, np.nan)

    out["MA5"] = close.rolling(5).mean()
    out["MA20"] = close.rolling(20).mean()
    out["MA60"] = close.rolling(60).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["RSI14"] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR14"] = tr.rolling(14).mean()
    out["VOL_MA20"] = volume.rolling(20).mean()
    out["VOL_RATIO"] = volume / out["VOL_MA20"]
    return out


def local_levels(df: pd.DataFrame, current_price: float) -> Tuple[float, float]:
    recent = df.tail(120).copy()
    if len(recent) < 10:
        return current_price * 0.98, current_price * 1.03
    lows = recent["Low"].astype(float).to_numpy()
    highs = recent["High"].astype(float).to_numpy()
    support_candidates: List[float] = []
    resistance_candidates: List[float] = []
    window = 3
    for i in range(window, len(recent) - window):
        low_slice = lows[i - window : i + window + 1]
        high_slice = highs[i - window : i + window + 1]
        if lows[i] <= np.nanmin(low_slice) and lows[i] < current_price:
            support_candidates.append(float(lows[i]))
        if highs[i] >= np.nanmax(high_slice) and highs[i] > current_price:
            resistance_candidates.append(float(highs[i]))
    if not support_candidates:
        below = recent[recent["Low"] < current_price]["Low"].astype(float)
        support = float(below.max()) if not below.empty else current_price * 0.98
    else:
        support = max(support_candidates)
    if not resistance_candidates:
        above = recent[recent["High"] > current_price]["High"].astype(float)
        resistance = float(above.min()) if not above.empty else current_price * 1.03
    else:
        resistance = min(resistance_candidates)
    if support >= current_price:
        support = current_price * 0.98
    if resistance <= current_price:
        resistance = current_price * 1.03
    return support, resistance


@dataclass
class StrategyResult:
    symbol: str
    market: str
    current_price: float
    market_mode: str
    decision: str
    buy_low: float
    buy_high: float
    chase_limit: float
    stop_price: float
    target1: float
    target2: float
    target3: float
    risk_reward: float
    basis: Dict


def detect_market_mode(last: pd.Series) -> str:
    price = safe_float(last.get("Close"))
    ma5 = safe_float(last.get("MA5"), price)
    ma20 = safe_float(last.get("MA20"), price)
    ma60 = safe_float(last.get("MA60"), price)
    rsi = safe_float(last.get("RSI14"), 50)

    if price > ma20 > ma60 and ma5 > ma20 and rsi < 72:
        return "상승장/상승추세"
    if price < ma20 and ma5 < ma20:
        return "하락장/약세추세"
    if rsi >= 72:
        return "급등장/과열"
    return "횡보장/중립"


def make_strategy(df: pd.DataFrame, symbol: str, market: str, manual_current: Optional[float] = None) -> StrategyResult:
    data = add_indicators(df)
    last = data.iloc[-1].copy()
    if manual_current and manual_current > 0:
        last["Close"] = manual_current
    price = safe_float(last["Close"])
    atr = safe_float(last.get("ATR14"), price * 0.02)
    if atr <= 0:
        atr = price * 0.02
    rsi = safe_float(last.get("RSI14"), 50)
    vol_ratio = safe_float(last.get("VOL_RATIO"), 1)
    ma5 = safe_float(last.get("MA5"), price)
    ma20 = safe_float(last.get("MA20"), price)
    ma60 = safe_float(last.get("MA60"), price)

    support, resistance = local_levels(data, price)
    support = round_price(support, market)
    resistance = round_price(resistance, market)

    market_mode = detect_market_mode(last)
    trend_down = price < ma20 and ma5 < ma20
    overheat = rsi >= 72 and price > ma20 * 1.04
    breakout = price > resistance and vol_ratio >= 1.3 and rsi < 75

    if breakout:
        buy_low = resistance
        buy_high = resistance + atr * 0.25
        stop = resistance - atr * 0.55
        basis_type = "거래량 동반 돌파"
    else:
        buy_low = support
        buy_high = support + atr * 0.55
        stop = support - atr * 0.65
        basis_type = "지지 확인 눌림"

    buy_low = round_price(buy_low, market)
    buy_high = round_price(max(buy_high, buy_low * 1.001), market)
    stop = round_price(min(stop, buy_low * 0.995), market)
    entry_mid = (buy_low + buy_high) / 2
    risk_per_share = max(entry_mid - stop, entry_mid * 0.005)
    target1 = round_price(max(resistance, entry_mid + risk_per_share * 1.0), market)
    target2 = round_price(entry_mid + risk_per_share * 2.0, market)
    target3 = round_price(entry_mid + risk_per_share * 3.0, market)
    chase_limit = round_price(buy_high + atr * 0.45, market)
    risk_reward = (target2 - entry_mid) / risk_per_share if risk_per_share > 0 else 0

    in_buy_zone = buy_low <= price <= buy_high
    below_stop = price <= stop
    over_chase = price >= chase_limit and not breakout

    if below_stop:
        decision = "매수 금지"
        reason = "손절/무효화 기준 아래에 있음"
    elif overheat or over_chase:
        decision = "매수 금지"
        reason = "과열 또는 추격 금지 구간"
    elif trend_down and rsi < 45:
        decision = "매수 금지"
        reason = "단기 약세 추세"
    elif in_buy_zone and risk_reward >= 1.5 and not trend_down:
        decision = "매수 가능"
        reason = "매수 구간 안 + 손익비 기준 충족"
    elif breakout and risk_reward >= 1.3:
        decision = "돌파 매수 가능"
        reason = "저항 돌파 + 거래량 기준 충족"
    else:
        decision = "대기"
        reason = "가격은 관찰 구간이나 확인 조건이 부족함"

    basis = {
        "판단근거": reason,
        "전략유형": basis_type,
        "현재가": price,
        "지지선": support,
        "저항선": resistance,
        "ATR14": atr,
        "RSI14": rsi,
        "거래량비율": vol_ratio,
        "MA5": ma5,
        "MA20": ma20,
        "MA60": ma60,
        "전략변경조건": [
            "손절가 이탈",
            "추격 금지선 이상 급등",
            "저항선 거래량 동반 돌파",
            "시장 전체 급락 또는 종목 중요 뉴스",
        ],
    }

    return StrategyResult(
        symbol=symbol,
        market=market,
        current_price=price,
        market_mode=market_mode,
        decision=decision,
        buy_low=buy_low,
        buy_high=buy_high,
        chase_limit=chase_limit,
        stop_price=stop,
        target1=target1,
        target2=target2,
        target3=target3,
        risk_reward=risk_reward,
        basis=basis,
    )


def strategy_to_dict(s: StrategyResult) -> Dict:
    return {
        "symbol": s.symbol,
        "market": s.market,
        "current_price": s.current_price,
        "market_mode": s.market_mode,
        "decision": s.decision,
        "buy_low": s.buy_low,
        "buy_high": s.buy_high,
        "chase_limit": s.chase_limit,
        "stop_price": s.stop_price,
        "target1": s.target1,
        "target2": s.target2,
        "target3": s.target3,
        "risk_reward": s.risk_reward,
        "basis": s.basis,
    }


def compare_with_old_strategy(old: sqlite3.Row, current_price: float, market: str) -> Tuple[str, str]:
    stop = safe_float(old["stop_price"])
    buy_low = safe_float(old["buy_low"])
    buy_high = safe_float(old["buy_high"])
    chase = safe_float(old["chase_limit"])
    target1 = safe_float(old["target1"])
    target2 = safe_float(old["target2"])
    target3 = safe_float(old["target3"])

    if current_price <= stop:
        return "전략 무효화", f"현재가 {fmt_price(current_price, market)}가 기존 손절가 {fmt_price(stop, market)} 이하입니다. 기존 매수 전략은 폐기 대상입니다."
    if current_price >= target3:
        return "최종 매도 후보", f"현재가가 기존 최종 목표가 {fmt_price(target3, market)} 이상입니다. 잔여 물량 정리 후보입니다."
    if current_price >= target2:
        return "2차 매도 후보", f"현재가가 기존 2차 목표가 {fmt_price(target2, market)} 이상입니다. 2차 분할매도 후보입니다."
    if current_price >= target1:
        return "1차 매도 후보", f"현재가가 기존 1차 목표가 {fmt_price(target1, market)} 이상입니다. 1차 분할매도 후보입니다."
    if buy_low <= current_price <= buy_high:
        return "전략 유지", f"현재가가 기존 매수 구간 {fmt_price(buy_low, market)}~{fmt_price(buy_high, market)} 안에 있습니다."
    if current_price >= chase:
        return "추격 금지", f"현재가가 기존 추격 금지선 {fmt_price(chase, market)} 이상입니다. 근거 없는 상향 조정은 금지합니다."
    return "전략 유지", "전략 변경 조건이 아직 발생하지 않았습니다. 기존 전략을 우선 유지합니다."


def score_strategy(strategy: StrategyResult, old_status: str = "") -> float:
    base_map = {
        "매수 가능": 88,
        "돌파 매수 가능": 82,
        "대기": 58,
        "매수 금지": 18,
    }
    score = float(base_map.get(strategy.decision, 50))
    rr_bonus = min(max(strategy.risk_reward, 0.0), 3.0) / 3.0 * 8.0
    vol_bonus = min(max(safe_float(strategy.basis.get("거래량비율"), 1.0) - 1.0, 0.0), 1.0) * 4.0

    price = strategy.current_price
    buy_low = strategy.buy_low
    buy_high = strategy.buy_high
    if buy_low <= price <= buy_high:
        proximity_bonus = 6.0
    else:
        mid = (buy_low + buy_high) / 2
        distance_pct = abs(price - mid) / mid if mid > 0 else 1
        proximity_bonus = -min(18.0, distance_pct * 250.0)

    old_penalty = 0.0
    if old_status in ["추격 금지", "전략 무효화"]:
        old_penalty = -18.0
    elif old_status in ["1차 매도 후보", "2차 매도 후보", "최종 매도 후보"]:
        old_penalty = -8.0

    return round(max(0.0, min(100.0, score + rr_bonus + vol_bonus + proximity_bonus + old_penalty)), 1)


# -----------------------------
# OCR / 캡쳐 자동 입력
# -----------------------------

@st.cache_data(show_spinner=False, ttl=600)
def ocr_image_bytes(image_bytes: bytes) -> str:
    if Image is None:
        return ""
    if pytesseract is None:
        return ""

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return ""

    variants = []
    variants.append(img)
    try:
        gray = ImageOps.grayscale(img)
        w, h = gray.size
        scale = 2 if max(w, h) < 1800 else 1
        if scale > 1:
            gray = gray.resize((w * scale, h * scale))
        sharp = gray.filter(ImageFilter.SHARPEN)
        variants.append(sharp)
        threshold = sharp.point(lambda p: 255 if p > 165 else 0)
        variants.append(threshold)
    except Exception:
        pass

    texts: List[str] = []
    for variant in variants:
        for lang in ["kor+eng", "eng"]:
            try:
                text = pytesseract.image_to_string(variant, lang=lang, config="--oem 3 --psm 6")
                if text and text.strip():
                    texts.append(text.strip())
            except Exception:
                continue
    # 중복 라인 제거
    seen = set()
    lines: List[str] = []
    for text in texts:
        for line in text.splitlines():
            clean = line.strip()
            if clean and clean not in seen:
                seen.add(clean)
                lines.append(clean)
    return "\n".join(lines)


def extract_numbers_from_text(text: str) -> List[float]:
    if not text:
        return []
    # 0.702803주 같은 소수 수량을 702803 종목코드로 오인하지 않도록 소수/콤마 숫자를 통째로 읽는다.
    pattern = r"(?<![\d.])(?:\$\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{4,})(?:\s*원|\s*주|\s*USD|\s*달러)?"
    values: List[float] = []
    for match in re.finditer(pattern, text):
        val = clean_number_text(match.group(0))
        if val is not None and val > 0:
            values.append(float(val))
    out: List[float] = []
    seen = set()
    for v in values:
        key = round(v, 6)
        if key not in seen:
            out.append(v)
            seen.add(key)
    return out


def extract_dollar_numbers(text: str) -> List[float]:
    """미장 캡쳐에서 $182.42, $127.95 같은 달러 숫자만 분리."""
    if not text:
        return []
    pattern = r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    values: List[float] = []
    for match in re.finditer(pattern, text):
        val = clean_number_text(match.group(1))
        if val is not None and val > 0:
            values.append(float(val))
    out: List[float] = []
    seen = set()
    for v in values:
        key = round(v, 6)
        if key not in seen:
            out.append(v)
            seen.add(key)
    return out


def extract_krw_numbers(text: str) -> List[float]:
    """원/₩ 표기가 붙은 숫자만 분리. 보유현금 후보를 엉뚱한 평가금액으로 잡지 않기 위한 보조 함수."""
    if not text:
        return []
    pattern = r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?)\s*원|₩\s*(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?)"
    values: List[float] = []
    for match in re.finditer(pattern, text):
        raw = match.group(1) or match.group(2)
        val = clean_number_text(raw)
        if val is not None and val > 0:
            values.append(float(val))
    out: List[float] = []
    seen = set()
    for v in values:
        key = round(v, 6)
        if key not in seen:
            out.append(v)
            seen.add(key)
    return out


def find_label_value(text: str, labels: Iterable[str]) -> Optional[float]:
    if not text:
        return None
    label_re = "|".join(re.escape(label) for label in labels)
    num_re = r"(?:\$\s*)?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{4,}(?:\.\d+)?|\d+\.\d+)"
    pattern = rf"(?:{label_re})[^\d$]{{0,35}}{num_re}"
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if m:
        return clean_number_text(m.group(1))
    return None


def parse_ocr_text(text: str) -> Dict:
    cleaned = text or ""
    # 0.702803주에서 702803을 종목코드로 잡는 문제 방지: 앞뒤에 숫자/소수점이 없어야 6자리 코드로 인정.
    six_digit_codes = re.findall(r"(?<![\d.])(\d{6})(?![\d.])", cleaned)

    stopwords = {
        "KRW", "USD", "ETF", "ETN", "PER", "PBR", "ROE", "EPS",
        "OCR", "AI", "APP", "BUY", "SELL", "HOLD", "KRX", "NYSE", "NASDAQ"
    }
    ticker_candidates = re.findall(r"\b[A-Z]{1,6}\b", cleaned)
    ticker_candidates = [t for t in ticker_candidates if t not in stopwords]
    # 한 글자 티커는 OCR 잡음이 많아서 뒤로 보낸다. 단, 직접 수정 가능하게 후보에는 남긴다.
    ticker_candidates = sorted(dict.fromkeys(ticker_candidates), key=lambda t: (len(t) == 1, cleaned.find(t)))

    has_dollar = "$" in cleaned or "달러" in cleaned or "프리마켓" in cleaned or "애프터" in cleaned
    dollar_numbers = extract_dollar_numbers(cleaned)
    krw_numbers = extract_krw_numbers(cleaned)

    cash = find_label_value(cleaned, ["보유현금", "예수금", "주문가능", "출금가능", "매수가능", "현금"])
    current_price = find_label_value(cleaned, ["현재가", "체결가", "주문가", "매수가", "매도가", "평균단가", "단가"])
    qty = find_label_value(cleaned, ["보유 수량", "보유수량", "수량", "가능수량", "주식수"])
    total = find_label_value(cleaned, ["거래 총액", "주문금액", "매수금액", "매도금액", "체결금액"])

    numbers = extract_numbers_from_text(cleaned)
    code_set = {float(code) for code in six_digit_codes}

    # 시장 판정: 달러 표기가 있으면 6자리 숫자가 있어도 미장으로 우선 처리.
    if has_dollar or (ticker_candidates and not six_digit_codes):
        market = "US"
    elif six_digit_codes:
        market = "KRX"
    else:
        market = "KRX"

    if market == "US":
        # 첫 번째 $ 가격은 보통 현재가다. 평가금액/투자원금도 후보로 두되 기본값은 첫 달러 가격.
        price_candidates = [v for v in dollar_numbers if 0.01 <= v <= 10_000]
        if current_price is None and price_candidates:
            current_price = price_candidates[0]
        # 미장 보유화면의 $127.95, $130.14는 현금이 아니라 평가금/원금일 가능성이 커서 현금 후보로 자동 반영하지 않는다.
        cash_candidates = [cash] if cash is not None and cash >= 1_000 else []
        symbol = ticker_candidates[0] if ticker_candidates else ""
    else:
        price_candidates = []
        for v in numbers:
            if v in code_set:
                continue
            if 100 <= v <= 2_000_000:
                price_candidates.append(v)
        if current_price is None and price_candidates:
            current_price = price_candidates[0]
        # 현금은 라벨이 확실할 때만 기본 반영. 라벨 없는 큰 숫자는 평가금액일 수 있어 자동 현금 후보에서 제외.
        cash_candidates = [cash] if cash is not None and cash >= 1_000 else []
        symbol = six_digit_codes[0] if six_digit_codes else ""

    return {
        "market": market,
        "symbol": symbol,
        "current_price": current_price or 0.0,
        "cash": cash or 0.0,
        "qty": qty or 0.0,
        "total": total or 0.0,
        "codes": six_digit_codes,
        "tickers": ticker_candidates,
        "numbers": numbers,
        "price_candidates": price_candidates,
        "cash_candidates": cash_candidates,
    }


# -----------------------------
# 화면 공통
# -----------------------------

def render_header() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title("📈 주식 전략 판단기")
    st.caption("한 번에 매수 · 추가매수 없음 · 분할매도 · 전략 저장으로 일관성 유지 · 관심종목 스캔 · 캡쳐 자동입력")


def apply_pending_analysis_defaults() -> None:
    pending = st.session_state.pop("pending_analysis", None)
    if not pending:
        return
    st.session_state["analysis_market_widget"] = pending.get("market", "KRX")
    st.session_state["analysis_raw_symbol"] = pending.get("symbol", "")
    st.session_state["analysis_manual_current"] = float(pending.get("current_price", 0.0) or 0.0)
    st.session_state["analysis_interval_widget"] = pending.get("interval", "5m")
    st.toast("캡쳐에서 읽은 값을 전략 분석 입력칸에 넣었습니다.")


# -----------------------------
# 현금/포트 탭
# -----------------------------

def render_cash_tab() -> None:
    st.subheader("💰 보유현금 관리")
    balance = cash_balance_krw()
    st.metric("현재 앱 기준 보유현금", fmt_krw(balance))

    with st.expander("현금 입금/출금/보정 입력", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            event_type = st.selectbox("구분", ["입금", "출금", "현금보정+", "현금보정-"])
            amount = st.number_input("금액(원)", min_value=0.0, value=0.0, step=10_000.0, format="%.0f")
        with col2:
            memo = st.text_input("메모", placeholder="예: 토스증권 현금 맞춤")
            if st.button("현금 기록 추가", use_container_width=True):
                if amount <= 0:
                    st.warning("금액을 입력하세요.")
                else:
                    add_cash_event(event_type, amount, memo)
                    st.success("현금 기록을 추가했습니다.")
                    st.rerun()

    with st.expander("현재 현금으로 바로 맞추기"):
        target_cash = st.number_input("증권앱에 보이는 실제 보유현금(원)", min_value=0.0, value=max(balance, 0.0), step=10_000.0, format="%.0f")
        diff = target_cash - balance
        st.write(f"보정 차이: **{fmt_krw(diff)}**")
        if st.button("이 금액으로 현금 맞추기", use_container_width=True):
            if abs(diff) < 1:
                st.info("이미 동일합니다.")
            elif diff > 0:
                add_cash_event("현금보정+", diff, "현재 현금 직접 맞춤")
                st.success("현금을 보정했습니다.")
                st.rerun()
            else:
                add_cash_event("현금보정-", abs(diff), "현재 현금 직접 맞춤")
                st.success("현금을 보정했습니다.")
                st.rerun()

    st.divider()
    st.subheader("거래 기록")
    with st.form("trade_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            market = st.selectbox("시장", ["KRX", "US"])
            symbol = st.text_input("종목코드", placeholder="KRX: 005930 / US: NVDA")
        with c2:
            side = st.selectbox("매수/매도", ["매수", "매도"])
            qty = st.number_input("수량(주)", min_value=0.0, value=0.0, step=0.000001, format="%.6f")
        with c3:
            price = st.number_input("체결가(원 또는 달러)", min_value=0.0, value=0.0, step=1.0)
            total_krw = st.number_input("현금 반영 총액(원)", min_value=0.0, value=0.0, step=1_000.0, format="%.0f")
        memo = st.text_input("거래 메모", placeholder="예: 수수료/세금 포함 총액")
        submitted = st.form_submit_button("거래 기록 추가", use_container_width=True)
        if submitted:
            clean_symbol = display_symbol(symbol, market)
            if not clean_symbol or qty <= 0 or total_krw <= 0:
                st.warning("종목코드, 수량, 현금 반영 총액을 입력하세요.")
            else:
                add_trade(clean_symbol, market, side, qty, price, total_krw, memo)
                st.success("거래 기록을 추가했습니다.")
                st.rerun()

    positions = calculate_positions()
    if not positions.empty:
        show = positions.copy()
        show["qty"] = show["qty"].map(lambda x: f"{x:,.6f}".rstrip("0").rstrip("."))
        show["cost_krw"] = show["cost_krw"].map(fmt_krw)
        show["avg_cost_krw_per_share"] = show["avg_cost_krw_per_share"].map(fmt_krw)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("아직 보유 종목 기록이 없습니다.")

    with st.expander("기록 원본 보기"):
        st.write("현금 기록")
        st.dataframe(load_table("cash_events"), use_container_width=True, hide_index=True)
        st.write("거래 기록")
        st.dataframe(load_table("trades"), use_container_width=True, hide_index=True)


# -----------------------------
# 전략 분석 탭
# -----------------------------

def budget_from_cash(cash: float) -> Tuple[float, str]:
    if cash >= MAX_BUDGET_KRW:
        return DEFAULT_BUDGET_KRW, "기본 매수금액 105만원을 사용합니다."
    if MIN_BUDGET_KRW <= cash < MAX_BUDGET_KRW:
        return cash, "보유현금이 100~110만원 사이이므로 현재 현금 한도 안에서 계산합니다."
    if cash > 0:
        return cash, "보유현금이 100만원 미만입니다. 신규 매수는 보수적으로 보거나 대기 권장입니다."
    return 0.0, "앱 기준 보유현금이 없습니다. 현금 탭에서 먼저 맞춰 주세요."


def split_qty(qty: float, market: str) -> Tuple[str, str, str]:
    if market == "KRX":
        total = int(math.floor(qty))
        q1 = int(round(total * 0.4))
        q2 = int(round(total * 0.4))
        q3 = max(0, total - q1 - q2)
        return f"{q1}주", f"{q2}주", f"{q3}주"
    q1 = math.floor(qty * 0.4 * 1_000_000) / 1_000_000
    q2 = math.floor(qty * 0.4 * 1_000_000) / 1_000_000
    q3 = max(0.0, qty - q1 - q2)
    return f"{q1:.6f}주", f"{q2:.6f}주", f"{q3:.6f}주"


def render_sizing(strategy: StrategyResult, cash: float) -> None:
    st.subheader("🧮 매수금액/수량 계산")
    base_budget, budget_note = budget_from_cash(cash)
    st.caption(budget_note)

    if strategy.market == "US":
        fx = st.number_input("환율(원/USD)", min_value=1.0, value=1400.0, step=1.0)
        fractional = st.checkbox("미장 소수점 수량 허용", value=True)
        price_krw_per_share = strategy.current_price * fx
        if fractional:
            qty = base_budget / price_krw_per_share if price_krw_per_share > 0 else 0
            qty = math.floor(qty * 1_000_000) / 1_000_000
        else:
            qty = math.floor(base_budget / price_krw_per_share) if price_krw_per_share > 0 else 0
        used = qty * price_krw_per_share
        remain = cash - used
        st.write(f"현재가 환산 = {fmt_price(strategy.current_price, 'US')}/주 × {fx:,.0f}원/USD = **{fmt_krw(price_krw_per_share)}/주**")
    else:
        qty = math.floor(base_budget / strategy.current_price) if strategy.current_price > 0 else 0
        used = qty * strategy.current_price
        remain = cash - used

    col1, col2, col3 = st.columns(3)
    col1.metric("추천 매수 예산", fmt_krw(base_budget))
    col2.metric("계산 수량", f"{qty:,.6f}주".rstrip("0").rstrip("."))
    col3.metric("예상 사용금액", fmt_krw(used))
    st.write(f"남는 현금 = {fmt_krw(cash)} - {fmt_krw(used)} = **{fmt_krw(remain)}**")

    st.write("분할매도 기본 수량")
    q1, q2, q3 = split_qty(qty, strategy.market)
    st.table(
        pd.DataFrame(
            [
                {"구간": "1차 매도", "비중": "40%", "수량": q1, "가격": fmt_price(strategy.target1, strategy.market)},
                {"구간": "2차 매도", "비중": "40%", "수량": q2, "가격": fmt_price(strategy.target2, strategy.market)},
                {"구간": "최종 매도", "비중": "20%", "수량": q3, "가격": fmt_price(strategy.target3, strategy.market)},
            ]
        )
    )


def render_strategy_tab() -> None:
    apply_pending_analysis_defaults()
    st.subheader("🎯 종목 전략 분석")
    cash = cash_balance_krw()
    st.metric("현재 앱 기준 보유현금", fmt_krw(cash))

    if "analysis_market_widget" not in st.session_state:
        st.session_state["analysis_market_widget"] = "KRX"
    if "analysis_interval_widget" not in st.session_state:
        st.session_state["analysis_interval_widget"] = "5m"
    if "analysis_raw_symbol" not in st.session_state:
        st.session_state["analysis_raw_symbol"] = ""
    if "analysis_manual_current" not in st.session_state:
        st.session_state["analysis_manual_current"] = 0.0

    with st.form("analyze_form"):
        c1, c2 = st.columns(2)
        with c1:
            market = st.selectbox("시장", ["KRX", "US"], key="analysis_market_widget")
            raw_symbol = st.text_input("종목코드", placeholder="KRX: 005930 / US: RKLB, NVDA", key="analysis_raw_symbol")
        with c2:
            interval = st.selectbox("차트 간격", ["5m", "15m", "1d"], index=["5m", "15m", "1d"].index(st.session_state.get("analysis_interval_widget", "5m")), key="analysis_interval_widget")
            period = "5d" if interval in ["5m", "15m"] else "6mo"
            manual_current = st.number_input("실시간 현재가 직접 입력(선택)", min_value=0.0, value=float(st.session_state.get("analysis_manual_current", 0.0)), step=1.0, key="analysis_manual_current")
        analyze = st.form_submit_button("분석하기", use_container_width=True)

    if not analyze:
        st.info("종목을 입력하고 분석하기를 누르세요. 실시간 현재가가 중요하면 증권앱 가격을 직접 입력하세요. 캡쳐 자동입력 탭에서 스크린샷 값도 보낼 수 있습니다.")
        return

    symbol = display_symbol(raw_symbol, market)
    yf_symbol = normalize_symbol(raw_symbol, market)
    if not symbol:
        st.warning("종목코드를 입력하세요.")
        return

    with st.spinner("차트 데이터를 불러오는 중..."):
        df = fetch_ohlcv(yf_symbol, interval=interval, period=period)

    if df.empty:
        st.error("시세 데이터를 불러오지 못했습니다. 종목코드, 시장, 차트 간격을 확인하세요. KRX 종목은 005930처럼 6자리로 입력해 보세요.")
        return

    strategy = make_strategy(df, symbol=symbol, market=market, manual_current=manual_current if manual_current > 0 else None)
    old = get_active_strategy(symbol, market)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", fmt_price(strategy.current_price, market))
    c2.metric("판단", strategy.decision)
    c3.metric("시장 모드", strategy.market_mode)
    c4.metric("손익비", f"{strategy.risk_reward:.2f}배")

    if old:
        status, message = compare_with_old_strategy(old, strategy.current_price, market)
        st.warning(f"기존 저장 전략: **{status}**\n\n{message}")
        st.caption("기존 전략이 살아 있으면 새 가격으로 근거 없이 매수가를 올리지 않습니다.")

    st.subheader("📌 이번 계산 전략")
    rows = [
        {"항목": "매수 구간", "값": f"{fmt_price(strategy.buy_low, market)} ~ {fmt_price(strategy.buy_high, market)}"},
        {"항목": "추격 금지", "값": fmt_price(strategy.chase_limit, market)},
        {"항목": "손절/무효화", "값": fmt_price(strategy.stop_price, market)},
        {"항목": "1차 매도", "값": fmt_price(strategy.target1, market)},
        {"항목": "2차 매도", "값": fmt_price(strategy.target2, market)},
        {"항목": "최종 매도", "값": fmt_price(strategy.target3, market)},
    ]
    st.table(pd.DataFrame(rows))

    basis = strategy.basis
    with st.expander("판단 근거 보기", expanded=True):
        st.write(f"- 전략유형: **{basis['전략유형']}**")
        st.write(f"- 판단근거: **{basis['판단근거']}**")
        st.write(f"- 지지선: **{fmt_price(basis['지지선'], market)}**")
        st.write(f"- 저항선: **{fmt_price(basis['저항선'], market)}**")
        st.write(f"- RSI14: **{fmt_num(basis['RSI14'], 2)}**")
        st.write(f"- 거래량비율: **{fmt_num(basis['거래량비율'], 2)}배**")
        st.write(f"- MA5/MA20/MA60: **{fmt_price(basis['MA5'], market)} / {fmt_price(basis['MA20'], market)} / {fmt_price(basis['MA60'], market)}**")
        st.write("- 전략변경조건: " + ", ".join(basis["전략변경조건"]))

    render_sizing(strategy, cash)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("이번 전략 저장/고정", use_container_width=True):
            save_strategy(strategy_to_dict(strategy))
            st.success("전략을 저장했습니다. 다음 분석부터 이 전략을 기준으로 일관성 체크합니다.")
            st.rerun()
    with c2:
        if st.button("데이터 새로고침", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.line_chart(df.set_index(df.columns[0])["Close"].tail(120), height=240)


# -----------------------------
# 관심종목 스캔 탭
# -----------------------------

def scan_symbols(symbols: List[Tuple[str, str]], interval: str) -> Tuple[pd.DataFrame, List[Dict], List[str]]:
    period = "5d" if interval in ["5m", "15m"] else "6mo"
    rows: List[Dict] = []
    strategy_dicts: List[Dict] = []
    errors: List[str] = []

    progress = st.progress(0)
    for idx, (symbol, market) in enumerate(symbols, start=1):
        try:
            yf_symbol = normalize_symbol(symbol, market)
            df = fetch_ohlcv(yf_symbol, interval=interval, period=period)
            if df.empty:
                errors.append(f"{market}:{symbol} - 데이터 없음")
                continue
            strategy = make_strategy(df, symbol=symbol, market=market)
            old = get_active_strategy(symbol, market)
            old_status = ""
            old_message = ""
            if old:
                old_status, old_message = compare_with_old_strategy(old, strategy.current_price, market)
            score = score_strategy(strategy, old_status)
            rows.append(
                {
                    "점수": score,
                    "시장": market,
                    "종목": symbol,
                    "현재가": fmt_price(strategy.current_price, market),
                    "판단": strategy.decision,
                    "기존전략": old_status or "없음",
                    "매수구간": f"{fmt_price(strategy.buy_low, market)} ~ {fmt_price(strategy.buy_high, market)}",
                    "추격금지": fmt_price(strategy.chase_limit, market),
                    "손절": fmt_price(strategy.stop_price, market),
                    "1차매도": fmt_price(strategy.target1, market),
                    "2차매도": fmt_price(strategy.target2, market),
                    "최종매도": fmt_price(strategy.target3, market),
                    "손익비": f"{strategy.risk_reward:.2f}배",
                    "시장모드": strategy.market_mode,
                    "근거": strategy.basis.get("판단근거", ""),
                    "기존전략메시지": old_message,
                }
            )
            d = strategy_to_dict(strategy)
            d["score"] = score
            d["old_status"] = old_status
            d["old_message"] = old_message
            strategy_dicts.append(d)
        except Exception as exc:
            errors.append(f"{market}:{symbol} - {exc}")
        finally:
            progress.progress(idx / max(len(symbols), 1))
    progress.empty()

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["점수", "판단"], ascending=[False, True]).reset_index(drop=True)
        strategy_dicts = sorted(strategy_dicts, key=lambda x: x.get("score", 0), reverse=True)
    return result_df, strategy_dicts, errors


def render_watchlist_scan_tab() -> None:
    st.subheader("🔎 관심종목 여러 개 스캔")
    st.caption("여러 종목을 같은 규칙으로 한 번에 돌려서 매수 가능/대기/매수 금지를 비교합니다.")

    with st.expander("관심종목 저장/관리", expanded=False):
        with st.form("watchlist_add_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                market = st.selectbox("시장", ["KRX", "US"], key="watch_add_market")
                symbol = st.text_input("종목코드", placeholder="005930 또는 NVDA", key="watch_add_symbol")
            with c2:
                name = st.text_input("종목명/별명", placeholder="선택")
                memo = st.text_input("메모", placeholder="예: 전력기기, AI, 방산")
            with c3:
                st.write("")
                submitted = st.form_submit_button("관심종목 추가/수정", use_container_width=True)
            if submitted:
                clean_symbol = display_symbol(symbol, market)
                if not clean_symbol:
                    st.warning("종목코드를 입력하세요.")
                else:
                    add_watchlist_item(clean_symbol, market, name, memo)
                    st.success("관심종목에 저장했습니다.")
                    st.rerun()

        watch_df = load_watchlist()
        if watch_df.empty:
            st.info("저장된 관심종목이 없습니다.")
        else:
            st.dataframe(watch_df[["id", "market", "symbol", "name", "memo"]], use_container_width=True, hide_index=True)
            delete_id = st.number_input("삭제할 관심종목 ID", min_value=0, value=0, step=1)
            if st.button("선택 ID 삭제", use_container_width=True):
                if delete_id > 0:
                    deactivate_watchlist_item(delete_id)
                    st.success("삭제했습니다.")
                    st.rerun()

    watch_df = load_watchlist()
    saved_text = "\n".join([f"{r['market']}:{r['symbol']}" for _, r in watch_df.iterrows()]) if not watch_df.empty else ""

    c1, c2 = st.columns([2, 1])
    with c1:
        use_saved = st.checkbox("저장된 관심종목 사용", value=bool(saved_text))
        raw_text = st.text_area(
            "스캔할 종목 목록",
            value=saved_text if use_saved else "",
            height=160,
            placeholder="예:\nKRX:005930\nKRX:267260\nUS:NVDA\nUS:RKLB",
        )
    with c2:
        default_market = st.selectbox("기본 시장", ["KRX", "US"])
        auto_detect = st.checkbox("6자리 숫자=국장, 영문=미장 자동감지", value=True)
        interval = st.selectbox("스캔 차트 간격", ["5m", "15m", "1d"], index=0)
        only_candidates = st.checkbox("매수 가능/돌파 가능만 보기", value=False)
        max_symbols = st.slider("최대 스캔 개수", min_value=3, max_value=50, value=20, step=1)

    symbols = parse_symbol_lines(raw_text, default_market=default_market, auto_detect=auto_detect)[:max_symbols]
    st.write(f"스캔 대상: **{len(symbols)}개**")

    if st.button("관심종목 스캔 시작", use_container_width=True):
        if not symbols:
            st.warning("스캔할 종목을 입력하세요.")
            return
        with st.spinner("관심종목을 같은 규칙으로 분석 중..."):
            result_df, strategy_dicts, errors = scan_symbols(symbols, interval)
        if only_candidates and not result_df.empty:
            result_df = result_df[result_df["판단"].isin(["매수 가능", "돌파 매수 가능"])].reset_index(drop=True)
            strategy_dicts = [s for s in strategy_dicts if s["decision"] in ["매수 가능", "돌파 매수 가능"]]
        st.session_state["last_scan_df"] = result_df
        st.session_state["last_scan_strategies"] = strategy_dicts
        st.session_state["last_scan_errors"] = errors
        st.rerun()

    result_df = st.session_state.get("last_scan_df")
    strategy_dicts = st.session_state.get("last_scan_strategies", [])
    errors = st.session_state.get("last_scan_errors", [])

    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        st.divider()
        st.subheader("스캔 결과")
        st.dataframe(result_df.drop(columns=["기존전략메시지"], errors="ignore"), use_container_width=True, hide_index=True)
        csv = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("스캔 결과 CSV 다운로드", csv, file_name="watchlist_scan.csv", mime="text/csv", use_container_width=True)

        top_items = strategy_dicts[:5]
        with st.expander("상위 후보 상세/전략 저장", expanded=True):
            for idx, item in enumerate(top_items, start=1):
                st.markdown(f"### {idx}. {item['market']}:{item['symbol']} · 점수 {item.get('score', 0)}")
                cols = st.columns(4)
                cols[0].metric("현재가", fmt_price(item["current_price"], item["market"]))
                cols[1].metric("판단", item["decision"])
                cols[2].metric("손익비", f"{item['risk_reward']:.2f}배")
                cols[3].metric("기존전략", item.get("old_status") or "없음")
                st.write(f"매수구간: **{fmt_price(item['buy_low'], item['market'])} ~ {fmt_price(item['buy_high'], item['market'])}**")
                st.write(f"손절: **{fmt_price(item['stop_price'], item['market'])}** / 1차: **{fmt_price(item['target1'], item['market'])}** / 2차: **{fmt_price(item['target2'], item['market'])}** / 최종: **{fmt_price(item['target3'], item['market'])}**")
                if item.get("old_message"):
                    st.caption(item["old_message"])
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"{item['market']}:{item['symbol']} 전략 저장", key=f"save_scan_{idx}", use_container_width=True):
                        save_strategy(item)
                        st.success(f"{item['symbol']} 전략을 저장했습니다.")
                        st.rerun()
                with c2:
                    if st.button(f"전략 분석 탭으로 보내기", key=f"send_scan_{idx}", use_container_width=True):
                        st.session_state["pending_analysis"] = {
                            "market": item["market"],
                            "symbol": item["symbol"],
                            "current_price": item["current_price"],
                            "interval": interval,
                        }
                        st.success("전략 분석 탭으로 보냈습니다. 상단의 전략 분석 탭을 열어 확인하세요.")
                        st.rerun()
                st.divider()
    elif isinstance(result_df, pd.DataFrame) and result_df.empty:
        st.info("조건에 맞는 스캔 결과가 없습니다.")

    if errors:
        with st.expander("데이터 불러오기 실패 목록"):
            for err in errors:
                st.write(f"- {err}")


# -----------------------------
# 캡쳐 자동입력 탭
# -----------------------------

def render_screenshot_tab() -> None:
    st.subheader("📸 캡쳐 자동입력")
    st.caption("증권앱/네이버/토스 화면 캡쳐를 올리면 OCR로 종목코드, 현재가, 현금, 수량 후보를 읽어옵니다. 숫자는 꼭 확인하세요.")

    uploaded = st.file_uploader("캡쳐 이미지 업로드", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is None:
        st.info("캡쳐 이미지를 업로드하면 자동입력 후보가 표시됩니다.")
        return

    image_bytes = uploaded.getvalue()
    if Image is not None:
        img = Image.open(io.BytesIO(image_bytes))
        st.image(img, caption="업로드한 캡쳐", use_container_width=True)

    if pytesseract is None:
        st.error("OCR 패키지가 설치되지 않았습니다. requirements.txt와 packages.txt를 같이 배포했는지 확인하세요.")
        return

    with st.spinner("캡쳐에서 글자/숫자를 읽는 중..."):
        text = ocr_image_bytes(image_bytes)
    parsed = parse_ocr_text(text)

    if not text.strip():
        st.warning("OCR 텍스트를 읽지 못했습니다. 화면을 더 밝게, 숫자가 크게 보이게 캡쳐하거나 직접 입력을 사용하세요.")
        return

    with st.expander("OCR 원문 보기", expanded=False):
        st.text_area("읽은 텍스트", text, height=180)

    st.subheader("자동입력 후보")
    c1, c2 = st.columns(2)
    with c1:
        market_index = 0 if parsed["market"] == "KRX" else 1
        market = st.selectbox("시장", ["KRX", "US"], index=market_index, key="ocr_market")
        symbol_options = parsed["codes"] if market == "KRX" else parsed["tickers"]
        default_symbol = parsed["symbol"]
        if symbol_options:
            default_symbol = st.selectbox("감지된 종목 후보", symbol_options, index=0)
        symbol = st.text_input("종목코드", value=display_symbol(default_symbol, market), key="ocr_symbol")
    with c2:
        price_candidates = parsed.get("price_candidates", [])[:15]
        price_default = safe_float(parsed.get("current_price"), 0.0)
        if price_candidates:
            labels = [f"{v:,.2f}" if market == "US" else f"{v:,.0f}" for v in price_candidates]
            selected_label = st.selectbox("가격 후보", labels, index=0)
            selected_price = clean_number_text(selected_label) or price_default
        else:
            selected_price = price_default
        price_step = 1.0 if market == "KRX" else 0.01
        current_price = st.number_input("현재가/체결가", min_value=0.0, value=float(selected_price), step=price_step, key="ocr_current_price")

    c3, c4 = st.columns(2)
    with c3:
        cash_candidates = parsed.get("cash_candidates", [])[:15]
        cash_default = safe_float(parsed.get("cash"), 0.0)
        if cash_candidates:
            cash_labels = [f"{v:,.0f}" for v in cash_candidates]
            selected_cash_label = st.selectbox("현금/금액 후보", cash_labels, index=0)
            selected_cash = clean_number_text(selected_cash_label) or cash_default
        else:
            selected_cash = cash_default
        detected_cash = st.number_input("보유현금/예수금 후보(원)", min_value=0.0, value=float(selected_cash), step=10_000.0, format="%.0f")
    with c4:
        qty = st.number_input("수량 후보(주)", min_value=0.0, value=float(safe_float(parsed.get("qty"), 0.0)), step=0.000001, format="%.6f")
        total_krw = st.number_input("거래 총액 후보(원)", min_value=0.0, value=float(safe_float(parsed.get("total"), 0.0)), step=1_000.0, format="%.0f")

    st.write("감지 숫자 후보:", ", ".join([f"{v:,.2f}" if v < 1000 and market == "US" else f"{v:,.0f}" for v in parsed.get("numbers", [])[:30]]))

    st.divider()
    st.subheader("바로 반영")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("전략 분석 탭으로 보내기", use_container_width=True):
            if not symbol or current_price <= 0:
                st.warning("종목코드와 현재가를 확인하세요.")
            else:
                st.session_state["pending_analysis"] = {
                    "market": market,
                    "symbol": display_symbol(symbol, market),
                    "current_price": current_price,
                    "interval": "5m",
                }
                st.success("전략 분석 탭 입력값으로 보냈습니다.")
                st.rerun()
    with c2:
        if st.button("이 현금으로 맞추기", use_container_width=True):
            if detected_cash <= 0:
                st.warning("현금 후보를 확인하세요.")
            else:
                balance = cash_balance_krw()
                diff = detected_cash - balance
                if abs(diff) < 1:
                    st.info("이미 동일합니다.")
                elif diff > 0:
                    add_cash_event("현금보정+", diff, "캡쳐 OCR 현금 맞춤")
                    st.success("현금을 보정했습니다.")
                    st.rerun()
                else:
                    add_cash_event("현금보정-", abs(diff), "캡쳐 OCR 현금 맞춤")
                    st.success("현금을 보정했습니다.")
                    st.rerun()
    with c3:
        side = st.selectbox("거래 기록 구분", ["매수", "매도"], key="ocr_trade_side")
        if st.button("거래 기록 추가", use_container_width=True):
            clean_symbol = display_symbol(symbol, market)
            price = current_price
            if not clean_symbol or qty <= 0 or total_krw <= 0:
                st.warning("종목코드, 수량, 거래 총액을 확인하세요.")
            else:
                add_trade(clean_symbol, market, side, qty, price, total_krw, "캡쳐 OCR에서 추가")
                st.success("거래 기록을 추가했습니다.")
                st.rerun()

    st.info("OCR은 보조 기능입니다. 실제 매수/매도 전에는 증권앱의 종목, 현재가, 수량, 총액을 한 번 더 확인하세요.")


# -----------------------------
# 저장 전략 / 백업 / 가이드
# -----------------------------

def render_strategy_history_tab() -> None:
    st.subheader("🧷 저장 전략")
    df = load_table("strategies")
    if df.empty:
        st.info("저장된 전략이 없습니다.")
        return
    show = df.copy()
    for col in ["baseline_price", "current_price", "buy_low", "buy_high", "chase_limit", "stop_price", "target1", "target2", "target3"]:
        if col in show.columns:
            show[col] = show.apply(lambda r: fmt_price(r[col], r["market"]), axis=1)
    show["risk_reward"] = show["risk_reward"].map(lambda x: f"{safe_float(x):.2f}배")
    st.dataframe(show.drop(columns=["basis_json"], errors="ignore"), use_container_width=True, hide_index=True)


def render_export_tab() -> None:
    st.subheader("📦 백업/복원")
    st.caption("Streamlit Cloud 같은 무료 배포 환경은 저장소가 초기화될 수 있습니다. 거래 기록은 자주 백업하세요.")

    data = {
        "cash_events": load_table("cash_events").to_dict(orient="records"),
        "trades": load_table("trades").to_dict(orient="records"),
        "strategies": load_table("strategies").to_dict(orient="records"),
        "watchlist": load_table("watchlist").to_dict(orient="records"),
        "exported_at": now_text(),
    }
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("백업 JSON 다운로드", json_bytes, file_name="stock_strategy_backup.json", mime="application/json", use_container_width=True)

    uploaded = st.file_uploader("백업 JSON 업로드", type=["json"])
    if uploaded is not None:
        try:
            restored = json.loads(uploaded.read().decode("utf-8"))
            st.success("백업 파일을 읽었습니다. 복원은 안전상 현재 버전에서 수동 처리로 제한합니다.")
            st.json({k: len(v) if isinstance(v, list) else v for k, v in restored.items()})
        except Exception as exc:
            st.error(f"백업 파일을 읽지 못했습니다: {exc}")


def render_guide_tab() -> None:
    st.subheader("⚙️ 사용 원칙")
    st.markdown(
        """
### 이 앱의 고정 규칙

1. **종목 수 제한 없음**  
   국장/미장 개수 제한 없이 관심종목을 저장하고 스캔합니다.

2. **신규 매수 예산은 100만~110만원 기준**  
   앱 기준 보유현금이 110만원 이상이면 기본 105만원으로 계산합니다.  
   앱 기준 보유현금이 100만~110만원이면 가진 현금 안에서 계산합니다.

3. **추가매수 없음**  
   물타기 판단은 하지 않습니다.

4. **매도는 40% / 40% / 20% 분할매도**  
   국장은 정수주로 자동 보정합니다.

5. **전략 저장 후 일관성 유지**  
   기존 매수 구간, 손절가, 추격 금지선이 남아 있으면 새 분석이 나와도 근거 없이 매수가를 올리지 않습니다.

6. **관심종목 스캔**  
   여러 종목을 같은 알고리즘으로 분석하고 점수순으로 보여줍니다.  
   `KRX:005930`, `US:NVDA`처럼 입력하면 시장을 섞어서 스캔할 수 있습니다.

7. **캡쳐 자동입력**  
   스크린샷 OCR로 종목코드, 현재가, 현금, 수량 후보를 읽어옵니다.  
   OCR은 틀릴 수 있으므로 반영 전 숫자를 직접 확인하세요.

8. **무료 시세 데이터 주의**  
   무료 데이터는 지연/누락될 수 있습니다. 실제 진입 전 현재가는 토스/네이버/거래소 앱에서 확인하세요.
        """
    )


def main() -> None:
    init_db()
    render_header()
    tabs = st.tabs(["전략 분석", "관심종목 스캔", "캡쳐 자동입력", "현금/포트", "저장 전략", "백업", "가이드"])
    with tabs[0]:
        render_strategy_tab()
    with tabs[1]:
        render_watchlist_scan_tab()
    with tabs[2]:
        render_screenshot_tab()
    with tabs[3]:
        render_cash_tab()
    with tabs[4]:
        render_strategy_history_tab()
    with tabs[5]:
        render_export_tab()
    with tabs[6]:
        render_guide_tab()


if __name__ == "__main__":
    main()
