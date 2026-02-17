"""
tools_trade.py — Watty Options Trading Tool
=============================================
Live market data + paper trading for SPY options.

Actions:
  market        — Price, volume, VIX, 52w range
  options_chain — Full options chain with Greeks
  paper_buy     — Open a paper position
  paper_sell    — Close a paper position
  portfolio     — View positions and stats
  analyze       — Market analysis + strategy suggestion

MCP tool name: watty_trade
Data: yfinance (free, no API key)

Hugo & Watty · February 2026
"""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

from mcp.types import Tool, TextContent

try:
    import yfinance as yf
except ImportError:
    yf = None

from watty.config import WATTY_HOME

PORTFOLIO_DIR = WATTY_HOME / "trading"
PORTFOLIO_FILE = PORTFOLIO_DIR / "portfolio.json"


def _load_portfolio() -> dict:
    if PORTFOLIO_FILE.exists():
        return json.loads(PORTFOLIO_FILE.read_text())
    return {
        "cash": 50000.0,
        "open_positions": [],
        "closed_positions": [],
        "total_trades": 0,
    }


def _save_portfolio(portfolio: dict):
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    PORTFOLIO_FILE.write_text(json.dumps(portfolio, indent=2, default=str))


# ── Actions ────────────────────────────────────────────────

def action_market(params: dict) -> str:
    if not yf:
        return "yfinance not installed. Run: pip install yfinance"

    ticker = params.get("ticker", "SPY")
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", "?")
        prev = info.get("previousClose", 0)
        change = ((price - prev) / prev * 100) if prev and isinstance(price, (int, float)) else 0
        volume = info.get("regularMarketVolume", "?")
        high52 = info.get("fiftyTwoWeekHigh", "?")
        low52 = info.get("fiftyTwoWeekLow", "?")
        name = info.get("shortName", ticker)

        # Get VIX
        vix_price = "?"
        try:
            vix = yf.Ticker("^VIX")
            vix_info = vix.info
            vix_price = vix_info.get("regularMarketPrice") or vix_info.get("previousClose", "?")
        except Exception:
            pass

        vix_read = ""
        if isinstance(vix_price, (int, float)):
            if vix_price < 15:
                vix_read = " (complacent — sell premium)"
            elif vix_price < 20:
                vix_read = " (normal)"
            elif vix_price < 30:
                vix_read = " (elevated — fear)"
            else:
                vix_read = " (PANIC)"

        lines = [
            f"=== {name} ({ticker}) ===",
            f"Price: ${price}  ({change:+.2f}%)" if isinstance(price, (int, float)) else f"Price: {price}",
            f"Volume: {volume:,}" if isinstance(volume, (int, float)) else f"Volume: {volume}",
            f"52w Range: ${low52} — ${high52}",
            f"VIX: {vix_price}{vix_read}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching {ticker}: {e}"


def action_options_chain(params: dict) -> str:
    if not yf:
        return "yfinance not installed. Run: pip install yfinance"

    ticker = params.get("ticker", "SPY")
    expiry = params.get("expiry")

    try:
        t = yf.Ticker(ticker)
        dates = t.options
        if not dates:
            return f"No options available for {ticker}"

        if expiry and expiry in dates:
            target_date = expiry
        else:
            target_date = dates[0]  # nearest expiry

        chain = t.option_chain(target_date)
        calls = chain.calls
        puts = chain.puts

        # Get current price for context
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", 0)

        lines = [f"=== {ticker} Options — Expires {target_date} (price: ${price}) ==="]
        lines.append(f"Available expiries: {', '.join(dates[:8])}")
        lines.append("")

        # Show calls near the money (5 ITM + 5 OTM)
        lines.append("--- CALLS ---")
        lines.append(f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Vol':>8} {'OI':>8} {'IV':>8}")
        if price and isinstance(price, (int, float)):
            atm = calls.iloc[(calls['strike'] - price).abs().argsort()[:1]].index[0]
            nearby = calls.iloc[max(0, atm - 5):atm + 6]
        else:
            nearby = calls.head(10)

        for _, row in nearby.iterrows():
            strike = row.get("strike", 0)
            bid = row.get("bid", 0)
            ask = row.get("ask", 0)
            vol = row.get("volume", 0)
            oi = row.get("openInterest", 0)
            iv = row.get("impliedVolatility", 0)
            marker = " <-- ATM" if isinstance(price, (int, float)) and abs(strike - price) < 1 else ""
            lines.append(f"${strike:>7.1f} ${bid:>7.2f} ${ask:>7.2f} {vol or 0:>8.0f} {oi or 0:>8.0f} {iv:>7.1%}{marker}")

        lines.append("")
        lines.append("--- PUTS ---")
        lines.append(f"{'Strike':>8} {'Bid':>8} {'Ask':>8} {'Vol':>8} {'OI':>8} {'IV':>8}")
        if price and isinstance(price, (int, float)):
            atm = puts.iloc[(puts['strike'] - price).abs().argsort()[:1]].index[0]
            nearby = puts.iloc[max(0, atm - 5):atm + 6]
        else:
            nearby = puts.head(10)

        for _, row in nearby.iterrows():
            strike = row.get("strike", 0)
            bid = row.get("bid", 0)
            ask = row.get("ask", 0)
            vol = row.get("volume", 0)
            oi = row.get("openInterest", 0)
            iv = row.get("impliedVolatility", 0)
            marker = " <-- ATM" if isinstance(price, (int, float)) and abs(strike - price) < 1 else ""
            lines.append(f"${strike:>7.1f} ${bid:>7.2f} ${ask:>7.2f} {vol or 0:>8.0f} {oi or 0:>8.0f} {iv:>7.1%}{marker}")

        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching options for {ticker}: {e}"


def action_paper_buy(params: dict) -> str:
    ticker = params.get("ticker", "SPY")
    option_type = params.get("option_type", "call").lower()
    strike = params.get("strike")
    expiry = params.get("expiry")
    contracts = params.get("contracts", 1)

    if not strike:
        return "Need a strike price. Example: paper_buy(ticker='SPY', option_type='call', strike=605, expiry='2026-02-21', contracts=1)"

    if option_type not in ("call", "put"):
        return "option_type must be 'call' or 'put'"

    # Get current option price
    try:
        t = yf.Ticker(ticker)
        if not expiry:
            dates = t.options
            expiry = dates[0] if dates else None
        if not expiry:
            return f"No options expiry available for {ticker}"

        chain = t.option_chain(expiry)
        df = chain.calls if option_type == "call" else chain.puts
        match = df[df["strike"] == float(strike)]
        if match.empty:
            available = df["strike"].tolist()
            close_strikes = [s for s in available if abs(s - float(strike)) < 10]
            return f"Strike ${strike} not found for {expiry}. Near strikes: {close_strikes[:10]}"

        row = match.iloc[0]
        ask = row.get("ask", 0)
        if not ask or ask == 0:
            ask = row.get("lastPrice", 0)
        cost = ask * 100 * contracts  # per contract = 100 shares
    except Exception as e:
        return f"Error looking up option: {e}"

    portfolio = _load_portfolio()
    if cost > portfolio["cash"]:
        return f"Not enough cash. Need ${cost:,.2f}, have ${portfolio['cash']:,.2f}"

    position = {
        "id": str(uuid.uuid4())[:8],
        "ticker": ticker,
        "type": option_type,
        "strike": float(strike),
        "expiry": expiry,
        "contracts": contracts,
        "entry_price": ask,
        "cost": cost,
        "opened": datetime.now().isoformat(),
    }

    portfolio["cash"] -= cost
    portfolio["open_positions"].append(position)
    portfolio["total_trades"] += 1
    _save_portfolio(portfolio)

    return (
        f"BOUGHT {contracts}x {ticker} ${strike} {option_type} ({expiry})\n"
        f"Entry: ${ask:.2f}/contract  Cost: ${cost:,.2f}\n"
        f"Position ID: {position['id']}\n"
        f"Cash remaining: ${portfolio['cash']:,.2f}"
    )


def action_paper_sell(params: dict) -> str:
    position_id = params.get("position_id", "")
    if not position_id:
        return "Need a position_id. Use portfolio action to see open positions."

    portfolio = _load_portfolio()
    pos = None
    for p in portfolio["open_positions"]:
        if p["id"] == position_id:
            pos = p
            break

    if not pos:
        return f"No open position with ID '{position_id}'"

    # Get current price
    try:
        t = yf.Ticker(pos["ticker"])
        chain = t.option_chain(pos["expiry"])
        df = chain.calls if pos["type"] == "call" else chain.puts
        match = df[df["strike"] == pos["strike"]]
        if match.empty:
            return f"Can't find current price for this option (may have expired)"
        row = match.iloc[0]
        bid = row.get("bid", 0)
        if not bid:
            bid = row.get("lastPrice", 0)
    except Exception as e:
        return f"Error getting current price: {e}"

    proceeds = bid * 100 * pos["contracts"]
    pnl = proceeds - pos["cost"]
    pnl_pct = (pnl / pos["cost"] * 100) if pos["cost"] else 0

    closed = {
        **pos,
        "exit_price": bid,
        "proceeds": proceeds,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "closed": datetime.now().isoformat(),
    }

    portfolio["open_positions"].remove(pos)
    portfolio["closed_positions"].append(closed)
    portfolio["cash"] += proceeds
    _save_portfolio(portfolio)

    win = "WIN" if pnl > 0 else "LOSS"
    return (
        f"SOLD {pos['contracts']}x {pos['ticker']} ${pos['strike']} {pos['type']}\n"
        f"Exit: ${bid:.2f}  Proceeds: ${proceeds:,.2f}\n"
        f"P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%) — {win}\n"
        f"Cash: ${portfolio['cash']:,.2f}"
    )


def action_portfolio(params: dict) -> str:
    portfolio = _load_portfolio()
    lines = ["=== PAPER PORTFOLIO ==="]
    lines.append(f"Cash: ${portfolio['cash']:,.2f}")
    lines.append(f"Total trades: {portfolio['total_trades']}")

    if portfolio["open_positions"]:
        lines.append(f"\n--- Open Positions ({len(portfolio['open_positions'])}) ---")
        for p in portfolio["open_positions"]:
            lines.append(
                f"  [{p['id']}] {p['contracts']}x {p['ticker']} ${p['strike']} {p['type']} "
                f"({p['expiry']}) @ ${p['entry_price']:.2f}"
            )
    else:
        lines.append("\nNo open positions.")

    closed = portfolio["closed_positions"]
    if closed:
        wins = sum(1 for c in closed if c.get("pnl", 0) > 0)
        total_pnl = sum(c.get("pnl", 0) for c in closed)
        win_rate = (wins / len(closed) * 100) if closed else 0

        lines.append(f"\n--- Closed ({len(closed)}) ---")
        lines.append(f"Win rate: {win_rate:.0f}% ({wins}/{len(closed)})")
        lines.append(f"Total P&L: ${total_pnl:+,.2f}")

        for c in closed[-5:]:  # last 5
            win = "W" if c.get("pnl", 0) > 0 else "L"
            lines.append(
                f"  [{win}] {c['ticker']} ${c['strike']} {c['type']} "
                f"P&L: ${c.get('pnl', 0):+,.2f} ({c.get('pnl_pct', 0):+.1f}%)"
            )

    return "\n".join(lines)


def action_analyze(params: dict) -> str:
    if not yf:
        return "yfinance not installed."

    ticker = params.get("ticker", "SPY")
    try:
        t = yf.Ticker(ticker)
        info = t.info
        price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose", 0)

        # VIX
        vix = yf.Ticker("^VIX")
        vix_price = vix.info.get("regularMarketPrice") or vix.info.get("previousClose", 0)

        # Get nearest options chain for implied move
        dates = t.options
        if dates:
            chain = t.option_chain(dates[0])
            atm_calls = chain.calls.iloc[(chain.calls['strike'] - price).abs().argsort()[:1]]
            atm_puts = chain.puts.iloc[(chain.puts['strike'] - price).abs().argsort()[:1]]
            call_price = atm_calls.iloc[0].get("ask", 0) if not atm_calls.empty else 0
            put_price = atm_puts.iloc[0].get("ask", 0) if not atm_puts.empty else 0
            straddle = (call_price or 0) + (put_price or 0)
            implied_move = (straddle / price * 100) if price else 0
            expiry = dates[0]
        else:
            straddle = implied_move = 0
            expiry = "?"

        # Strategy suggestion
        if isinstance(vix_price, (int, float)) and vix_price > 25:
            strategy = "HIGH VIX — Sell premium. Credit spreads or iron condors. Elevated premiums = edge for sellers."
        elif isinstance(vix_price, (int, float)) and vix_price < 15:
            strategy = "LOW VIX — Buy premium is cheap. Straddles before catalysts. Or wait for better setup."
        elif implied_move > 2:
            strategy = f"BIG IMPLIED MOVE ({implied_move:.1f}%). Catalyst expected. Straddle/strangle if you agree, iron condor if you disagree."
        else:
            strategy = "NORMAL CONDITIONS — Credit spreads at 15-20 delta, 5-7 DTE. Take profit at 50-75% max."

        lines = [
            f"=== {ticker} ANALYSIS ===",
            f"Price: ${price}",
            f"VIX: {vix_price}",
            f"Nearest expiry: {expiry}",
            f"ATM straddle: ${straddle:.2f} (implied move: {implied_move:.1f}%)",
            f"",
            f"STRATEGY: {strategy}",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing {ticker}: {e}"


# ── Standalone helper for chat.py ──────────────────────────

def execute_trade(action: str, params: dict) -> str:
    """Standalone entry point for chat.py (no MCP dependency)."""
    dispatch = {
        "market": action_market,
        "options_chain": action_options_chain,
        "paper_buy": action_paper_buy,
        "paper_sell": action_paper_sell,
        "portfolio": action_portfolio,
        "analyze": action_analyze,
    }
    handler = dispatch.get(action)
    if not handler:
        return f"Unknown action: {action}. Use: {list(dispatch.keys())}"
    return handler(params)


# ── MCP Registration ───────────────────────────────────────

TOOLS = [
    Tool(
        name="watty_trade",
        description=(
            "Watty's trading desk. Live market data + paper trading.\n"
            "Actions: market (price/VIX), options_chain (full chain with Greeks), "
            "paper_buy (open position), paper_sell (close position), "
            "portfolio (view positions), analyze (strategy suggestion)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["market", "options_chain", "paper_buy", "paper_sell", "portfolio", "analyze"],
                    "description": "Action to perform",
                },
                "ticker": {"type": "string", "description": "Ticker symbol (default: SPY)"},
                "expiry": {"type": "string", "description": "Options expiry date (YYYY-MM-DD)"},
                "option_type": {"type": "string", "enum": ["call", "put"], "description": "paper_buy: call or put"},
                "strike": {"type": "number", "description": "paper_buy: strike price"},
                "contracts": {"type": "integer", "description": "paper_buy: number of contracts (default: 1)"},
                "position_id": {"type": "string", "description": "paper_sell: ID of position to close"},
            },
            "required": ["action"],
        },
    ),
]


async def handle_watty_trade(params: dict) -> list[TextContent]:
    action = params.get("action", "market")
    result = execute_trade(action, params)
    return [TextContent(type="text", text=result)]


HANDLERS = {
    "watty_trade": handle_watty_trade,
}
