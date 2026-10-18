"""Risk rules every order must pass. The strategy cannot override them.

The same RiskManager is used in backtests now and will gate live orders later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .config import RiskConfig


@dataclass
class Decision:
    approved: bool
    qty: float = 0.0
    reason: str = ""


@dataclass
class RiskManager:
    cfg: RiskConfig
    peak_equity: float = 0.0
    day: date | None = None
    day_start_equity: float = 0.0
    halted_today: bool = False
    killed: bool = False
    events: list[str] = field(default_factory=list)

    def update(self, equity: float, today: date) -> None:
        """Feed the latest marked-to-market equity. Call once per candle."""
        if today != self.day:
            self.day = today
            self.day_start_equity = equity
            self.halted_today = False
        self.peak_equity = max(self.peak_equity, equity)

        if not self.killed and equity <= self.peak_equity * (1 - self.cfg.max_drawdown):
            self.killed = True
            self.events.append(f"{today}: kill switch, equity {equity:.2f} is "
                               f">= {self.cfg.max_drawdown:.0%} below peak {self.peak_equity:.2f}")
        if not self.halted_today and equity <= self.day_start_equity * (1 - self.cfg.daily_loss_limit):
            self.halted_today = True
            self.events.append(f"{today}: daily loss limit hit, no new entries today")

    @property
    def can_open(self) -> bool:
        return not (self.killed or self.halted_today)

    def size_entry(self, equity: float, cash: float, price: float, stop_price: float,
                   last_price: float, fee_rate: float = 0.0) -> Decision:
        """Size a long entry so a stop-out loses about `risk_per_trade` of equity."""
        if self.killed:
            return Decision(False, reason="kill switch active")
        if self.halted_today:
            return Decision(False, reason="daily loss limit hit")
        if price <= 0 or equity <= 0:
            return Decision(False, reason="invalid price or equity")
        if abs(price - last_price) / last_price > self.cfg.max_price_deviation:
            return Decision(False, reason="price too far from last price")
        if stop_price >= price or stop_price <= 0:
            return Decision(False, reason="stop must be below entry and above zero")

        qty_by_risk = equity * self.cfg.risk_per_trade / (price - stop_price)
        qty_by_cap = equity * self.cfg.max_position_pct / price
        qty_by_cash = cash / (price * (1 + fee_rate))
        qty = min(qty_by_risk, qty_by_cap, qty_by_cash)
        if qty <= 0:
            return Decision(False, reason="no cash available")
        return Decision(True, qty=qty)
