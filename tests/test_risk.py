from datetime import date

import pytest

from bot.config import RiskConfig
from bot.risk import RiskManager


def manager(**overrides):
    return RiskManager(RiskConfig(**overrides))


def test_size_limited_by_risk_per_trade():
    rm = manager(risk_per_trade=0.01, max_position_pct=1.0)
    d = rm.size_entry(equity=1_000, cash=1_000, price=100, stop_price=90, last_price=100)
    assert d.approved
    assert d.qty == pytest.approx(1.0)  # 1% of 1000 = 10 USDT risk / 10 per unit


def test_size_capped_by_max_position():
    rm = manager(risk_per_trade=0.5, max_position_pct=0.2)
    d = rm.size_entry(equity=1_000, cash=1_000, price=100, stop_price=99, last_price=100)
    assert d.qty == pytest.approx(2.0)  # 20% of 1000 / 100


def test_size_capped_by_cash_including_fee():
    rm = manager(risk_per_trade=1.0, max_position_pct=1.0)
    d = rm.size_entry(equity=1_000, cash=100, price=100, stop_price=50, last_price=100, fee_rate=0.001)
    assert d.qty * 100 * 1.001 == pytest.approx(100)


def test_rejects_bad_orders():
    rm = manager()
    assert not rm.size_entry(1_000, 1_000, 100, 101, 100).approved  # stop above entry
    assert not rm.size_entry(1_000, 1_000, 110, 100, 100).approved  # 10% from last price


def test_daily_loss_limit_blocks_entries_until_next_day():
    rm = manager(daily_loss_limit=0.03)
    rm.update(1_000, date(2024, 1, 1))
    rm.update(965, date(2024, 1, 1))
    assert not rm.can_open
    assert not rm.size_entry(965, 965, 100, 90, 100).approved
    rm.update(965, date(2024, 1, 2))
    assert rm.can_open


def test_kill_switch_is_permanent():
    rm = manager(max_drawdown=0.15)
    rm.update(1_000, date(2024, 1, 1))
    rm.update(1_100, date(2024, 1, 2))
    rm.update(930, date(2024, 1, 3))  # 15.5% below the 1100 peak
    assert rm.killed
    rm.update(2_000, date(2024, 1, 4))
    assert rm.killed and not rm.can_open
