"""Synthetic tick data generator using GBM + Jump Diffusion."""

import math
from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np

from ..domain.enums import ProductType, Session
from ..domain.models import Tick


def generate_ticks(
    start_price: float = 20000.0,
    num_ticks: int = 10000,
    start_time: datetime | None = None,
    product: ProductType = ProductType.TX,
    session: Session = Session.DAY,
    mu: float = 0.0,
    sigma: float = 0.15,
    jump_intensity: float = 0.01,
    jump_mean: float = 0.0,
    jump_std: float = 0.005,
    tick_interval_ms: float = 200.0,
    seed: int | None = None,
) -> list[Tick]:
    """Generate synthetic tick data using GBM with jump diffusion.

    Args:
        start_price: Initial price level.
        num_ticks: Number of ticks to generate.
        start_time: Starting timestamp (defaults to today 08:45).
        product: Futures product type.
        session: Trading session.
        mu: Drift (annualized).
        sigma: Volatility (annualized).
        jump_intensity: Poisson jump intensity (probability per tick).
        jump_mean: Mean of jump size (log-normal).
        jump_std: Std of jump size (log-normal).
        tick_interval_ms: Average milliseconds between ticks.
        seed: Random seed for reproducibility.

    Returns:
        List of Tick objects with synthetic prices.
    """
    rng = np.random.default_rng(seed)

    if start_time is None:
        today = datetime.now().replace(hour=8, minute=45, second=0, microsecond=0)
        start_time = today

    # Time step in years (assuming ~252 trading days, ~5h per day session)
    dt = tick_interval_ms / (1000.0 * 3600.0 * 5.0 * 252.0)

    # GBM increments
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * math.sqrt(dt) * rng.standard_normal(num_ticks)

    # Jump component (compound Poisson)
    jumps_occur = rng.poisson(jump_intensity, num_ticks)
    jump_sizes = jumps_occur * rng.normal(jump_mean, jump_std, num_ticks)

    # Build log-price path
    log_returns = drift + diffusion + jump_sizes
    log_prices = np.cumsum(log_returns)
    log_prices = np.insert(log_prices, 0, 0.0)[:-1]
    prices = start_price * np.exp(log_prices)

    # Round prices to integer (TAIEX futures tick size = 1 point)
    prices = np.round(prices).astype(int)

    # Generate volumes (log-normal distribution)
    volumes = rng.lognormal(mean=1.5, sigma=1.0, size=num_ticks)
    volumes = np.clip(volumes, 1, 200).astype(int)

    # Generate timestamps
    intervals_ms = rng.exponential(tick_interval_ms, num_ticks)
    intervals_ms = np.clip(intervals_ms, 10, tick_interval_ms * 10)
    cumulative_ms = np.cumsum(intervals_ms)

    ticks = []
    for i in range(num_ticks):
        ts = start_time + timedelta(milliseconds=float(cumulative_ms[i]))
        ticks.append(Tick(
            timestamp=ts,
            price=Decimal(str(int(prices[i]))),
            volume=int(volumes[i]),
            product=product,
            session=session,
        ))

    return ticks
