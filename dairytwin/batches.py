"""Age-structured stock.

Every stock in the twin is an integer array whose last axis is *age in days since
production*: stock[..., a] units are a days old. This is what makes waste come from
real expiry ("this pouch was made 3 days ago and nobody bought it") instead of a
single average decay rate.

All helpers are vectorised over the leading axes, so the same code serves one plant,
six dark stores, or 240 kirana actors at once.
"""
from __future__ import annotations

import numpy as np


def fifo_take(stock: np.ndarray, qty: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove up to `qty` units, oldest first.

    stock: (..., A) int, qty: (...) int.  Returns (taken (..., A), remaining (..., A)).
    """
    rev = stock[..., ::-1]
    cum = np.cumsum(rev, axis=-1)
    before = cum - rev
    take_rev = np.clip(qty[..., None] - before, 0, rev)
    taken = take_rev[..., ::-1]
    return taken, stock - taken


def add_batch(stock: np.ndarray, qty: np.ndarray, age: int = 0) -> np.ndarray:
    """Add units of a given age. Broadcasts qty over the leading axes."""
    stock[..., age] += qty
    return stock


def add_aged(stock: np.ndarray, incoming: np.ndarray) -> np.ndarray:
    """Add a full age profile (..., A) of incoming units."""
    stock += incoming
    return stock


def age_and_expire(stock: np.ndarray, shelf: np.ndarray, extra: np.ndarray | None = None,
                   base: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Age every unit by `base` days (1 at the evening close, 0 for a mid-day power cut)
    plus `extra` days for stock that sat in a dead fridge, then expire units whose age
    reached their product's shelf life.

    stock: (..., P, A); shelf: (P,) int; extra: (..., P) int extra days (>= 0).
    Returns (expired units (..., P), new stock).
    """
    A = stock.shape[-1]
    shift = np.full(stock.shape[:-1], base, dtype=np.int64)
    if extra is not None:
        shift = shift + extra
    new = np.zeros_like(stock)
    expired = np.zeros(stock.shape[:-1], dtype=stock.dtype)
    for s in np.unique(shift):
        part = np.where((shift == s)[..., None], stock, 0)
        if s >= A:                       # pushed off the age axis entirely
            expired += part.sum(axis=-1)
            continue
        new[..., s:] += part[..., : A - s]
        expired += part[..., A - s:].sum(axis=-1)   # pushed past the last age bin

    ages = np.arange(A)
    dead = ages[None, :] >= shelf[:, None]                    # (P, A)
    expired += np.where(dead, new, 0).sum(axis=-1)
    new = np.where(dead, 0, new)
    return expired, new


def total(stock: np.ndarray) -> np.ndarray:
    return stock.sum(axis=-1)
