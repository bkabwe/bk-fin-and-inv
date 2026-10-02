"""Price rounding and display helpers.

Kept free of heavy imports (no pandas / scoring engine) so the emailed-report
scripts can use them without loading the whole forecasting stack.
"""

from __future__ import annotations


def price_decimals(reference_price: float | None) -> int:
    """Decimal places to keep when rounding price-like values (targets,
    confidence-band edges, current price) for a security trading around
    ``reference_price``.

    Whole cents are plenty for a $50 stock but destroy a penny stock's
    projection: at $0.12 one cent is ~8% of the price, so cent-rounding a
    target moves its "upside" by several points (and can even push it past the
    +200% cap or below the current price), while a sub-cent band edge collapses
    to exactly $0.00, which downstream code then mistakes for "missing" (see
    modules.profit_opportunities.profit_row_from_analysis). US equities quote
    in $0.0001 steps below $1, so keep four decimals from $0.01 up to $1 and
    six below a cent. Prices of $1 and above keep the historical two decimals.
    """
    price = abs(float(reference_price or 0.0))
    if price >= 1.0:
        return 2
    if price >= 0.01:
        return 4
    return 6


def round_price(value: float, reference_price: float | None = None) -> float:
    """Round ``value`` to the precision appropriate for ``reference_price``
    (defaults to ``value`` itself) -- see :func:`price_decimals`."""
    return round(float(value), price_decimals(value if reference_price is None else reference_price))


def format_price(value: object) -> str:
    """Display a price as ``$1,234.56`` (or with four/six decimals for
    sub-dollar prices, see :func:`price_decimals`). Missing or non-numeric
    values render as an em dash / the original text."""
    if value is None or value == "":
        return "—"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "—"
    return f"${number:,.{price_decimals(number)}f}"
