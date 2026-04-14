"""
utils.py
--------
Shared utility functions used across the project.

Includes:
    - Thread-safe worker wrapper for running blocking tasks in the background.
    - Date validation helpers.
    - Formatting helpers for metrics display.
    - Matplotlib dark-theme configuration.
"""

import threading
from datetime import datetime, date
from typing import Callable, Any


# ---------------------------------------------------------------------------
# Background task runner
# ---------------------------------------------------------------------------

class BackgroundWorker:
    """
    Runs a blocking callable in a daemon thread so the GUI stays responsive.

    Usage::

        def on_done(result, error):
            if error:
                print(f"Error: {error}")
            else:
                print(f"Result: {result}")

        worker = BackgroundWorker(my_function, arg1, arg2, on_complete=on_done)
        worker.start()
    """

    def __init__(self, fn: Callable, *args, on_complete: Callable = None, **kwargs):
        """
        Args:
            fn:          The blocking function to execute in the background.
            *args:       Positional arguments forwarded to fn.
            on_complete: Optional callback called with (result, exception) when
                         fn finishes. Called from the worker thread — GUI updates
                         must be dispatched back to the main thread via after().
            **kwargs:    Keyword arguments forwarded to fn.
        """
        self._fn          = fn
        self._args        = args
        self._kwargs      = kwargs
        self._on_complete = on_complete
        self._thread      = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        """Internal: execute fn and invoke the completion callback."""
        try:
            result = self._fn(*self._args, **self._kwargs)
            if self._on_complete:
                self._on_complete(result, None)
        except Exception as exc:
            if self._on_complete:
                self._on_complete(None, exc)

    def start(self) -> None:
        """Launch the background thread."""
        self._thread.start()

    def is_alive(self) -> bool:
        """Return True if the background thread is still running."""
        return self._thread.is_alive()


# ---------------------------------------------------------------------------
# Date utilities
# ---------------------------------------------------------------------------

def validate_date_range(start: str, end: str) -> tuple[bool, str]:
    """
    Check that start and end are valid dates and in the correct order.

    Args:
        start: Start date string "YYYY-MM-DD".
        end:   End date string "YYYY-MM-DD".

    Returns:
        (True, "") if valid, or (False, reason_string) if not.
    """
    fmt = "%Y-%m-%d"
    try:
        s = datetime.strptime(start, fmt).date()
        e = datetime.strptime(end, fmt).date()
    except ValueError:
        return False, "Dates must be in YYYY-MM-DD format."

    if s >= e:
        return False, "Start date must be before end date."

    if e > date.today():
        return False, "End date cannot be in the future."

    # Need at least ~2 years of daily data for meaningful LSTM training
    delta_days = (e - s).days
    if delta_days < 365:
        return False, "Date range should span at least 1 year for reliable predictions."

    return True, ""


def date_to_str(d) -> str:
    """Convert a date / datetime to 'YYYY-MM-DD' string."""
    if hasattr(d, "strftime"):
        return d.strftime("%Y-%m-%d")
    return str(d)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_metrics(metrics: dict, ticker: str = "") -> str:
    """
    Build a human-readable string from an evaluation metrics dictionary.

    Args:
        metrics: Dict with 'rmse' and 'mae' keys (float values in USD).
        ticker:  Optional ticker name to include in the header.

    Returns:
        Multi-line string ready for display in the GUI.
    """
    header = f"  {ticker} — Model Performance  " if ticker else "  Model Performance  "
    sep    = "=" * len(header)
    lines  = [
        sep,
        header,
        sep,
        f"  RMSE : ${metrics['rmse']:>10.4f}",
        f"  MAE  : ${metrics['mae']:>10.4f}",
        sep,
    ]
    return "\n".join(lines)


def format_large_number(n) -> str:
    """Format a large integer (e.g. market cap) into a readable string."""
    if not isinstance(n, (int, float)):
        return str(n)
    if n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"${n / 1e9:.2f}B"
    if n >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.0f}"


# ---------------------------------------------------------------------------
# Matplotlib dark theme
# ---------------------------------------------------------------------------

DARK_THEME = {
    # Figure / axes backgrounds
    "figure.facecolor":     "#1a1a2e",
    "axes.facecolor":       "#16213e",
    "axes.edgecolor":       "#4a4a7a",

    # Grid
    "axes.grid":            True,
    "grid.color":           "#2a2a4a",
    "grid.linestyle":       "--",
    "grid.alpha":           0.7,

    # Text colours
    "text.color":           "#e0e0e0",
    "axes.labelcolor":      "#e0e0e0",
    "xtick.color":          "#a0a0c0",
    "ytick.color":          "#a0a0c0",

    # Legend
    "legend.facecolor":     "#16213e",
    "legend.edgecolor":     "#4a4a7a",
    "legend.labelcolor":    "#e0e0e0",

    # Lines
    "lines.linewidth":      1.5,
}

# Colour palette used across all charts
CHART_COLORS = {
    "actual":     "#4fc3f7",   # light blue  — actual prices
    "train_pred": "#81c784",   # green        — training predictions
    "test_pred":  "#ffb74d",   # amber        — test predictions
    "forecast":   "#f06292",   # pink         — future forecast
    "volume":     "#7986cb",   # indigo       — volume bars
    "sma20":      "#fff176",   # yellow       — SMA 20
    "sma50":      "#ce93d8",   # purple       — SMA 50
    "rsi":        "#80deea",   # cyan         — RSI line
    "macd":       "#a5d6a7",   # mint         — MACD line
    "signal":     "#ef9a9a",   # red          — signal line
}


def apply_dark_theme(fig, axes) -> None:
    """
    Apply the project's dark theme to a Matplotlib figure and its axes.

    Args:
        fig:  Matplotlib Figure object.
        axes: Single Axes or list/array of Axes objects.
    """
    import matplotlib as mpl
    mpl.rcParams.update(DARK_THEME)

    fig.patch.set_facecolor(DARK_THEME["figure.facecolor"])

    # Normalise to a list so we can iterate uniformly
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    else:
        axes = list(axes) if not isinstance(axes, list) else axes

    for ax in axes:
        ax.set_facecolor(DARK_THEME["axes.facecolor"])
        for spine in ax.spines.values():
            spine.set_edgecolor(DARK_THEME["axes.edgecolor"])
        ax.tick_params(colors=DARK_THEME["xtick.color"])
        ax.grid(
            True,
            color=DARK_THEME["grid.color"],
            linestyle=DARK_THEME["grid.linestyle"],
            alpha=DARK_THEME["grid.alpha"],
        )
