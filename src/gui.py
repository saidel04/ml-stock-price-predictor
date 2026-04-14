"""
gui.py
------
Main application window built with CustomTkinter.

Layout:
    ┌──────────────┬────────────────────────────────────────┐
    │  LEFT PANEL  │  CHART TABS                            │
    │  Ticker      │  Overview | Predictions | Forecast     │
    │  Dates       │                                        │
    │  Model pick  │                                        │
    │  Train btn   │                                        │
    │  Forecast btn│                                        │
    │  Metrics     │                                        │
    └──────────────┴────────────────────────────────────────┘
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import customtkinter as ctk

from data_fetcher import fetch_stock_data, get_company_info
from preprocessor import StockPreprocessor
from model        import StockPredictor, MODEL_OPTIONS
from utils        import (
    BackgroundWorker, validate_date_range,
    format_large_number, apply_dark_theme, CHART_COLORS, DARK_THEME,
)

# ── Appearance ───────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
plt.rcParams.update(DARK_THEME)

# ── Constants ────────────────────────────────────────────────────────────────
PAD       = 12
PANEL_W   = 270
FONT_H2   = ("Segoe UI", 13, "bold")
FONT_BODY = ("Segoe UI", 12)
FONT_MONO = ("Consolas", 11)
ACCENT    = "#4fc3f7"
BG_CARD   = "#1e2a3a"


class App(ctk.CTk):
    """
    Root application window.

    Workflow:
        1. User enters ticker + date range, picks a model, clicks Train.
        2. Data is fetched, indicators computed, model trained in a background thread.
        3. Predictions tab shows actual vs predicted on the test set.
        4. User clicks Forecast → autoregressive n-day forecast is drawn.
    """

    def __init__(self):
        super().__init__()
        self.title("Stock Price Predictor")
        self.geometry("1200x760")
        self.minsize(1000, 650)

        # Internal state
        self._df:           Optional[pd.DataFrame]      = None
        self._data:         Optional[dict]              = None
        self._predictor:    Optional[StockPredictor]    = None
        self._preprocessor: Optional[StockPreprocessor] = None
        self._test_pred:    Optional[np.ndarray]        = None
        self._future_pred:  Optional[np.ndarray]        = None

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_charts()

    # ─────────────────────────────────────────────────────────────────────────
    # Sidebar
    # ─────────────────────────────────────────────────────────────────────────

    def _build_sidebar(self) -> None:
        """Build the left control panel."""
        sb = ctk.CTkScrollableFrame(self, width=PANEL_W, corner_radius=0, fg_color="#0f1923")
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_columnconfigure(0, weight=1)
        r = 0

        # Title
        ctk.CTkLabel(sb, text="Stock Predictor", font=("Segoe UI", 20, "bold"),
                     text_color=ACCENT).grid(row=r, column=0, padx=PAD, pady=(20, 4), sticky="ew"); r += 1
        ctk.CTkLabel(sb, text="scikit-learn  |  dark theme",
                     font=("Segoe UI", 10), text_color="#5a7090").grid(
            row=r, column=0, padx=PAD, pady=(0, 16), sticky="ew"); r += 1

        # Ticker
        self._label(sb, r, "TICKER"); r += 1
        self._ticker = ctk.StringVar(value="AAPL")
        ctk.CTkEntry(sb, textvariable=self._ticker, height=38,
                     font=("Segoe UI", 14, "bold"),
                     placeholder_text="e.g. AAPL, TSLA, MSFT").grid(
            row=r, column=0, padx=PAD, pady=(0, PAD), sticky="ew"); r += 1

        # Date range
        self._label(sb, r, "DATE RANGE"); r += 1
        today         = datetime.today()
        self._start   = ctk.StringVar(value=(today - timedelta(days=5*365)).strftime("%Y-%m-%d"))
        self._end     = ctk.StringVar(value=today.strftime("%Y-%m-%d"))
        for label, var in [("Start", self._start), ("End", self._end)]:
            ctk.CTkLabel(sb, text=label, font=FONT_BODY, anchor="w").grid(
                row=r, column=0, padx=PAD, sticky="ew"); r += 1
            ctk.CTkEntry(sb, textvariable=var, height=34).grid(
                row=r, column=0, padx=PAD, pady=(0, 6), sticky="ew"); r += 1

        # Model selector
        self._label(sb, r, "MODEL"); r += 1
        self._model_var = ctk.StringVar(value="Random Forest")
        ctk.CTkOptionMenu(sb, variable=self._model_var,
                          values=list(MODEL_OPTIONS.keys()),
                          height=34).grid(
            row=r, column=0, padx=PAD, pady=(0, PAD), sticky="ew"); r += 1

        # Forecast horizon
        self._label(sb, r, "FORECAST HORIZON"); r += 1
        self._horizon = ctk.IntVar(value=30)
        ctk.CTkSlider(sb, variable=self._horizon, from_=5, to=60,
                      number_of_steps=11).grid(
            row=r, column=0, padx=PAD, sticky="ew")
        self._horizon_lbl = ctk.CTkLabel(sb, text="30 days", font=FONT_BODY)
        self._horizon_lbl.grid(row=r, column=0, padx=(PAD+155, PAD), sticky="e")
        self._horizon.trace_add("write", lambda *_: self._horizon_lbl.configure(
            text=f"{self._horizon.get()} days")); r += 1

        # Divider
        ctk.CTkFrame(sb, height=1, fg_color="#2a3a4a").grid(
            row=r, column=0, padx=PAD, pady=14, sticky="ew"); r += 1

        # Buttons
        self._train_btn = ctk.CTkButton(
            sb, text="Train Model", height=42, font=FONT_H2,
            fg_color="#1565c0", hover_color="#1976d2",
            command=self._on_train)
        self._train_btn.grid(row=r, column=0, padx=PAD, pady=(0, 8), sticky="ew"); r += 1

        self._forecast_btn = ctk.CTkButton(
            sb, text="Generate Forecast", height=42, font=FONT_H2,
            fg_color="#1b5e20", hover_color="#2e7d32",
            state="disabled", command=self._on_forecast)
        self._forecast_btn.grid(row=r, column=0, padx=PAD, pady=(0, PAD), sticky="ew"); r += 1

        # Progress / status
        self._progress = ctk.CTkProgressBar(sb, mode="indeterminate")
        self._progress.grid(row=r, column=0, padx=PAD, sticky="ew"); r += 1
        self._progress.set(0)

        self._status = ctk.CTkLabel(sb, text="Ready.", font=FONT_BODY,
                                    text_color="#6080a0", wraplength=PANEL_W - 2*PAD)
        self._status.grid(row=r, column=0, padx=PAD, pady=(4, PAD), sticky="ew"); r += 1

        # Metrics card
        card = ctk.CTkFrame(sb, fg_color=BG_CARD, corner_radius=8)
        card.grid(row=r, column=0, padx=PAD, pady=(0, PAD), sticky="ew"); r += 1
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(card, text="Metrics", font=FONT_H2,
                     text_color=ACCENT).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")
        self._rmse_lbl = ctk.CTkLabel(card, text="RMSE  —", font=FONT_MONO)
        self._rmse_lbl.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self._mae_lbl  = ctk.CTkLabel(card, text="MAE   —", font=FONT_MONO)
        self._mae_lbl.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self._r2_lbl   = ctk.CTkLabel(card, text="R²    —", font=FONT_MONO)
        self._r2_lbl.grid(row=3, column=0, padx=10, pady=(2, 8), sticky="w")

        # Company info card
        info = ctk.CTkFrame(sb, fg_color=BG_CARD, corner_radius=8)
        info.grid(row=r, column=0, padx=PAD, pady=(0, 24), sticky="ew"); r += 1
        info.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(info, text="Company", font=FONT_H2,
                     text_color=ACCENT).grid(row=0, column=0, padx=10, pady=(8, 4), sticky="w")
        self._info_name   = ctk.CTkLabel(info, text="Name —", font=FONT_MONO)
        self._info_name.grid(row=1, column=0, padx=10, pady=2, sticky="w")
        self._info_sector = ctk.CTkLabel(info, text="Sector —", font=FONT_MONO)
        self._info_sector.grid(row=2, column=0, padx=10, pady=2, sticky="w")
        self._info_cap    = ctk.CTkLabel(info, text="Cap —", font=FONT_MONO)
        self._info_cap.grid(row=3, column=0, padx=10, pady=(2, 8), sticky="w")

    def _label(self, parent, row: int, text: str) -> None:
        """Small all-caps section label."""
        ctk.CTkLabel(parent, text=text, font=("Segoe UI", 10, "bold"),
                     text_color="#4a6880").grid(
            row=row, column=0, padx=PAD, pady=(PAD, 2), sticky="w")

    # ─────────────────────────────────────────────────────────────────────────
    # Chart area
    # ─────────────────────────────────────────────────────────────────────────

    def _build_charts(self) -> None:
        """Three-tab chart area: Overview, Predictions, Forecast."""
        self._tabs = ctk.CTkTabview(
            self, fg_color="#12192a",
            segmented_button_selected_color=ACCENT,
            segmented_button_selected_hover_color="#37a6d4",
        )
        self._tabs.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)

        for name in ("Overview", "Predictions", "Forecast"):
            self._tabs.add(name)
            self._tabs.tab(name).grid_columnconfigure(0, weight=1)
            self._tabs.tab(name).grid_rowconfigure(0, weight=1)

        self._build_overview_tab()
        self._build_predictions_tab()
        self._build_forecast_tab()

    def _embed(self, fig: plt.Figure, parent) -> FigureCanvasTkAgg:
        """Embed a Matplotlib figure with a navigation toolbar."""
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        tb_frame = ctk.CTkFrame(parent, fg_color="#0f1923", height=28)
        tb_frame.grid(row=1, column=0, sticky="ew")
        parent.grid_rowconfigure(1, weight=0)

        tb = NavigationToolbar2Tk(canvas, tb_frame)
        tb.update()
        tb.config(background="#0f1923")
        for child in tb.winfo_children():
            try:
                child.config(background="#0f1923", foreground="#c0c0c0")
            except Exception:
                pass

        canvas.draw()
        fig._canvas = canvas   # store for later redraws
        return canvas

    @staticmethod
    def _placeholder(ax: plt.Axes, msg: str) -> None:
        ax.clear()
        ax.set_facecolor(DARK_THEME["axes.facecolor"])
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=13,
                color="#334455", style="italic")
        ax.set_xticks([]); ax.set_yticks([])

    def _build_overview_tab(self) -> None:
        tab = self._tabs.tab("Overview")
        fig, (ax_p, ax_v, ax_r) = plt.subplots(
            3, 1, figsize=(9, 6),
            gridspec_kw={"height_ratios": [4, 1.5, 1.5]},
            facecolor=DARK_THEME["figure.facecolor"],
        )
        apply_dark_theme(fig, [ax_p, ax_v, ax_r])
        fig.tight_layout(pad=1.5); fig.subplots_adjust(hspace=0.12)
        self._ov_fig, self._ax_p, self._ax_v, self._ax_r = fig, ax_p, ax_v, ax_r
        self._embed(fig, tab)
        self._placeholder(ax_p, "Enter a ticker and click Train Model")
        self._placeholder(ax_v, "Volume")
        self._placeholder(ax_r, "RSI")

    def _build_predictions_tab(self) -> None:
        tab = self._tabs.tab("Predictions")
        fig, ax = plt.subplots(figsize=(9, 6),
                               facecolor=DARK_THEME["figure.facecolor"])
        apply_dark_theme(fig, ax)
        fig.tight_layout(pad=2.0)
        self._pr_fig, self._ax_pred = fig, ax
        self._embed(fig, tab)
        self._placeholder(ax, "Train the model to see test-set predictions")

    def _build_forecast_tab(self) -> None:
        tab = self._tabs.tab("Forecast")
        fig, ax = plt.subplots(figsize=(9, 6),
                               facecolor=DARK_THEME["figure.facecolor"])
        apply_dark_theme(fig, ax)
        fig.tight_layout(pad=2.5)
        self._fc_fig, self._ax_fc = fig, ax
        self._embed(fig, tab)
        self._placeholder(ax, "Click Generate Forecast after training")

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _on_train(self) -> None:
        """Validate inputs, fetch data, train model — all in a background thread."""
        ticker = self._ticker.get().strip().upper()
        start  = self._start.get().strip()
        end    = self._end.get().strip()

        if not ticker:
            self._error("Please enter a ticker symbol."); return

        ok, reason = validate_date_range(start, end)
        if not ok:
            self._error(reason); return

        model_key = MODEL_OPTIONS[self._model_var.get()]
        self._busy(True)
        self._forecast_btn.configure(state="disabled")

        def _run():
            self._set_status(f"Downloading {ticker} data…")
            df = fetch_stock_data(ticker, start, end)
            self._df = df

            self._set_status("Loading company info…")
            info = get_company_info(ticker)
            self.after(0, lambda: self._update_info(info))
            self.after(0, self._plot_overview)

            self._set_status("Computing features…")
            pre  = StockPreprocessor()
            data = pre.prepare(df)
            self._preprocessor = pre
            self._data         = data

            self._set_status(f"Training {self._model_var.get()}…")
            pred = StockPredictor(model_type=model_key)
            pred.train(data["X_train"], data["y_train"])
            self._predictor = pred

            self._set_status("Evaluating on test set…")
            test_pred       = pred.predict(data["X_test"])
            self._test_pred = test_pred
            metrics         = pred.evaluate(data["y_test"], test_pred)

            self.after(0, lambda: self._update_metrics(metrics))
            self.after(0, self._plot_predictions)
            return "done"

        def _done(result, err):
            self._busy(False)
            if err:
                self._error(f"Training failed:\n{err}")
                self._set_status("Error.")
            else:
                self._set_status("Done — model trained successfully.")
                self.after(0, lambda: self._forecast_btn.configure(state="normal"))

        BackgroundWorker(_run, on_complete=_done).start()

    def _on_forecast(self) -> None:
        """Generate the autoregressive price forecast in a background thread."""
        if not self._predictor or not self._preprocessor or self._data is None:
            self._error("Train the model first."); return

        self._busy(True)
        horizon = self._horizon.get()

        def _run():
            df_clean = self._data["df_clean"]
            last_row = df_clean[self._preprocessor.feature_cols].values[-1]
            future   = self._predictor.predict_future(
                last_row, horizon, self._preprocessor, df_clean
            )
            self._future_pred = future
            return future

        def _done(result, err):
            self._busy(False)
            if err:
                self._error(f"Forecast failed:\n{err}")
            else:
                self.after(0, self._plot_forecast)
                self._set_status(f"Forecast: next {horizon} trading days.")
                self._tabs.set("Forecast")

        BackgroundWorker(_run, on_complete=_done).start()

    # ─────────────────────────────────────────────────────────────────────────
    # Chart drawing
    # ─────────────────────────────────────────────────────────────────────────

    def _plot_overview(self) -> None:
        """Price + SMA/Bollinger, volume bars, RSI."""
        if self._df is None:
            return

        df = self._df.copy()
        pre = StockPreprocessor()
        pre.add_technical_indicators(df)
        df.dropna(inplace=True)

        dates = df.index
        ticker = self._ticker.get().upper()

        ax_p, ax_v, ax_r = self._ax_p, self._ax_v, self._ax_r
        for ax in (ax_p, ax_v, ax_r):
            ax.clear()
        apply_dark_theme(self._ov_fig, [ax_p, ax_v, ax_r])

        # Price
        ax_p.plot(dates, df["Close"],   color=CHART_COLORS["actual"],  lw=1.5, label="Close")
        ax_p.plot(dates, df["SMA_20"],  color=CHART_COLORS["sma20"],   lw=1.0, ls="--", label="SMA 20")
        ax_p.plot(dates, df["SMA_50"],  color=CHART_COLORS["sma50"],   lw=1.0, ls="--", label="SMA 50")
        ax_p.fill_between(dates, df["BB_Upper"], df["BB_Lower"],
                          alpha=0.08, color=CHART_COLORS["actual"], label="Bollinger Bands")
        ax_p.set_title(f"{ticker} — Price & Indicators", color="#e0e0e0", fontsize=12, pad=6)
        ax_p.set_ylabel("Price (USD)", color="#e0e0e0")
        ax_p.legend(loc="upper left", fontsize=9, framealpha=0.3)
        ax_p.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # Volume
        colors = [CHART_COLORS["actual"] if c >= o else "#ef5350"
                  for c, o in zip(df["Close"], df["Open"])]
        ax_v.bar(dates, df["Volume"], color=colors, alpha=0.7, width=1)
        ax_v.set_ylabel("Volume", color="#e0e0e0", fontsize=9)
        ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))
        ax_v.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # RSI
        ax_r.plot(dates, df["RSI_14"], color=CHART_COLORS["rsi"], lw=1.2)
        ax_r.axhline(70, color="#ef5350", lw=0.8, ls="--", alpha=0.6)
        ax_r.axhline(30, color="#66bb6a", lw=0.8, ls="--", alpha=0.6)
        ax_r.fill_between(dates, df["RSI_14"], 70,
                          where=df["RSI_14"] >= 70, alpha=0.15, color="#ef5350")
        ax_r.fill_between(dates, df["RSI_14"], 30,
                          where=df["RSI_14"] <= 30, alpha=0.15, color="#66bb6a")
        ax_r.set_ylim(0, 100)
        ax_r.set_ylabel("RSI", color="#e0e0e0", fontsize=9)
        ax_r.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        self._ov_fig.tight_layout(pad=1.5)
        self._ov_fig.subplots_adjust(hspace=0.12)
        self._ov_fig._canvas.draw()

    def _plot_predictions(self) -> None:
        """Actual vs predicted close on the test set."""
        if self._data is None or self._test_pred is None:
            return

        data   = self._data
        df     = data["df_clean"]
        dates  = data["dates"]
        ticker = self._ticker.get().upper()
        ts     = data["train_size"]

        # Align: test predictions map to dates[ts : ts+len(test_pred)]
        test_dates  = dates[ts: ts + len(self._test_pred)]
        actual_test = data["y_test"][: len(self._test_pred)]

        ax = self._ax_pred
        ax.clear()
        apply_dark_theme(self._pr_fig, ax)

        ax.plot(dates[ts:],  df["Close"].values[ts:],
                color=CHART_COLORS["actual"],    lw=1.5, alpha=0.7, label="Actual")
        ax.plot(test_dates, actual_test,
                color=CHART_COLORS["actual"],    lw=1.5, alpha=0.7)
        ax.plot(test_dates, self._test_pred,
                color=CHART_COLORS["test_pred"], lw=1.8, label="Predicted")

        ax.axvline(dates[ts], color="#ffb74d", lw=1, ls="--", alpha=0.6)
        ax.set_title(f"{ticker} — Test-Set Predictions vs Actual  ({self._model_var.get()})",
                     color="#e0e0e0", fontsize=12, pad=6)
        ax.set_ylabel("Price (USD)", color="#e0e0e0")
        ax.set_xlabel("Date", color="#e0e0e0")
        ax.legend(fontsize=9, framealpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        self._pr_fig.tight_layout(pad=2.0)
        self._pr_fig._canvas.draw()

    def _plot_forecast(self) -> None:
        """Last 90 days of actual + future forecast with confidence shading."""
        if self._future_pred is None or self._df is None:
            return

        df      = self._df
        horizon = self._horizon.get()
        ticker  = self._ticker.get().upper()

        ctx          = min(90, len(df))
        hist_prices  = df["Close"].values[-ctx:]
        hist_dates   = df.index[-ctx:]
        future_dates = pd.bdate_range(df.index[-1], periods=horizon + 1, freq="B")[1:]

        ax = self._ax_fc
        ax.clear()
        apply_dark_theme(self._fc_fig, ax)

        ax.plot(hist_dates,   hist_prices,
                color=CHART_COLORS["actual"],   lw=1.8, label="Historical")
        ax.plot(future_dates, self._future_pred,
                color=CHART_COLORS["forecast"], lw=2.2,
                marker="o", markersize=3, label="Forecast")
        ax.fill_between(future_dates,
                        self._future_pred * 0.95,
                        self._future_pred * 1.05,
                        color=CHART_COLORS["forecast"], alpha=0.15, label="±5 % band")
        ax.axvline(df.index[-1], color="#555577", lw=1, ls="--")

        ax.set_title(f"{ticker} — {horizon}-Day Forecast  ({self._model_var.get()})",
                     color="#e0e0e0", fontsize=12, pad=6)
        ax.set_xlabel("Date",        color="#e0e0e0")
        ax.set_ylabel("Price (USD)", color="#e0e0e0")
        ax.legend(fontsize=9, framealpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

        self._fc_fig.tight_layout(pad=2.5)
        self._fc_fig._canvas.draw()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _update_metrics(self, m: dict) -> None:
        self._rmse_lbl.configure(text=f"RMSE  ${m['rmse']:.4f}")
        self._mae_lbl.configure( text=f"MAE   ${m['mae']:.4f}")
        self._r2_lbl.configure(  text=f"R²    {m['r2']:.4f}")

    def _update_info(self, info: dict) -> None:
        name = info.get("name", "—")
        if len(name) > 24:
            name = name[:22] + "…"
        self._info_name.configure(  text=f"Name    {name}")
        self._info_sector.configure(text=f"Sector  {info.get('sector', '—')}")
        self._info_cap.configure(   text=f"Cap     {format_large_number(info.get('market_cap', '—'))}")

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self._status.configure(text=msg))

    def _busy(self, on: bool) -> None:
        def _update():
            state = "disabled" if on else "normal"
            self._train_btn.configure(state=state)
            if on:
                self._progress.start()
            else:
                self._progress.stop()
                self._progress.set(0)
        self.after(0, _update)

    def _error(self, msg: str) -> None:
        dlg = ctk.CTkToplevel(self)
        dlg.title("Error"); dlg.geometry("420x160"); dlg.grab_set()
        ctk.CTkLabel(dlg, text=msg, font=FONT_BODY, wraplength=380).pack(padx=20, pady=(20, 10))
        ctk.CTkButton(dlg, text="OK", width=90, command=dlg.destroy).pack(pady=(0, 14))


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point."""
    App().mainloop()


if __name__ == "__main__":
    main()
