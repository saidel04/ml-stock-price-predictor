# ML Stock Price Predictor

> A desktop application that uses scikit-learn ensemble models to predict stock prices, with technical indicator analysis, interactive dark-themed charts, and a clean CustomTkinter GUI.

---

## Screenshots

| Overview Tab | Predictions Tab | Forecast Tab |
|:---:|:---:|:---:|
| ![Overview](screenshots/overview.png) | ![Predictions](screenshots/predictions.png) | ![Forecast](screenshots/forecast.png) |

> _Screenshots will appear after your first run._

---

## Features

- **Live data** — fetches OHLCV history for any ticker via `yfinance`, with local CSV caching to avoid redundant downloads
- **Feature engineering** — computes 15+ features: SMA (20/50), EMA (12/26), RSI, MACD, Bollinger Bands, daily returns, and 5-day lag features
- **Two model options** — switch between Random Forest and Gradient Boosting from a dropdown in the UI
- **Evaluation metrics** — RMSE, MAE, and R² displayed after training
- **Three interactive chart tabs**
  - _Overview_: price with SMA/Bollinger overlay, volume bars, RSI
  - _Predictions_: test-set predicted vs actual closing prices
  - _Forecast_: autoregressive N-day price forecast with ±5% confidence band
- **Adjustable forecast horizon** — 5 to 60 trading days via slider
- **Non-blocking** — all ML work runs in background daemon threads; the GUI never freezes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| GUI | CustomTkinter |
| Data | yfinance, pandas |
| ML | scikit-learn (Random Forest, Gradient Boosting) |
| Features | NumPy, MinMaxScaler |
| Charts | Matplotlib (TkAgg backend) |
| Model saving | joblib |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ml-stock-predictor.git
cd ml-stock-predictor
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

### Workflow

1. **Enter a ticker** (e.g. `AAPL`, `TSLA`, `NVDA`, `BTC-USD`)
2. **Set the date range** — at least 1 year recommended; 3–5 years gives better results
3. **Pick a model** — Random Forest or Gradient Boosting
4. **Set the forecast horizon** with the slider (5–60 trading days)
5. **Click "Train Model"** — data is fetched, features are computed, and the model trains in seconds
6. Switch to **Predictions** to review test-set accuracy
7. **Click "Generate Forecast"** — the Forecast tab shows the predicted trajectory

### Tips

- Start with **AAPL** or **MSFT** for well-known, liquid stocks
- **Gradient Boosting** typically gives slightly lower RMSE; **Random Forest** trains faster
- Use at least **3 years** of data for meaningful results
- The model uses 80% of data for training and 20% for evaluation (chronological split — no shuffling)

---

## Project Structure

```
ml-stock-predictor/
├── main.py               # Entry point
├── requirements.txt      # Python dependencies
├── .gitignore
├── LICENSE               # MIT
├── README.md
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py   # Yahoo Finance download + CSV caching
│   ├── preprocessor.py   # Technical indicators + lag features + scaling
│   ├── model.py          # scikit-learn regressors, evaluation, joblib persistence
│   ├── gui.py            # CustomTkinter application window
│   └── utils.py          # Background workers, chart theme, helpers
├── data/                 # Auto-created — cached CSV files
├── models/               # Auto-created — saved model files
└── screenshots/          # Placeholder for README images
```

---

## How the Model Works

Each row in the training data represents one trading day. The target is the **next day's closing price**.

**Input features per row:**
- Raw OHLCV columns (Open, High, Low, Close, Volume)
- SMA 20, SMA 50, EMA 12, EMA 26
- RSI (14-period)
- MACD line, MACD signal
- Bollinger Band upper/lower
- Daily return (% change)
- Lag features: Close price from 1–5 days ago

**Training:**
- Features are scaled to [0, 1] with MinMaxScaler (fit on training data only)
- Chronological 80/20 train/test split — no shuffling to preserve time ordering
- Random Forest: 200 trees, max depth 10
- Gradient Boosting: 200 estimators, learning rate 0.05, max depth 5

**Forecasting:**
- Autoregressively feeds each predicted price back as the next row's lag features
- Runs for the number of days set by the forecast horizon slider

---

## Limitations & Disclaimer

> **This project is for educational and portfolio purposes only.**
> Stock prices are influenced by countless unpredictable factors (news, macro events, sentiment). A model trained on historical price data **cannot reliably predict future prices** and should **never** be used for real investment decisions.

---

## What I Learned

- **Feature engineering for finance** — implementing RSI, MACD, and Bollinger Bands from scratch using vectorised pandas, and building lag features for autoregressive forecasting
- **Preventing data leakage** — fitting the MinMaxScaler on the training split only and using a strict chronological train/test split
- **GUI + ML integration** — keeping the UI responsive by offloading all blocking work to daemon threads and dispatching GUI updates back through `after()` callbacks
- **Matplotlib in Tkinter** — embedding interactive figures with NavigationToolbar2Tk and redrawing canvases dynamically without flickering

---

## License

MIT — see [LICENSE](LICENSE).
