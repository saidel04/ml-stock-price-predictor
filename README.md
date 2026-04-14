# ML Stock Price Predictor

> A desktop application that uses a dual-layer LSTM neural network to predict stock prices, complete with technical indicator analysis, interactive charts, and a polished dark-themed GUI.

---

## Screenshots

| Overview Tab | Train / Test Tab | Forecast Tab |
|:---:|:---:|:---:|
| ![Overview](screenshots/overview.png) | ![Train Test](screenshots/train_test.png) | ![Forecast](screenshots/forecast.png) |

> _Screenshots will appear after your first run._

---

## Features

- **Live data** — fetches OHLCV history for any ticker via `yfinance`, with local CSV caching to avoid redundant downloads
- **Feature engineering** — automatically computes SMA (20/50), EMA (12/26), RSI, MACD, Bollinger Bands, daily returns, and volume MA
- **LSTM model** — two-layer LSTM (128 → 64 units) with Dropout regularisation, trained with early stopping and learning-rate scheduling
- **Evaluation metrics** — RMSE and MAE displayed in real time after training
- **Three interactive chart tabs**
  - _Overview_: price with SMA/Bollinger overlay, volume, RSI
  - _Train / Test_: predictions vs actual with train/test split visualised, plus loss curves
  - _Forecast_: future N-day forecast with confidence band
- **Fully adjustable** — lookback window, epochs, and forecast horizon are all slider-controlled in the UI
- **Non-blocking** — all ML work runs in background daemon threads; the GUI never freezes

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| GUI | CustomTkinter |
| Data | yfinance, pandas |
| ML | TensorFlow / Keras (LSTM) |
| Features | scikit-learn (MinMaxScaler), NumPy |
| Charts | Matplotlib (TkAgg backend) |

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

> **Note:** TensorFlow requires Python 3.8–3.11. On Apple Silicon use `tensorflow-macos` instead.

---

## Usage

```bash
python main.py
```

### Workflow

1. **Enter a ticker** (e.g. `AAPL`, `TSLA`, `NVDA`, `BTC-USD`)
2. **Set the date range** — at least 1 year is recommended; 3–5 years gives the best results
3. **Adjust model settings** with the sliders (lookback window, epochs, forecast horizon)
4. **Click "Train Model"** — the status bar reports live epoch progress
5. Switch to the **Train / Test** tab to review prediction accuracy and loss curves
6. **Click "Generate Forecast"** — the Forecast tab shows the predicted trajectory for the next N days

### Tips

- Start with **AAPL** or **MSFT** for stable, well-studied stocks
- A **60-day lookback** window is the default and works well for most tickers
- Use **100+ epochs** with early stopping for better convergence
- The model uses ~80 % of data for training and reserves 20 % for evaluation

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
│   ├── data_fetcher.py   # Yahoo Finance download + caching
│   ├── preprocessor.py   # Feature engineering + LSTM sequence builder
│   ├── model.py          # LSTM architecture, training, inference
│   ├── gui.py            # CustomTkinter application window
│   └── utils.py          # Background workers, chart theme, helpers
├── data/                 # Auto-created — cached CSV files
├── models/               # Auto-created — saved Keras checkpoints
└── screenshots/          # Placeholder for README images
```

---

## Model Architecture

```
Input  (sequence_length × n_features)
  └─► LSTM(128, return_sequences=True)
        └─► Dropout(0.20)
              └─► LSTM(64, return_sequences=False)
                    └─► Dropout(0.20)
                          └─► Dense(32, ReLU)
                                └─► Dense(1)   ← predicted Close price
```

- **Optimiser:** Adam (lr = 1e-3, with ReduceLROnPlateau)
- **Loss:** Mean Squared Error
- **Callbacks:** EarlyStopping (patience 15), ReduceLROnPlateau (patience 8), ModelCheckpoint

---

## Limitations & Disclaimer

> **This project is for educational and portfolio purposes only.**
> Stock prices are influenced by countless unpredictable factors (news, macro events, sentiment). An LSTM trained on historical price data **cannot reliably predict future prices** and should **never** be used for real investment decisions.

---

## What I Learned

Building this project gave me hands-on experience with:

- **Time-series ML** — structuring sliding-window sequences for LSTM input, preventing data leakage by fitting scalers only on the training portion, and using autoregressive inference for multi-step forecasting
- **Feature engineering for finance** — implementing RSI, MACD, and Bollinger Bands from scratch using vectorised pandas operations
- **GUI + ML integration** — keeping the UI responsive while TensorFlow trains by offloading all blocking work to daemon threads and dispatching GUI updates through `after()` callbacks
- **Keras best practices** — EarlyStopping with weight restoration, ReduceLROnPlateau, and ModelCheckpoint for robust training loops
- **Matplotlib in Tkinter** — embedding interactive figures with NavigationToolbar2Tk and dynamically redrawing canvases without flickering

---

## License

MIT — see [LICENSE](LICENSE).
