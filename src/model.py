"""
model.py
--------
Stock price prediction using scikit-learn ensemble regressors.

Two models are available:
    - RandomForestRegressor  : robust, fast, good out-of-the-box
    - GradientBoostingRegressor : higher accuracy, slightly slower to train

Both are trained on a flat feature matrix (technical indicators + lag features)
and predict the next day's closing price.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Available model choices exposed to the GUI
MODEL_OPTIONS = {
    "Random Forest":       "rf",
    "Gradient Boosting":   "gb",
}


class StockPredictor:
    """
    Wrapper around scikit-learn regressors for next-day stock price prediction.

    Attributes:
        model_type: 'rf' for Random Forest, 'gb' for Gradient Boosting.
        model:      Fitted sklearn estimator (None until trained).
    """

    def __init__(self, model_type: str = "rf"):
        """
        Args:
            model_type: 'rf' or 'gb'.
        """
        if model_type not in ("rf", "gb"):
            raise ValueError(f"model_type must be 'rf' or 'gb', got '{model_type}'")
        self.model_type = model_type
        self.model      = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Instantiate the estimator with sensible defaults."""
        if self.model_type == "rf":
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1,      # use all CPU cores
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42,
            )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the model on training data.

        Args:
            X_train: Feature matrix, shape (n_samples, n_features).
            y_train: Target vector (next-day Close prices), shape (n_samples,).
        """
        if self.model is None:
            self.build()
        self.model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate price predictions for feature rows.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted prices, shape (n_samples,).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.model.predict(X)

    def predict_future(
        self,
        last_row: np.ndarray,
        n_steps: int,
        preprocessor,
        df_clean,
    ) -> np.ndarray:
        """
        Autoregressively forecast n_steps days ahead.

        At each step the predicted price is fed back as the next row's
        Close (and Lag) features, then re-scaled before prediction.

        Args:
            last_row:     The last raw (unscaled) feature row from df_clean,
                          shape (n_features,).
            n_steps:      Number of future trading days to forecast.
            preprocessor: StockPreprocessor with fitted scaler.
            df_clean:     Cleaned DataFrame (used for column ordering).

        Returns:
            Predicted Close prices for the next n_steps days.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        feature_cols = preprocessor.feature_cols
        current_row  = last_row.copy()
        predictions  = []

        for _ in range(n_steps):
            # Scale and predict
            x_scaled   = preprocessor.transform(current_row.reshape(1, -1))
            next_price = self.model.predict(x_scaled)[0]
            predictions.append(next_price)

            # Update lag features and Close for next step
            next_row = current_row.copy()

            close_idx = feature_cols.index("Close")
            # Shift lags: Lag_5←Lag_4, Lag_4←Lag_3, …, Lag_1←Close
            for lag in range(5, 1, -1):
                if f"Lag_{lag}" in feature_cols and f"Lag_{lag-1}" in feature_cols:
                    next_row[feature_cols.index(f"Lag_{lag}")] = \
                        current_row[feature_cols.index(f"Lag_{lag-1}")]
            if "Lag_1" in feature_cols:
                next_row[feature_cols.index("Lag_1")] = current_row[close_idx]

            # Update Close to the just-predicted price
            next_row[close_idx] = next_price

            current_row = next_row

        return np.array(predictions)

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute regression metrics in the original price scale.

        Args:
            y_true: Actual next-day Close prices.
            y_pred: Predicted next-day Close prices.

        Returns:
            Dict with keys 'rmse', 'mae', 'r2'.
        """
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        r2   = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str = "models/predictor.joblib") -> None:
        """Save the fitted model to disk with joblib."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        os.makedirs(os.path.dirname(path) or "models", exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str = "models/predictor.joblib") -> None:
        """Load a previously saved model from disk."""
        self.model = joblib.load(path)
