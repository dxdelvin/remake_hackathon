"""
ml_model.py
30% component of the hybrid decision engine.

Dual-purpose MLP:
  1. Energy spike detection — learned from power_vs_baseline deviations
  2. Action classification  — 3 classes: pause_live_view, optimize_tile_scan_settings, no_action

Architecture: MLPClassifier(128, 64, 32) trained on aggregated segments from S1-S10.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.data_processor import PHASE_BASELINES_W

# ── Feature list (order matters — must be consistent between train + predict) ─
FEATURE_COLS = [
    # Core boolean share features
    "live_view_enabled_share",
    "user_interacting_share",
    "monitoring_required_share",
    "continuous_acquisition_display_share",
    "tile_scan_enabled_share",
    "experiment_running_share",
    # Inactivity
    "median_seconds_since_last_ui_interaction",
    # Tile scan geometry
    "tile_overlap_pct_mean",
    "total_tiles_mean",
    "planned_scan_area_mm2_mean",
    # Power & compute
    "perf_gpu_power_w_mean",
    "perf_gpu_usage_pct_mean",
    "perf_cpu_pct_mean",
    "processing_items_in_flight_mean",
    "perf_disk_write_mb_s_mean",
    "perf_incoming_data_mb_s_mean",
    "estimated_system_power_w_mean",
    "power_vs_baseline",
    # Direct illumination signal (resolves S13 live_view_flag=0 ambiguity)
    "camera_light_usage_index_pct_mean",
    "preview_resolution_pct_mean",
    # Categorical encodings
    "phase_encoded",
    "quality_encoded",
]

PHASE_ORDER = ["idle", "processing", "live_view_monitoring", "tile_scan_acquisition"]
QUALITY_ORDER = ["low", "medium", "high"]
SPIKE_THRESHOLD_W = 20.0  # watts above phase baseline → flagged as energy spike

ACTION_CLASSES = ["no_action", "optimize_tile_scan_settings", "pause_live_view"]


def _encode_phase(phase: str) -> int:
    phase = str(phase).strip().lower()
    try:
        return PHASE_ORDER.index(phase)
    except ValueError:
        return 0


def _encode_quality(quality: str) -> int:
    quality = str(quality).strip().lower()
    try:
        return QUALITY_ORDER.index(quality)
    except ValueError:
        return 1  # default medium


def _build_feature_row(segment: dict) -> np.ndarray:
    """Convert a segment dict to a 1D feature array."""
    row = {col: float(segment.get(col, 0.0)) for col in FEATURE_COLS
           if col not in ("phase_encoded", "quality_encoded")}
    row["phase_encoded"] = float(_encode_phase(segment.get("phase_name", "idle")))
    row["quality_encoded"] = float(_encode_quality(segment.get("quality_constraint_mode", "medium")))

    # Phase-context correction: keep the heuristic as a fallback for scenarios
    # where both live_view_enabled_share AND camera_light_usage_index_pct_mean
    # are zero but the phase is live_view_monitoring (physically impossible).
    phase_name = str(segment.get("phase_name", "")).strip().lower()
    if row["live_view_enabled_share"] < 0.05 and row["camera_light_usage_index_pct_mean"] < 5.0:
        if phase_name == "live_view_monitoring":
            row["live_view_enabled_share"] = 1.0
        elif phase_name in ("idle", "tile_scan_acquisition"):
            row["live_view_enabled_share"] = 0.5

    return np.array([row[col] for col in FEATURE_COLS], dtype=float)


class EnergyMLP:
    """
    Trained once at server startup from S1-S10 aggregated segments.
    Thread-safe for read-only inference after training.
    """

    def __init__(self):
        self._scaler = StandardScaler()
        self._label_encoder = LabelEncoder()
        self._model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
        self._trained = False
        self.accuracy: float = 0.0
        self.f1_macro: float = 0.0
        self.class_report: dict = {}

    @property
    def is_trained(self) -> bool:
        return self._trained

    def train(self, segments_df: pd.DataFrame) -> dict:
        """
        Fit scaler + MLP on labelled segment rows.

        Parameters
        ----------
        segments_df : DataFrame with FEATURE_COLS-compatible columns + 'label' column.

        Returns
        -------
        dict with keys: accuracy, f1_macro, n_train, classes
        """
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split

        df = segments_df.dropna(subset=["label"]).copy()
        if len(df) < 10:
            raise ValueError(f"Too few labelled training samples: {len(df)}")

        # Build feature matrix
        df["phase_encoded"] = df["phase_name"].apply(_encode_phase).astype(float)
        df["quality_encoded"] = df["quality_constraint_mode"].apply(_encode_quality).astype(float)

        # Fill any missing feature columns with 0
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        # Encode labels before splitting
        self._label_encoder.fit(ACTION_CLASSES)
        df["_y"] = df["label"].astype(str).str.strip().apply(
            lambda v: v if v in ACTION_CLASSES else "no_action"
        )

        # Per-scenario 70/30 split — each scenario contributes 30% to validation
        # so the model is never tested on a segment from the same scenario block
        # it was trained on.  Falls back to random 80/20 if source info is missing.
        if "_source_file" in df.columns and df["_source_file"].nunique() > 1:
            train_frames, val_frames = [], []
            for _, scenario_df in df.groupby("_source_file"):
                try:
                    t, v = train_test_split(
                        scenario_df, test_size=0.30, random_state=42,
                        stratify=scenario_df["_y"] if scenario_df["_y"].nunique() > 1 else None,
                    )
                except ValueError:
                    t, v = train_test_split(scenario_df, test_size=0.30, random_state=42)
                train_frames.append(t)
                val_frames.append(v)
            train_df = pd.concat(train_frames, ignore_index=True)
            val_df   = pd.concat(val_frames,   ignore_index=True)
            split_method = "per-scenario 70/30"
        else:
            train_df, val_df = train_test_split(
                df, test_size=0.20, random_state=42,
                stratify=df["_y"] if df["_y"].nunique() > 1 else None,
            )
            split_method = "random 80/20"

        X_train = train_df[FEATURE_COLS].values.astype(float)
        X_val   = val_df[FEATURE_COLS].values.astype(float)
        y_train = self._label_encoder.transform(train_df["_y"])
        y_val   = self._label_encoder.transform(val_df["_y"])

        self._scaler.fit(X_train)
        X_train_s = self._scaler.transform(X_train)
        X_val_s   = self._scaler.transform(X_val)

        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
        self._model.fit(X_train_s, y_train, sample_weight=sample_weights)

        y_pred = self._model.predict(X_val_s)
        self.accuracy = float(accuracy_score(y_val, y_pred))
        self.f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

        self._trained = True
        return {
            "accuracy": round(self.accuracy, 4),
            "f1_macro": round(self.f1_macro, 4),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "split_method": split_method,
            "classes": list(self._label_encoder.classes_),
        }

    def predict(self, segment: dict) -> dict:
        """
        Run inference on a single segment.

        Returns
        -------
        dict with keys:
          action          : str  — predicted class
          probabilities   : dict — {action: probability}
          is_energy_spike : bool — True if power_vs_baseline > SPIKE_THRESHOLD_W
          spike_magnitude_w: float
          explanation     : str  — human-readable reason
        """
        if not self._trained:
            return {
                "action": "no_action",
                "probabilities": {a: 0.0 for a in ACTION_CLASSES},
                "is_energy_spike": False,
                "spike_magnitude_w": 0.0,
                "explanation": "Model not yet trained.",
            }

        x = _build_feature_row(segment).reshape(1, -1)
        x_scaled = self._scaler.transform(x)

        proba = self._model.predict_proba(x_scaled)[0]
        classes = self._label_encoder.classes_
        prob_dict = {str(cls): float(p) for cls, p in zip(classes, proba)}

        # Ensure all 3 classes present
        for a in ACTION_CLASSES:
            prob_dict.setdefault(a, 0.0)

        predicted_idx = int(np.argmax(proba))
        action = str(classes[predicted_idx])

        power_vs_baseline = float(segment.get("power_vs_baseline", 0.0))
        is_spike = power_vs_baseline > SPIKE_THRESHOLD_W
        spike_magnitude = max(0.0, power_vs_baseline)

        # Build explanation
        phase = str(segment.get("phase_name", "unknown"))
        baseline = PHASE_BASELINES_W.get(phase, 170.0)
        avg_power = float(segment.get("estimated_system_power_w_mean", 0.0))

        # Identify top contributing feature for explanation text
        feature_values = _build_feature_row(segment)
        top_feat_idx = int(np.argmax(np.abs(feature_values)))
        top_feat_name = FEATURE_COLS[top_feat_idx].replace("_", " ")

        if is_spike:
            explanation = (
                f"Power {avg_power:.0f}W is {spike_magnitude:.0f}W above "
                f"{phase.replace('_', ' ')} baseline ({baseline:.0f}W) — "
                f"highest signal: {top_feat_name}"
            )
        else:
            explanation = (
                f"Operating within normal range for {phase.replace('_', ' ')} "
                f"(baseline {baseline:.0f}W)"
            )

        return {
            "action": action,
            "probabilities": prob_dict,
            "is_energy_spike": is_spike,
            "spike_magnitude_w": round(spike_magnitude, 1),
            "explanation": explanation,
        }


    def evaluate(self, segments_df: pd.DataFrame) -> dict:
        """
        Evaluate the trained model on a fully held-out scenario DataFrame.
        Use this to measure true generalisation to unseen scenarios (e.g. S13).

        Returns
        -------
        dict with keys: accuracy, f1_macro, n_samples, report (text)
        """
        from sklearn.metrics import accuracy_score, classification_report, f1_score

        if not self._trained:
            raise RuntimeError("Model not trained — call train() first")

        df = segments_df.dropna(subset=["label"]).copy()
        df["phase_encoded"] = df["phase_name"].apply(_encode_phase).astype(float)
        df["quality_encoded"] = df["quality_constraint_mode"].apply(_encode_quality).astype(float)

        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        y_raw = df["label"].astype(str).str.strip().apply(
            lambda v: v if v in ACTION_CLASSES else "no_action"
        )
        y_true = self._label_encoder.transform(y_raw)

        X = df[FEATURE_COLS].values.astype(float)
        X_scaled = self._scaler.transform(X)
        y_pred = self._model.predict(X_scaled)

        acc = float(accuracy_score(y_true, y_pred))
        f1  = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        report = classification_report(
            y_true, y_pred,
            target_names=self._label_encoder.classes_,
            zero_division=0,
        )
        return {
            "accuracy": round(acc, 4),
            "f1_macro": round(f1, 4),
            "n_samples": len(y_true),
            "report": report,
        }


# Singleton — instantiated once, trained at startup
model = EnergyMLP()
