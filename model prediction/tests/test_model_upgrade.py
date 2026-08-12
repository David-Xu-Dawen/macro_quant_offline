from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from black_litterman import build_view_confidence, decay_view_history  # noqa: E402
from config import TRADE_UNIVERSE  # noqa: E402
from macro_features import add_labels, feature_columns  # noqa: E402
from portfolio_optimizers import cvar_weights, historical_cvar  # noqa: E402
from prepare_local_raw_data import (  # noqa: E402
    RAW_COLUMNS,
    SOURCE_COLUMNS,
    prepare_local_raw_data,
)


class OfflineRawDataTests(unittest.TestCase):
    def test_prepare_local_raw_data_writes_model_schema(self):
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        source = pd.DataFrame({"date": dates})
        for index, asset in enumerate(TRADE_UNIVERSE):
            source[SOURCE_COLUMNS[asset]] = [100 + index, 101 + index, 102 + index]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            source_path = temp / "combined_close.csv"
            raw_dir = temp / "raw"
            source.to_csv(source_path, index=False)

            summary = prepare_local_raw_data(source_path, raw_dir)

            self.assertEqual(len(summary), len(TRADE_UNIVERSE))
            for asset in TRADE_UNIVERSE:
                raw = pd.read_csv(raw_dir / f"{asset}.csv")
                self.assertEqual(raw.columns.tolist(), RAW_COLUMNS)
                self.assertEqual(raw["asset"].unique().tolist(), [asset])
                pd.testing.assert_series_equal(
                    raw["close"], raw["open"], check_names=False
                )
                pd.testing.assert_series_equal(
                    raw["close"], raw["high"], check_names=False
                )
                pd.testing.assert_series_equal(
                    raw["close"], raw["low"], check_names=False
                )


class DynamicViewTests(unittest.TestCase):
    def test_disagreement_reduces_confidence(self):
        scores = pd.Series({"a": 0.9, "b": 0.5, "c": 0.1})
        low = build_view_confidence(
            scores,
            disagreement=pd.Series({"a": 0.0, "b": 0.0, "c": 0.0}),
        )
        high = build_view_confidence(
            scores,
            disagreement=pd.Series({"a": 1.0, "b": 1.0, "c": 1.0}),
        )
        self.assertTrue((low > high).all())
        self.assertTrue(low.between(0.05, 0.80).all())

    def test_view_decay_weights_recent_signal_more(self):
        old = pd.Series({"a": 0.0, "b": 1.0})
        new = pd.Series({"a": 1.0, "b": 0.0})
        result = decay_view_history(
            [(pd.Timestamp("2024-01-01"), old), (pd.Timestamp("2024-01-21"), new)],
            pd.Timestamp("2024-01-21"),
        )
        self.assertGreater(result["a"], result["b"])


class CVaRTests(unittest.TestCase):
    def test_historical_cvar_is_positive_loss(self):
        returns = pd.Series([0.01, -0.02, -0.05, 0.005, -0.01])
        self.assertGreater(historical_cvar(returns, 0.8), 0)

    def test_cvar_weights_respect_constraints(self):
        rng = np.random.default_rng(42)
        scenarios = pd.DataFrame(
            rng.normal(0.0002, 0.01, size=(180, 4)),
            columns=list("abcd"),
        )
        mu = pd.Series([0.04, 0.03, 0.02, 0.01], index=list("abcd"))
        result = cvar_weights(
            mu,
            scenarios,
            alpha=0.95,
            risk_aversion=4.0,
            w_min=0.0,
            w_max=0.35,
        )
        if result.status == "cvxpy_not_installed":
            self.skipTest("cvxpy not installed")
        self.assertIsNotNone(result.weights)
        assert result.weights is not None
        self.assertAlmostEqual(float(result.weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(result.weights.min()), -1e-8)
        self.assertLessEqual(float(result.weights.max()), 0.350001)


class LabelExperimentTests(unittest.TestCase):
    def test_risk_adjusted_label_uses_target_for_ranking(self):
        dates = pd.date_range("2024-01-01", periods=12, freq="B")
        close = pd.DataFrame(
            {
                "a": 100 * np.cumprod(1 + np.linspace(0.001, 0.012, len(dates))),
                "bond_gov": 100 * np.cumprod(1 + np.linspace(0.0002, 0.0004, len(dates))),
                "c": 100 * np.cumprod(1 + np.array([0.02, -0.015] * 6)),
            },
            index=dates,
        )
        panel = pd.MultiIndex.from_product(
            [dates, close.columns],
            names=["date", "asset"],
        ).to_frame(index=False)
        panel["cn_name"] = panel["asset"]
        result = add_labels(
            panel,
            close,
            forward_days=3,
            mode="ranking",
            target="risk_adjusted",
        )
        self.assertTrue(result["y_risk_adjusted"].notna().all())
        expected = result.groupby("date")["y_target"].rank(ascending=True, method="first") - 1
        pd.testing.assert_series_equal(
            result["y"].astype(float).reset_index(drop=True),
            expected.astype(float).reset_index(drop=True),
            check_names=False,
        )

    def test_future_label_columns_are_not_features(self):
        panel = pd.DataFrame(
            {
                "signal": [1.0],
                "asset_id": [0],
                "y_holding_vol": [0.1],
                "y_risk_adjusted": [0.2],
                "y_target": [0.2],
            }
        )
        self.assertEqual(set(feature_columns(panel)), {"signal", "asset_id"})


if __name__ == "__main__":
    unittest.main()
