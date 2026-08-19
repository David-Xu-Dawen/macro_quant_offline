from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from panel_config import sync_asset_factor_mask  # noqa: E402


FACTORS = ["增长因子", "通胀因子", "利率因子", "信用因子", "汇率因子", "地缘因子", "流动性因子"]


def _cfg(mask: dict) -> dict:
    return {
        "heatmap": {"include_factors": ["增长因子"]},
        "exposure": {
            "factors": FACTORS,
            "include_factors": FACTORS,
            "exclude_factors": [],
            "credit_only_for_bonds": True,
            "bond_assets": ["中债国债"],
            "bond_name_markers": ["债", "转债"],
            "asset_factor_mask": mask,
        },
    }


class SyncAssetFactorMaskTests(unittest.TestCase):
    def test_adds_and_removes_to_match_excel_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel_config.json"
            path.write_text(
                json.dumps(
                    _cfg(
                        {
                            "上证50": {f: 1 for f in FACTORS} | {"信用因子": 0},
                            "旧资产": {f: 1 for f in FACTORS},
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            mask = sync_asset_factor_mask(["上证50", "中债国债", "新易盛(300502)"], path)
            self.assertEqual(list(mask), ["上证50", "中债国债", "新易盛(300502)"])
            self.assertEqual(mask["上证50"]["信用因子"], 0)
            self.assertEqual(mask["中债国债"]["信用因子"], 1)
            self.assertEqual(mask["新易盛(300502)"]["信用因子"], 0)
            self.assertEqual(mask["新易盛(300502)"]["增长因子"], 1)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(list(saved["exposure"]["asset_factor_mask"]), list(mask))
            self.assertNotIn("旧资产", saved["exposure"]["asset_factor_mask"])

    def test_keeps_hand_edited_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "panel_config.json"
            custom = {f: 1 for f in FACTORS}
            custom["地缘因子"] = 0
            path.write_text(json.dumps(_cfg({"沪金": custom}), ensure_ascii=False, indent=2), encoding="utf-8")
            mask = sync_asset_factor_mask(["沪金"], path)
            self.assertEqual(mask["沪金"]["地缘因子"], 0)
            self.assertEqual(mask["沪金"]["增长因子"], 1)


if __name__ == "__main__":
    unittest.main()
