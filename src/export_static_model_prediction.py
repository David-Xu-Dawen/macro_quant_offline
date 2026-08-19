#!/usr/bin/env python3
"""把模型回测摘要导出为静态 HTML 可直接读取的 JSON。"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

from paths import MODEL_OUT, MODEL_RUN_DIR, MODEL_SRC, ensure_output_dirs

sys.path.insert(0, str(MODEL_SRC))

from model_summary import list_profiles, summarize  # noqa: E402

OUTPUT = MODEL_OUT / "model_prediction_static.json"


def static_figure_url(key: str, filename: str) -> str:
    output_dir = "output" if key == "balanced" else f"output/aggression_{key}"
    relative = f"output/model/{output_dir}/figures/{filename}"
    return "/" + quote(relative, safe="/")


def cleanup_model_intermediates() -> None:
    """静态摘要写出后只保留网页使用的模型图片。"""
    output_root = MODEL_RUN_DIR
    keep_dirs = {"figures", "aggression_conservative", "aggression_aggressive"}
    if not output_root.exists():
        return
    for child in output_root.iterdir():
        if child.is_file():
            child.unlink()
        elif child.name not in keep_dirs:
            shutil.rmtree(child)
    for profile_dir in (
        output_root / "aggression_conservative",
        output_root / "aggression_aggressive",
    ):
        if not profile_dir.exists():
            continue
        for child in profile_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.name != "figures":
                shutil.rmtree(child)


def main() -> None:
    ensure_output_dirs()
    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    profiles = list_profiles()
    for profile in profiles:
        key = profile["key"]
        if not profile["ready"]:
            continue
        try:
            data = summarize(aggression=key)
            for figure in data.get("figures", []):
                figure["url"] = static_figure_url(key, figure["file"])
            results[key] = data
        except Exception as exc:
            errors[key] = str(exc)

    payload = {
        "default": "balanced",
        "profiles": profiles,
        "results": results,
        "errors": errors,
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cleanup_model_intermediates()
    print(f"模型预测静态 JSON: {OUTPUT}（{len(results)} 个档位）")


if __name__ == "__main__":
    main()
