#!/usr/bin/env python3
"""一键重算：data → corr → exposure → model。

在项目根目录运行:
  python3 scripts/update_all_data.py
  python3 scripts/update_all_data.py --skip-model
  python3 scripts/update_all_data.py --only exposure
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "model"))

from load_wind_data import DEFAULT_DATA  # noqa: E402
from panel_config import load_panel_config, summarize_config  # noqa: E402
from paths import ensure_output_dirs  # noqa: E402

PYTHON = sys.executable
STAGES = ("data", "corr", "exposure", "model")


@dataclass
class Step:
    name: str
    cwd: Path
    argv: list[str]
    stage: str


def _step(name: str, script: Path, *args: str, stage: str, cwd: Path | None = None) -> Step:
    return Step(
        name=name,
        cwd=cwd or ROOT,
        argv=[PYTHON, str(script), *args],
        stage=stage,
    )


def build_steps(args: argparse.Namespace) -> list[Step]:
    cfg = load_panel_config()
    exp = cfg["exposure"]
    bootstrap = args.bootstrap if args.bootstrap is not None else int(exp["bootstrap_samples"])
    alpha_scale = args.alpha_scale if args.alpha_scale is not None else float(exp["alpha_scale"])
    rolling_window = (
        args.rolling_window_weeks
        if args.rolling_window_weeks is not None
        else int(exp["rolling_window_weeks"])
    )
    sample_length = (
        args.sample_length_weeks
        if args.sample_length_weeks is not None
        else int(exp["sample_length_weeks"])
    )
    exposure_args = [
        "--bootstrap",
        str(bootstrap),
        "--alpha-scale",
        str(alpha_scale),
        "--rolling-window-weeks",
        str(rolling_window),
        "--sample-length-weeks",
        str(sample_length),
    ]
    end_date = args.end_date or exp.get("end_date")
    if end_date:
        exposure_args.extend(["--end-date", str(end_date)])
    return [
        _step("Wind 本地数据因子", SRC / "update_from_xlsx.py", "--data", str(args.data), stage="data"),
        _step("高频因子日频导出", SRC / "export_hf_factor_daily.py", stage="data"),
        _step("月频相关矩阵", SRC / "plot_macro_factor_corr.py", stage="corr"),
        _step("周频矩阵与静态警报", SRC / "plot_macro_hf_corr.py", stage="corr"),
        _step("因子暴露", SRC / "compute_factor_exposure.py", *exposure_args, stage="exposure"),
        _step("离线模型预测", SRC / "model" / "run_all.py", stage="model", cwd=SRC / "model"),
        _step("模型预测静态导出", SRC / "export_static_model_prediction.py", stage="model"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 data/data1.xlsx 一键重算全部离线结果")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Wind 导出路径，默认 data/data1.xlsx",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help=f"只跑指定阶段，逗号分隔。可选: {','.join(STAGES)}",
    )
    parser.add_argument("--skip-corr", action="store_true")
    parser.add_argument("--skip-exposure", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="某一步失败后继续后续步骤",
    )
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--alpha-scale", type=float, default=None)
    parser.add_argument("--rolling-window-weeks", type=int, default=None)
    parser.add_argument("--sample-length-weeks", type=int, default=None)
    parser.add_argument("--end-date", type=str, default=None, help="暴露窗口结束周，默认读 config/panel_config.json")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的命令，不真正运行",
    )
    return parser.parse_args()


def selected_stages(args: argparse.Namespace) -> set[str]:
    if args.only.strip():
        stages = {s.strip() for s in args.only.split(",") if s.strip()}
        unknown = stages - set(STAGES)
        if unknown:
            raise SystemExit(f"未知阶段: {', '.join(sorted(unknown))}")
        return stages

    stages = set(STAGES)
    if args.skip_corr:
        stages.discard("corr")
    if args.skip_exposure:
        stages.discard("exposure")
    if args.skip_model:
        stages.discard("model")
    return stages


def run_step(step: Step, *, dry_run: bool) -> None:
    script_path = Path(step.argv[1])
    cmd = " ".join(str(a) for a in step.argv)
    print(f"\n=== [{step.stage}] {step.name} ===")
    print(f"$ cd {step.cwd}")
    print(f"$ {cmd}")
    if dry_run:
        return
    if not script_path.exists():
        raise FileNotFoundError(f"脚本不存在: {script_path}")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "macro_quant_matplotlib"))
    env["MACRO_QUANT_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), str(SRC / "model"), env.get("PYTHONPATH", "")]
    )
    subprocess.check_call([str(a) for a in step.argv], cwd=step.cwd, env=env)


def main() -> int:
    args = parse_args()
    ensure_output_dirs()
    stages = selected_stages(args)
    steps = [s for s in build_steps(args) if s.stage in stages]
    if not steps:
        print("没有可执行的步骤。")
        return 1

    print(
        f"模式: {Path(args.data).name} 纯离线；将执行 {len(steps)} 步，阶段: "
        f"{', '.join(s for s in STAGES if s in stages)}"
    )
    print(f"参数: {summarize_config()}")
    started = time.perf_counter()
    failures: list[str] = []
    for step in steps:
        try:
            run_step(step, dry_run=args.dry_run)
        except Exception as exc:
            msg = f"{step.name}: {exc}"
            failures.append(msg)
            print(f"\n!! 失败: {msg}")
            if not args.continue_on_error:
                print("已中止。可用 --continue-on-error 继续后续步骤。")
                return 1

    elapsed = time.perf_counter() - started
    print("\n" + "=" * 60)
    if args.dry_run:
        print("Dry-run 完成，未实际执行。")
    elif failures:
        print(f"完成（有失败 {len(failures)} 步），耗时 {elapsed:.1f}s")
        for item in failures:
            print(f"  - {item}")
        return 1
    else:
        print(f"全部完成，耗时 {elapsed:.1f}s")
        print("查看页面: python3 -m http.server 8765")
        print("然后打开 http://127.0.0.1:8765/web/macro_factor_corr_interactive.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
