#!/usr/bin/env python3
"""用根目录 data.csv 完整重算离线宏观量化项目。

用法（在项目根目录）:
  python3 update_all_data.py
  python3 update_all_data.py --skip-model         # 调试时跳过较慢的模型重训
  python3 update_all_data.py --only corr,exposure # 只跑指定阶段
  python3 update_all_data.py --continue-on-error   # 某步失败后继续

阶段:
  data      从 Wind data.csv 重建 LF/HF 因子与资产
  corr      月/周频相关矩阵 JSON
  exposure  LASSO 因子暴露
  model     离线 LightGBM + BL 三档模型及静态页面数据
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

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable

STAGES = ("data", "corr", "exposure", "model")


@dataclass
class Step:
    name: str
    cwd: Path
    argv: list[str]
    stage: str


def _step(name: str, rel_dir: str, script: str, *args: str, stage: str) -> Step:
    cwd = ROOT / rel_dir if rel_dir else ROOT
    return Step(
        name=name,
        cwd=cwd,
        argv=[PYTHON, script, *args],
        stage=stage,
    )


def build_steps(args: argparse.Namespace) -> list[Step]:
    exposure_args = [
        "--bootstrap",
        str(args.bootstrap),
        "--alpha-scale",
        str(args.alpha_scale),
        "--rolling-window-weeks",
        str(args.rolling_window_weeks),
        "--sample-length-weeks",
        str(args.sample_length_weeks),
    ]
    return [
        _step("Wind CSV 因子与资产", "", "update_from_xlsx.py", "--data", str(ROOT / "data.csv"), stage="data"),
        _step("月频相关矩阵", "", "plot_macro_factor_corr.py", stage="corr"),
        _step("周频矩阵与静态警报", "", "plot_macro_hf_corr.py", stage="corr"),
        _step(
            "因子暴露",
            "factor exposure",
            "compute_factor_exposure.py",
            *exposure_args,
            stage="exposure",
        ),
        _step("离线模型预测", "model prediction", "run_all.py", stage="model"),
        _step("模型预测静态导出", "", "export_static_model_prediction.py", stage="model"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 data.csv 一键重算全部离线结果")
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
    parser.add_argument("--bootstrap", type=int, default=3000)
    parser.add_argument("--alpha-scale", type=float, default=0.5)
    parser.add_argument("--rolling-window-weeks", type=int, default=260)
    parser.add_argument("--sample-length-weeks", type=int, default=104)
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
    script_path = step.cwd / step.argv[1]
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
    subprocess.check_call([str(a) for a in step.argv], cwd=step.cwd, env=env)


def main() -> int:
    args = parse_args()
    stages = selected_stages(args)
    steps = [s for s in build_steps(args) if s.stage in stages]
    if not steps:
        print("没有可执行的步骤。")
        return 1

    print(
        f"模式: data.csv 纯离线；将执行 {len(steps)} 步，阶段: "
        f"{', '.join(s for s in STAGES if s in stages)}"
    )
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
        print("然后打开 http://127.0.0.1:8765/macro_factor_corr_interactive.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
