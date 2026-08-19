"""完全离线运行宏观配置框架及三个风险档位。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(script: str, *args: str):
    cmd = [sys.executable, str(ROOT / script), *args]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


if __name__ == "__main__":
    # 每次都从仓库内的统一价格面板重建 raw，避免旧文件与网络回退。
    run("prepare_local_raw_data.py")
    run("run_macro_strategy.py", "--forward-days", "20", "--label-mode", "ranking")
    run("plot_results.py")
    run("run_aggression_profiles.py", "--only", "conservative,aggressive")
    print(
        "\nDone. See output/, output/aggression_conservative/, "
        "output/aggression_aggressive/, and models/macro_lgbm_bl.joblib"
    )
