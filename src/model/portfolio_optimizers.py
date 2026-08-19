"""尾部风险组合优化器。CVaR 失败时由调用方执行稳健回退。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OptimizerResult:
    weights: pd.Series | None
    optimizer_used: str
    status: str
    cvar_95: float | None = None


def historical_cvar(returns: pd.Series, alpha: float = 0.95) -> float:
    """历史损失 CVaR；返回正数损失幅度。"""
    values = returns.dropna().to_numpy(dtype=float)
    if not len(values):
        return float("nan")
    losses = -values
    threshold = np.quantile(losses, alpha)
    tail = losses[losses >= threshold]
    return float(tail.mean()) if len(tail) else float(threshold)


def cvar_weights(
    mu: pd.Series,
    return_scenarios: pd.DataFrame,
    *,
    alpha: float,
    risk_aversion: float,
    w_min: float,
    w_max: float,
) -> OptimizerResult:
    """Rockafellar-Uryasev CVaR 优化：max μ'w - λ CVaRα。"""
    try:
        import cvxpy as cp
    except ImportError:
        return OptimizerResult(None, "cvar", "cvxpy_not_installed")

    assets = list(mu.index)
    scenarios = return_scenarios.reindex(columns=assets).dropna(how="any")
    if len(scenarios) < 40:
        return OptimizerResult(None, "cvar", "insufficient_scenarios")

    r = scenarios.to_numpy(dtype=float)
    n_scenarios, n_assets = r.shape
    w = cp.Variable(n_assets)
    var = cp.Variable()
    excess_loss = cp.Variable(n_scenarios, nonneg=True)
    losses = -(r @ w)
    cvar_daily = var + cp.sum(excess_loss) / ((1.0 - alpha) * n_scenarios)
    cvar_annualized = cvar_daily * np.sqrt(252.0)
    objective = cp.Maximize(mu.to_numpy(dtype=float) @ w - risk_aversion * cvar_annualized)
    constraints = [
        cp.sum(w) == 1.0,
        w >= w_min,
        w <= w_max,
        excess_loss >= losses - var,
    ]
    problem = cp.Problem(objective, constraints)
    try:
        for solver in ("CLARABEL", "ECOS", "SCS"):
            if solver not in cp.installed_solvers():
                continue
            try:
                problem.solve(solver=solver, warm_start=True, verbose=False)
            except Exception:
                continue
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and w.value is not None:
                break
    except Exception as exc:
        return OptimizerResult(None, "cvar", f"solver_error:{type(exc).__name__}")

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or w.value is None:
        return OptimizerResult(None, "cvar", f"solver_status:{problem.status}")

    values = np.asarray(w.value, dtype=float)
    values = np.clip(values, w_min, w_max)
    if not np.isfinite(values).all() or values.sum() <= 0:
        return OptimizerResult(None, "cvar", "invalid_solution")
    values /= values.sum()
    weights = pd.Series(values, index=assets, name="weight")
    portfolio_returns = scenarios @ weights
    return OptimizerResult(
        weights,
        "cvar",
        str(problem.status),
        historical_cvar(portfolio_returns, alpha=alpha),
    )
