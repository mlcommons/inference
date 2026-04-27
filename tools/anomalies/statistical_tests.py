from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config import ANOMALY_Z_THRESHOLD, MIN_SAMPLES


@dataclass
class AnomalyResult:
    model: str
    scenario: str
    accelerator: str
    new_value: float
    total_accelerators: int
    normalized_value: float
    mean: Optional[float]
    std: Optional[float]
    z_score: Optional[float]
    p_value: Optional[float]
    is_anomaly: bool
    reason: str
    historical_values: list


def test_anomaly(
    historical_df: pd.DataFrame,
    model: str,
    scenario: str,
    accelerator_std: str,
    result: float,
    total_accelerators: int,
) -> AnomalyResult:
    normalized = result / total_accelerators

    group = historical_df[
        (historical_df["Model MLC"] == model)
        & (historical_df["Scenario"] == scenario)
        & (historical_df["accelerator_std"] == accelerator_std)
    ]["normalized_result"].dropna().tolist()

    base = AnomalyResult(
        model=model,
        scenario=scenario,
        accelerator=accelerator_std,
        new_value=result,
        total_accelerators=total_accelerators,
        normalized_value=normalized,
        mean=None,
        std=None,
        z_score=None,
        p_value=None,
        is_anomaly=False,
        reason="",
        historical_values=group,
    )

    if len(group) < MIN_SAMPLES:
        base.reason = f"Insufficient historical data ({len(group)} samples, need {MIN_SAMPLES})"
        return base

    mean = np.mean(group)
    std = np.std(group, ddof=1)
    base.mean = float(mean)
    base.std = float(std)

    if std == 0:
        base.reason = "Zero variance in historical data"
        base.is_anomaly = normalized != mean
        return base

    z = (normalized - mean) / std
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    base.z_score = float(z)
    base.p_value = float(p)

    if abs(z) > ANOMALY_Z_THRESHOLD:
        base.is_anomaly = True
        direction = "above" if z > 0 else "below"
        base.reason = f"Result is {abs(z):.2f} std devs {direction} historical mean (p={p:.4f})"
    else:
        base.reason = f"Within normal range (z={z:.2f}, p={p:.4f})"

    return base


def get_group_stats(historical_df: pd.DataFrame) -> pd.DataFrame:
    stats_rows = []
    for (model, scenario, accel), grp in historical_df.groupby(
        ["Model MLC", "Scenario", "accelerator_std"]
    ):
        vals = grp["normalized_result"].dropna()
        stats_rows.append({
            "Model MLC": model,
            "Scenario": scenario,
            "Accelerator": accel,
            "Count": len(vals),
            "Mean": vals.mean() if len(vals) > 0 else None,
            "Std": vals.std(ddof=1) if len(vals) > 1 else None,
            "Min": vals.min() if len(vals) > 0 else None,
            "Max": vals.max() if len(vals) > 0 else None,
        })
    return pd.DataFrame(stats_rows)
