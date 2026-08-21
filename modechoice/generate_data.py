"""
Synthetic mode-choice dataset generator.

The data come from a KNOWN discrete-choice DGP: we specify true utility
coefficients (config.TRUE_PARAMS), build deterministic utilities for every
available alternative, add iid Gumbel error, and take the argmax as the observed
choice. A correctly specified MNL therefore recovers the true parameters up to
sampling noise, which makes this a valid test bed for the estimator and the tool.

Rules encoded here:
  - Parking (a monthly cost) applies only to Drive, Carpool, ParkRide.
  - Bike and Walk have no cost and no parking; only distance (via travel time)
    drives their utility.
  - Walk is available only when distance < WALK_MAX_MI (3 mi).
  - Bike is available only when distance < BIKE_MAX_MI (6 mi).
"""

import numpy as np
import pandas as pd

from . import config
from .config import MODES, TRUE_PARAMS, WALK_MAX_MI, BIKE_MAX_MI


def _noise(rng, size, sigma=0.10):
    """Multiplicative lognormal noise; keeps attributes from being a perfectly
    deterministic function of distance, which the model needs for identification."""
    return np.exp(rng.normal(0.0, sigma, size))


def generate(n=1000, seed=42):
    """Return a DataFrame: one row per person, wide format, with an observed mode."""
    rng = np.random.default_rng(seed)

    # One trip distance per person (miles), tuned so a meaningful share fall under
    # the walk (3 mi) and bike (6 mi) thresholds.
    dist = np.clip(rng.lognormal(np.log(5.0), 0.60, n), 0.3, 30.0)
    df = pd.DataFrame({"person_id": np.arange(1, n + 1), "distance_mi": dist})

    # ---- Travel time (minutes), door to door ----
    df["time_Drive"]         = (dist / 28 * 60 + 3)  * _noise(rng, n)
    df["time_Carpool"]       = (dist / 26 * 60 + 6)  * _noise(rng, n)
    df["time_TNC"]           = (dist / 28 * 60 + 7)  * _noise(rng, n)
    df["time_PublicTransit"] = (dist / 15 * 60 + 12) * _noise(rng, n)
    df["time_Bike"]          = (dist / 10 * 60 + 2)  * _noise(rng, n)
    df["time_Walk"]          = (dist / 3  * 60)      * _noise(rng, n)
    df["time_ParkRide"]      = (dist / 20 * 60 + 15) * _noise(rng, n)

    # ---- Travel cost ($ per trip; excludes monthly parking) ----
    df["cost_Drive"]         = 0.30 * dist * _noise(rng, n)
    df["cost_Carpool"]       = 0.30 * dist / 2.5 * _noise(rng, n)
    df["cost_TNC"]           = (2.5 + 1.20 * dist + 0.30 * (dist / 28 * 60)) * _noise(rng, n)
    df["cost_PublicTransit"] = (2.5 + 0.05 * dist) * _noise(rng, n)
    df["cost_Bike"]          = 0.0
    df["cost_Walk"]          = 0.0
    df["cost_ParkRide"]      = (2.5 + 0.30 * (0.4 * dist)) * _noise(rng, n)

    # ---- Monthly parking cost ($/month; only Drive, Carpool, ParkRide) ----
    park_base = rng.uniform(40, 220, n)              # destination parking varies by zone
    df["park_Drive"]    = park_base
    df["park_Carpool"]  = park_base * 0.5            # split / subsidy
    df["park_ParkRide"] = rng.uniform(0, 70, n)      # cheaper station parking
    for m in ("TNC", "PublicTransit", "Bike", "Walk"):
        df[f"park_{m}"] = 0.0

    # ---- Availability ----
    for m in ("Drive", "Carpool", "TNC", "PublicTransit", "ParkRide"):
        df[f"av_{m}"] = 1
    df["av_Bike"] = (dist < BIKE_MAX_MI).astype(int)
    df["av_Walk"] = (dist < WALK_MAX_MI).astype(int)

    # ---- Utilities + Gumbel draw -> observed choice ----
    def util(mode):
        v = TRUE_PARAMS["asc"][mode] + TRUE_PARAMS["b_time"] * df[f"time_{mode}"]
        if config.HAS_COST[mode]:
            v = v + TRUE_PARAMS["b_cost"] * df[f"cost_{mode}"]
        if config.HAS_PARK[mode]:
            v = v + TRUE_PARAMS["b_park"] * df[f"park_{mode}"]
        return v

    V = np.column_stack([util(m) for m in MODES])
    av = df[[f"av_{m}" for m in MODES]].to_numpy()
    gumbel = -np.log(-np.log(rng.uniform(size=V.shape)))
    U = np.where(av == 1, V + gumbel, -np.inf)
    df["chosen_mode"] = [MODES[i] for i in U.argmax(axis=1)]

    ordered = (["person_id", "distance_mi"]
               + [f"{p}_{m}" for m in MODES for p in ("time", "cost", "park", "av")]
               + ["chosen_mode"])
    return df[ordered]


def main():
    import pathlib
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_mode_choice.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate(seed=274)   # draw calibrated to ~47.7% Drive / 5.1% Bike base case
    df.to_csv(out, index=False)
    shares = df["chosen_mode"].value_counts(normalize=True).reindex(MODES).round(3)
    print(f"Wrote {len(df)} rows to {out}")
    print("Observed shares:\n" + shares.to_string())


if __name__ == "__main__":
    main()
