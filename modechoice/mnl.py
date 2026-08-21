"""
Simple MNL: estimation (with availability), predicted shares, scenario
adjustments, and single-attribute sensitivity.

Utility for person i, alternative j:
    V_ij = ASC_j + b_time*time_ij + b_cost*cost_ij + b_park*park_ij
Cost enters every mode except Bike/Walk; parking enters Drive/Carpool/ParkRide.
Estimation is plain maximum likelihood with an analytic gradient (scipy BFGS),
with no dependency on a choice-modelling package, so it stays easy to audit.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import config
from .config import MODES, REF, NONREF


# --------------------------------------------------------------------------- #
# Data plumbing
# --------------------------------------------------------------------------- #
def _matrices(df):
    time = df[[f"time_{m}" for m in MODES]].to_numpy(float)
    cost = df[[f"cost_{m}" for m in MODES]].to_numpy(float)
    park = df[[f"park_{m}" for m in MODES]].to_numpy(float)
    av = df[[f"av_{m}" for m in MODES]].to_numpy(float)
    cost = cost * np.array([config.HAS_COST[m] for m in MODES], float)
    park = park * np.array([config.HAS_PARK[m] for m in MODES], float)
    y = np.array([MODES.index(m) for m in df["chosen_mode"]])
    return time, cost, park, av, y


def _design(time, cost, park):
    """Design tensor X of shape (N, J, K): the covariate multiplying each of the
    K parameters, for every alternative."""
    n, J = time.shape
    layers = [time, cost, park]
    for m in NONREF:
        d = np.zeros((n, J)); d[:, MODES.index(m)] = 1.0
        layers.append(d)
    return np.stack(layers, axis=2)


def _utilities(params, time, cost, park):
    b_time, b_cost, b_park = params[:3]
    asc = dict(zip(NONREF, params[3:]))
    asc_vec = np.array([0.0 if m == REF else asc[m] for m in MODES])
    return asc_vec + b_time * time + b_cost * cost + b_park * park


def _probs(params, time, cost, park, av):
    V = np.where(av == 1, _utilities(params, time, cost, park), -1e9)
    V = V - V.max(axis=1, keepdims=True)
    e = np.exp(V) * av
    return e / e.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Model object
# --------------------------------------------------------------------------- #
PARAM_NAMES = ["b_time", "b_cost", "b_park"] + [f"ASC_{m}" for m in NONREF]


@dataclass
class MNLResult:
    params: np.ndarray
    table: pd.DataFrame      # coef, std_err, t_stat
    loglik: float
    converged: bool

    @property
    def value_of_time(self):
        """$/hr implied by the estimated time and cost coefficients."""
        b_time, b_cost = self.params[0], self.params[1]
        return b_time / b_cost * 60


def estimate(df):
    time, cost, park, av, y = _matrices(df)
    X = _design(time, cost, park)
    Y = np.zeros_like(av); Y[np.arange(len(y)), y] = 1.0

    def negll(p):
        prob = _probs(p, time, cost, park, av)
        return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-300, None)).sum()

    def grad(p):
        prob = _probs(p, time, cost, park, av)
        return -np.einsum("nj,njk->k", (Y - prob), X)

    res = minimize(negll, np.zeros(len(PARAM_NAMES)), jac=grad, method="BFGS")
    se = np.sqrt(np.diag(res.hess_inv))
    table = pd.DataFrame({"coef": res.x, "std_err": se}, index=PARAM_NAMES)
    table["t_stat"] = table["coef"] / table["std_err"]
    return MNLResult(res.x, table, -res.fun, bool(res.success))


# --------------------------------------------------------------------------- #
# Prediction, scenarios, sensitivity
# --------------------------------------------------------------------------- #
def predicted_shares(df, params):
    """Aggregate predicted mode shares (mean choice probability over people)."""
    time, cost, park, av, _ = _matrices(df)
    return pd.Series(_probs(params, time, cost, park, av).mean(0), index=MODES, name="share")


def choice_probabilities(df, params):
    """Full N x J probability matrix (for per-person inspection)."""
    time, cost, park, av, _ = _matrices(df)
    return pd.DataFrame(_probs(params, time, cost, park, av), columns=MODES, index=df.index)


def apply_adjustments(df, adjustments):
    """Return a copy of df with attributes scaled.
    adjustments: {(mode, attribute): pct} where attribute in {time, cost, park}
    and pct is a fraction (e.g. 0.20 = +20%, -0.10 = -10%)."""
    d = df.copy()
    for (mode, attr), pct in adjustments.items():
        if pct:
            d[f"{attr}_{mode}"] = d[f"{attr}_{mode}"] * (1 + pct)
    return d


def sensitivity(df, params, mode, attribute, pct_change):
    """Base vs new aggregate shares when one attribute of one mode is scaled."""
    base = predicted_shares(df, params)
    new = predicted_shares(apply_adjustments(df, {(mode, attribute): pct_change}), params)
    tbl = pd.DataFrame({"base": base, "new": new})
    tbl["abs_change"] = tbl["new"] - tbl["base"]
    tbl["pct_change"] = tbl["abs_change"] / tbl["base"]
    return tbl


# --------------------------------------------------------------------------- #
def _demo():
    import pathlib
    csv = pathlib.Path(__file__).resolve().parents[1] / "data" / "synthetic_mode_choice.csv"
    df = pd.read_csv(csv)
    r = estimate(df)
    print(f"log-likelihood {r.loglik:.1f}  converged={r.converged}  VOT ${r.value_of_time:.1f}/hr\n")
    print(r.table.round(4).to_string())
    print("\nobserved vs predicted:")
    obs = df["chosen_mode"].value_counts(normalize=True).reindex(MODES)
    print(pd.DataFrame({"observed": obs, "predicted": predicted_shares(df, r.params)}).round(3).to_string())


if __name__ == "__main__":
    _demo()
