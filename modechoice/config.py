"""Structural constants for the mode-choice model and the data generating process.

Everything that both the generator and the estimator need to agree on lives here,
so the two never drift apart.
"""

# Alternatives, in a fixed canonical order used everywhere (columns, arrays, plots).
MODES = ["Drive", "Carpool", "TNC", "PublicTransit", "Bike", "Walk", "ParkRide"]

# Reference alternative: its ASC is fixed to 0 for identification.
REF = "Drive"

# Which alternatives carry which attributes.
HAS_COST = {m: m not in ("Bike", "Walk") for m in MODES}          # bike/walk are free
HAS_PARK = {m: m in ("Drive", "Carpool", "ParkRide") for m in MODES}

# Availability rules for the distance-gated modes (miles).
WALK_MAX_MI = 3.0
BIKE_MAX_MI = 6.0

# ---- "True" parameters used only to SIMULATE the data. ----
# Estimation should recover these (up to sampling noise); keeping them here lets
# the tests and the README report recovery against a documented ground truth.
TRUE_PARAMS = {
    "b_time": -0.045,   # utils per minute
    "b_cost": -0.120,   # utils per $ (per trip)
    "b_park": -0.008,   # utils per $ (per month)
    "asc": {
        "Drive": 0.0,           # reference
        "Carpool": -1.637,
        "TNC": -1.699,
        "PublicTransit": -1.038,
        "Bike": -2.263,
        "Walk": -0.714,
        "ParkRide": -1.871,
    },
}
# Implied value of time = b_time / b_cost * 60 = 22.5 $/hr.

NONREF = [m for m in MODES if m != REF]
