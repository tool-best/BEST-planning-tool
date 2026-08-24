# Mode Share Prediction Tool

An open, transparent toolkit for estimating mode choice model and exploring how changes in various 
attributes shift the aggregate mode share.

Seven modes are considered: Drive, Carpool, TNC (Uber/Lyft), Public Transit, Bike,
Walk, and Park & Ride.

## What it does

- Generates a synthetic choice dataset from a documented data generating process.
- Estimates a simple MNL by maximum likelihood, respecting alternative availability.
- Reports predicted mode shares.
- Lets you scale travel time, per-trip cost, and monthly parking cost for any mode
  by a percentage and see how overall shares respond.

## Modeling approach
There are some considerations:

- Cost enters every mode except Bike and Walk, which are free.
- Monthly parking cost enters only Drive, Carpool, and Park & Ride, each with its own
  coefficient so it can be varied directly in the tool.
- Bike and Walk utility is driven by distance; they carry no cost
  or parking term.
- Walk is available only when trip distance is under 3 miles, Bike only under 6 miles.
  Availability is enforced in both estimation and prediction.

The synthetic data are drawn from known coefficients (see `modechoice/config.py`),
so estimation can be checked against ground truth. 

## Install

```bash
git clone <your-repo-url>
cd mode-choice-tool
pip install -e .
```

Requires Python 3.9+. Dependencies: numpy, pandas, scipy, streamlit, altair.

## Usage

Regenerate the dataset:

```bash
python -m modechoice.generate_data
```

Estimate the model and print a fit summary:

```bash
python -m modechoice.mnl
```

### Static web tool (HTML)

A self-contained browser version lives in `docs/`. It shows the attribute controls,
the predicted mode-share table, and the base-vs-scenario chart, with no server to run.
Rebuild its data whenever the model changes:

```bash
python -m modechoice.export_web      # writes docs/model_data.js from the fitted model
```

Then open `docs/index.html` in a browser, or host it for free on GitHub Pages:
in the repo settings under Pages, set the source to the `main` branch and the `/docs`
folder. The estimation runs in Python; the page applies the fitted coefficients and
recomputes shares client-side, so it stays exact and fast.

Use it as a library:

```python
import pandas as pd
from modechoice import estimate, predicted_shares, sensitivity

df = pd.read_csv("data/synthetic_mode_choice.csv")
result = estimate(df)
print(predicted_shares(df, result.params))

# Effect of a 25% increase in Drive parking on aggregate shares
print(sensitivity(df, result.params, "Drive", "park", 0.25))
```

## Repository layout

```
modechoice/            core package
  config.py            modes, availability rules, true coefficients
  generate_data.py     synthetic data generator
  mnl.py               estimation, prediction, sensitivity
  export_web.py        exports the fitted model to docs/model_data.js
docs/
  index.html           static web tool (HTML/CSS/JS)
  model_data.js        generated: fitted coefficients + attribute matrices
tests/
  test_modechoice.py   generation, estimation, and sensitivity tests
data/
  mode_choice_data.csv
```

## Data dictionary

One row per traveler. For each mode `M` the columns are `time_M` (minutes),
`cost_M` (dollars per trip), `park_M` (dollars per month), and `av_M` (1 if the mode
is available to that person). Plus `person_id`, `distance_mi`, and `chosen_mode`.

## Tests

```bash
pytest
```

## License

See `LICENSE`.
