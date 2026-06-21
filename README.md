# Battery Formation Recommendation

Configuration is stored under `inputs/`. Runtime code is separated into:

- `simulations/`: PyBaMM parameters, experiments, model execution, analysis, plots.
- `surrogate/`: trajectory features, dataset creation, training, prediction.
- `mco/`: formation candidate generation, evaluation, scoring, ranking.

Create the surrogate dataset once:

```bash
.venv/bin/python -m surrogate.create_dataset
```

The command skips generation when `surrogate/dataset.csv` exists. Regenerate with:

```bash
.venv/bin/python -m surrogate.create_dataset --force
```

Train the surrogate models from the existing dataset:

```bash
.venv/bin/python -m surrogate.train_models
```

Run one formation and ageing simulation using the defaults in
`inputs/simulation.toml`:

```bash
.venv/bin/python run_simulation.py
```

Run formation recommendations:

```bash
.venv/bin/python run_recommendation.py
```

`inputs/mco.toml` controls whether recommendation uses `auto`, `surrogate`, or
`pybamm` mode. In `auto` mode, trained surrogate models are preferred and PyBaMM
ageing is used when models are unavailable.
