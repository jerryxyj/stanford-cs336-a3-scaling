# CS336 Spring 2026 Assignment 3: Scaling

For a full description of the assignment, see the assignment handout at
[cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf).

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## For students

Install uv

```sh
uv sync
```

Set `A3_API_KEY` to your 8-digit student ID:

```sh
export A3_API_KEY=06123456
```

The hosted training API is available at:

```text
http://hyperturing.stanford.edu:8000
```

Click here for the [docs](http://hyperturing.stanford.edu:8000/docs) and [dashboard](http://hyperturing.stanford.edu:8000/dashboard).

See [`./examples/client_example.ipynb`](./examples/client_example.ipynb) for an
example of submitting and inspecting training runs.

## For non-students

Install dependencies:

```sh
uv sync --extra server
```

To download tokenized data:

```sh
uv run modal run scripts/1_download_tokenized_data.py
```

To run training directly:

```sh
uv run cs336_scaling/training/run.py
```

To run the API and dispatcher, set:

```sh
DATABASE_URL_PROD="postgresql://..."
DATABASE_URL_DEV="postgresql://..."
INTERNAL_API_KEY="SOMEKEY"
```

Then run:

```sh
DB_ENV=prod uv run fastapi run &
DB_ENV=prod uv run dispatcher &
```

## Scaling-law solution (`cs336_scaling/scaling_laws`)

The assignment's two problems are implemented in the `cs336_scaling.scaling_laws` package;
the write-up lives in [`writeup/writeup.md`](./writeup/writeup.md).

> **Note on dependencies.** `pyproject.toml` pins PyPI as the default package index (so
> `uv sync` works on machines whose global `uv` config points at a mirror that lacks some
> packages) and adds `matplotlib`/`scipy` to the `dev` group, which `uv sync` installs by
> default. JAX runs on CPU on macOS, which is enough for the tests and the simulator.

### Problem `chinchilla_isoflops`

```sh
uv run scripts/chinchilla_isoflops.py            # or: uv run python -m cs336_scaling.scaling_laws isoflops
```

Fits `N_opt(C)` and `D_opt(C)` power laws to the per-budget minima of
`data/isoflops_curves.json` and writes plots + `summary.json` to `results/isoflops/`.

### Problem `scaling_laws`

The study is a staged pipeline with a persistent state file
(`results/scaling_laws/<backend>/state.json`):

| command | what it does |
|---|---|
| `plan --stage probe` | short run per model shape to measure tokens/sec |
| `plan --stage lr_sweep` | peak-LR sweeps at two small scales -> `lr_opt(N)` rule |
| `plan --stage iso_time` | next iso-time tier: several shapes trained for the same wall-clock budget |
| `plan --stage bracket` | extend tiers whose minimum sits on the edge of the sampled shapes |
| `run` | submit planned runs, poll until finished, persist results (resumable) |
| `fit` | throughput model, LR rule, iso-time laws, parametric `L(N, D)`, hold-out check, bootstrap; writes `fit/report.md`, plots and `fit/final_submission.json` |
| `submit-final` | POST `final_submission.json` to the API |
| `all` | the whole study: probe -> lr_sweep -> iso-time tiers with bracketing -> fit |
| `evaluate` | (simulator only) score the final submission against the hidden truth |

Against the hosted API (students; needs `A3_API_KEY` and the Stanford network):

```sh
uv run python -m cs336_scaling.scaling_laws all --backend api          # or stage by stage with plan/run
uv run python -m cs336_scaling.scaling_laws fit --backend api
uv run python -m cs336_scaling.scaling_laws submit-final --backend api
```

Offline, against the built-in simulator (no GPUs; validates the whole pipeline end to end):

```sh
uv run python -m cs336_scaling.scaling_laws all --backend simulated --fast
```

Tests for the solution do not need Postgres:

```sh
uv run pytest tests/test_isoflops.py tests/test_scaling_laws.py
```
