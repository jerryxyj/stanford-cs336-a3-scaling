"""Problem ``chinchilla_isoflops``: fit IsoFLOPs scaling laws to ``data/isoflops_curves.json``.

Usage::

    uv run scripts/chinchilla_isoflops.py [--data PATH] [--out DIR]

Writes ``model_size_scaling_law.png``, ``dataset_size_scaling_law.png``,
``isoflops_profiles.png`` and ``summary.json`` to ``--out`` (default ``results/isoflops``)
and prints the predicted compute-optimal model and dataset sizes for 1e23 and 1e24 FLOPs.
Equivalent to ``python -m cs336_scaling.scaling_laws isoflops``.
"""

from cs336_scaling.scaling_laws.cli import main

if __name__ == "__main__":
    import sys

    main(["isoflops", *sys.argv[1:]])
