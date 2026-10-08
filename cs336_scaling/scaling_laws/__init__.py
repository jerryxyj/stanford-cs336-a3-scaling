"""Scaling-law tooling for CS336 Assignment 3.

Two problems are covered:

* ``chinchilla_isoflops`` -- reproduce the IsoFLOPs method on the synthetic data in
  ``data/isoflops_curves.json`` (see :mod:`cs336_scaling.scaling_laws.isoflops`).
* ``scaling_laws`` -- plan, run, and fit scaling laws against the training API in order to
  pick the compute-optimal configuration for a 48 B200-hour run (see
  :mod:`cs336_scaling.scaling_laws.cli` for the pipeline entry point).
"""
