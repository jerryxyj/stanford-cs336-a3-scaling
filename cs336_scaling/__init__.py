# `schemas` <-> `training.training_config` <-> `training.model.config` form an import cycle
# that only resolves when `schemas` is imported first. Pin that order here so any submodule
# can be imported standalone (e.g. `import cs336_scaling.training.model.config`).
import cs336_scaling.schemas  # noqa: F401
