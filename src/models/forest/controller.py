"""Reaction-controller configuration for FADO.

Every component of the controller is an independent switch so each one can be
ablated on its own. Previously the whole controller was a single boolean
(`use_drift_controller`), which made "ablate the effect of each component"
impossible to answer.

A run is described by a `ControllerConfig`; `PRESETS` names the configurations
that are registered as pipeline model names in `src/main.py`.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
    """Which parts of the reaction controller are active.

    Attributes:
      detect_accuracy_drift: run the ADWIN detectors on the error stream. With
        this off nothing can fire and the run reduces to plain Aranyani with
        the FADO evaluator's bookkeeping.
      prewarm: react to the warning-stage detector by pre-warming the learning
        rate before confirmation.
      react_lr: spike and then decay the learning rate on confirmed drift.
      react_temperature: collapse and then ramp the soft-routing temperature on
        confirmed drift.
      label_noise_guard: suppress detections triggered by a single
        high-confidence error (probable label noise).
      detect_fairness_drift: run the fairness monitor (a detector from
        ``detectors.py`` on a signal from ``fairness_signal.py``).
      react_lambda: the lambda controller (decision log 2.1). Projected
        stochastic dual ascent on the constraints |D_g| <= epsilon, driven by
        the per-sample Horvitz--Thompson estimate of D_g; see
        ``lambda_controller.py``. It runs at every step and does not need a
        detection.
      reset_fairness_stats: forget the fairness penalty's running statistics
        (``agg_y``, ``gradient_w``, ``gradient_b``, group counts) on every
        confirmed drift, accuracy or fairness. Without it they are means over
        the whole stream, so after a drift lambda would scale a gradient that
        still points at the pre-drift disparity.
    """

    detect_accuracy_drift: bool = True
    prewarm: bool = True
    react_lr: bool = True
    react_temperature: bool = True
    label_noise_guard: bool = True
    detect_fairness_drift: bool = True
    react_lambda: bool = True
    reset_fairness_stats: bool = True

    @property
    def any_reaction(self):
        return self.react_lr or self.react_temperature or self.react_lambda

    @property
    def is_inert(self):
        """True when the controller cannot change anything about the run."""
        if self.react_lambda:
            return False
        detects = self.detect_accuracy_drift or self.detect_fairness_drift
        return not detects or not (self.any_reaction or self.reset_fairness_stats)

    def describe(self):
        on = [f.name for f in dataclasses.fields(self) if getattr(self, f.name)]
        return ','.join(on) if on else 'none'

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def none(cls):
        return cls(**{f.name: False for f in dataclasses.fields(cls)})

    @classmethod
    def from_spec(cls, spec):
        """Build a config from a preset name or a comma-separated field list.

        'fado_full'            -> the named preset
        'react_lr,prewarm'     -> those fields on, everything else off
        'full,-react_temperature' -> full preset minus one component
        """
        if spec is None:
            return cls()
        if isinstance(spec, cls):
            return spec
        if isinstance(spec, dict):
            return cls(**spec)
        spec = str(spec).strip()
        if not spec:
            return cls()
        if spec in PRESETS:
            return PRESETS[spec]

        names = {f.name for f in dataclasses.fields(cls)}
        tokens = [t.strip() for t in spec.split(',') if t.strip()]
        if tokens and tokens[0] in PRESETS:
            base = PRESETS[tokens[0]].as_dict()
            tokens = tokens[1:]
        else:
            base = {n: False for n in names}
        for token in tokens:
            negate = token.startswith('-')
            name = token.lstrip('-')
            if name not in names:
                raise ValueError(
                    f"unknown controller component {name!r}; "
                    f"choose from {sorted(names)} or a preset in {sorted(PRESETS)}"
                )
            base[name] = not negate
        return cls(**base)


_FULL = ControllerConfig()

PRESETS = {
    # Everything on: accuracy-drift reaction (LR, temperature, prewarm, noise
    # guard), the fairness monitor, the lambda controller and the reset of the
    # fairness statistics on drift. This is the `aranyani` / FADO arm.
    'fado_full': _FULL,
    # FADO as it was before the lambda controller (decision log 2.1): the
    # accuracy-drift reaction only, lambda fixed. The key ablation -- the
    # difference between this and fado_full is what 2.1 adds.
    'fado_no_lambda': dataclasses.replace(
        _FULL, react_lambda=False, detect_fairness_drift=False,
        reset_fairness_stats=False),
    # The lambda controller alone: no LR / temperature / prewarm reaction.
    # Both detectors still run, because a confirmed drift triggers the reset.
    'fado_lambda_only': dataclasses.replace(
        _FULL, prewarm=False, react_lr=False, react_temperature=False),
    # Leave-one-out ablations, each relative to fado_full.
    'fado_no_reset': dataclasses.replace(_FULL, reset_fairness_stats=False),
    'fado_no_lr': dataclasses.replace(_FULL, react_lr=False, prewarm=False),
    'fado_no_temp': dataclasses.replace(_FULL, react_temperature=False),
    'fado_no_prewarm': dataclasses.replace(_FULL, prewarm=False),
    'fado_no_noise_guard': dataclasses.replace(_FULL, label_noise_guard=False),
    # Detect but never react: isolates the cost and the false-positive rate of
    # detection from the effect of any reaction.
    'fado_detect_only': dataclasses.replace(
        _FULL, prewarm=False, react_lr=False, react_temperature=False,
        react_lambda=False, reset_fairness_stats=False),
    # The fairness monitor alone, observing: used by the monitor ablation to
    # compare detector backends and signal designs on detection quality.
    'fado_monitor_only': dataclasses.replace(
        ControllerConfig.none(), detect_fairness_drift=True),
    # No controller at all; the pure-Aranyani arm routes to the baseline
    # evaluator rather than to this config.
    'none': ControllerConfig.none(),
}
