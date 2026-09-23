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
      detect_fairness_drift: monitor the fairness signal for drift. Reserved
        for the lambda controller; not implemented yet.
      react_lambda: modulate the fairness penalty weight in response to
        fairness drift. Reserved for the lambda controller; not implemented yet.
    """

    detect_accuracy_drift: bool = True
    prewarm: bool = True
    react_lr: bool = True
    react_temperature: bool = True
    label_noise_guard: bool = True
    detect_fairness_drift: bool = False
    react_lambda: bool = False

    @property
    def any_reaction(self):
        return self.react_lr or self.react_temperature or self.react_lambda

    @property
    def is_inert(self):
        """True when the controller cannot change anything about the run."""
        return not (self.detect_accuracy_drift or self.detect_fairness_drift) \
            or not self.any_reaction

    def describe(self):
        on = [f.name for f in dataclasses.fields(self) if getattr(self, f.name)]
        return ','.join(on) if on else 'none'

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def none(cls):
        return cls(detect_accuracy_drift=False, prewarm=False, react_lr=False,
                   react_temperature=False, label_noise_guard=False)

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


PRESETS = {
    # Everything on. Equivalent to the historical `aranyani` model.
    'fado_full': ControllerConfig(),
    # Detect but never react: isolates the cost and the false-positive rate of
    # detection from the effect of the reaction.
    'fado_detect_only': ControllerConfig(
        prewarm=False, react_lr=False, react_temperature=False),
    # Single-component ablations.
    'fado_lr_only': ControllerConfig(react_temperature=False),
    'fado_temp_only': ControllerConfig(prewarm=False, react_lr=False),
    'fado_no_prewarm': ControllerConfig(prewarm=False),
    'fado_no_noise_guard': ControllerConfig(label_noise_guard=False),
    # Monitor arms: the fairness detector runs and records what it *would*
    # have signalled, without reacting. Used by the monitor ablation to compare
    # detector backends and signal designs on detection quality alone.
    'fado_monitor': ControllerConfig(detect_fairness_drift=True),
    'fado_monitor_only': ControllerConfig(
        detect_accuracy_drift=False, prewarm=False, react_lr=False,
        react_temperature=False, detect_fairness_drift=True),
    # No controller at all; the pure-Aranyani arm routes to the baseline
    # evaluator rather than to this config.
    'none': ControllerConfig.none(),
}
