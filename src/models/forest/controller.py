import dataclasses


@dataclasses.dataclass(frozen=True)
class ControllerConfig:
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
    'fado_full': _FULL,
    'fado_no_lambda': dataclasses.replace(
        _FULL, react_lambda=False, detect_fairness_drift=False,
        reset_fairness_stats=False),
    'fado_lambda_only': dataclasses.replace(
        _FULL, prewarm=False, react_lr=False, react_temperature=False),

    'fado_no_reset': dataclasses.replace(_FULL, reset_fairness_stats=False),
    'fado_no_lr': dataclasses.replace(_FULL, react_lr=False, prewarm=False),
    'fado_no_temp': dataclasses.replace(_FULL, react_temperature=False),
    'fado_no_prewarm': dataclasses.replace(_FULL, prewarm=False),
    'fado_no_noise_guard': dataclasses.replace(_FULL, label_noise_guard=False),
    'fado_detect_only': dataclasses.replace(
        _FULL, prewarm=False, react_lr=False, react_temperature=False,
        react_lambda=False, reset_fairness_stats=False),
    'fado_monitor_only': dataclasses.replace(
        ControllerConfig.none(), detect_fairness_drift=True),
    'none': ControllerConfig.none(),
}
