"""Uniform drift-detector interface over river and CapyMOA.

The controller used to hard-code a river ADWIN. Comparing detectors is part of
the evaluation now, so detectors are addressed by a ``backend:name`` spec and
wrapped behind one ``update(value) -> bool`` call. This also absorbs river's
``change_detected`` -> ``drift_detected`` rename, so the code works on either
river version.

CapyMOA detectors are JVM-backed (JDK 17+ required); the import is lazy so the
repo still runs without a JVM as long as no ``capymoa:`` spec is requested.

    det = make_detector('capymoa:seed', delta=0.05)
    if det.update(error):
        ...
"""

import importlib

RIVER_DETECTORS = {
    'adwin': ('ADWIN', {'delta'}),
    'ddm': ('DDM', set()),
    'eddm': ('EDDM', set()),
    'hddm_a': ('HDDM_A', set()),
    'hddm_w': ('HDDM_W', set()),
    'kswin': ('KSWIN', {'alpha', 'window_size', 'stat_size'}),
    'page_hinkley': ('PageHinkley', {'min_instances', 'delta', 'threshold', 'alpha'}),
}

# All CapyMOA detectors share the MOADriftDetector API. The second element is
# the set of constructor keywords we forward; anything else is dropped with a
# warning so a sweep can pass one parameter dict to every arm.
CAPYMOA_DETECTORS = {
    'adwin': ('ADWIN', {'delta'}),
    'cusum': ('CUSUM', {'min_n_instances', 'delta', 'lambda_'}),
    'ddm': ('DDM', {'min_n_instances', 'warning_level', 'out_control_level'}),
    'ewma': ('EWMAChart', {'min_n_instances', 'lambda_'}),
    'gma': ('GeometricMovingAverage', {'min_n_instances', 'lambda_', 'alpha'}),
    'hddm_a': ('HDDMAverage', {'drift_confidence', 'warning_confidence', 'test_type'}),
    'hddm_w': ('HDDMWeighted', {'drift_confidence', 'warning_confidence', 'lambda_', 'test_type'}),
    'optwin': ('OPTWIN', {'rigor', 'drift_confidence', 'warning_confidence'}),
    'page_hinkley': ('PageHinkley', {'min_n_instances', 'delta', 'lambda_', 'alpha'}),
    'rddm': ('RDDM', {'min_n_instances', 'warning_level', 'drift_level'}),
    'seed': ('SEED', {'delta', 'block_size', 'epsilon_prime', 'alpha', 'compress_term'}),
    'stepd': ('STEPD', {'window_size', 'alpha_drift', 'alpha_warning'}),
}

DEFAULT_SPEC = 'river:adwin'


def available_detectors():
    """Every spec this module can build, whether or not the backend is installed."""
    return (sorted(f'river:{name}' for name in RIVER_DETECTORS)
            + sorted(f'capymoa:{name}' for name in CAPYMOA_DETECTORS))


def parse_spec(spec):
    """'capymoa:seed' -> ('capymoa', 'seed'). A bare name means river."""
    spec = str(spec or DEFAULT_SPEC).strip().lower()
    backend, _, name = spec.partition(':')
    if not name:
        backend, name = 'river', backend
    if backend not in ('river', 'capymoa'):
        raise ValueError(f"unknown detector backend {backend!r} in {spec!r}")
    table = RIVER_DETECTORS if backend == 'river' else CAPYMOA_DETECTORS
    if name not in table:
        raise ValueError(
            f"unknown {backend} detector {name!r}; available: {sorted(table)}")
    return backend, name


class DriftDetector:
    """One detector, one scalar stream, one boolean answer per update."""

    def __init__(self, spec=DEFAULT_SPEC, **params):
        self.spec = str(spec or DEFAULT_SPEC).strip().lower()
        self.backend, self.name = parse_spec(self.spec)
        self._params = dict(params)
        self._dropped = []
        self._detector = self._build()
        self.n_updates = 0
        self.n_drifts = 0

    # -- construction ------------------------------------------------------
    def _accepted_params(self):
        table = RIVER_DETECTORS if self.backend == 'river' else CAPYMOA_DETECTORS
        _, accepted = table[self.name]
        kept, dropped = {}, []
        for key, value in self._params.items():
            if key in accepted:
                kept[key] = value
            else:
                dropped.append(key)
        self._dropped = dropped
        return kept

    def _build(self):
        kwargs = self._accepted_params()
        if self.backend == 'river':
            module = importlib.import_module('river.drift')
            cls = getattr(module, RIVER_DETECTORS[self.name][0], None)
            if cls is None:  # river moved some detectors under drift.binary
                module = importlib.import_module('river.drift.binary')
                cls = getattr(module, RIVER_DETECTORS[self.name][0])
        else:
            try:
                module = importlib.import_module('capymoa.drift.detectors')
            except ImportError as exc:
                raise ImportError(
                    f"detector spec {self.spec!r} needs CapyMOA. Install it with "
                    f"`pip install capymoa` and make sure a JDK 17+ is on PATH."
                ) from exc
            cls = getattr(module, CAPYMOA_DETECTORS[self.name][0])
        return cls(**kwargs)

    # -- use ---------------------------------------------------------------
    def update(self, value):
        """Feed one observation. Returns True when drift is signalled."""
        self.n_updates += 1
        value = float(value)
        if self.backend == 'river':
            self._detector.update(value)
            detected = bool(getattr(self._detector, 'drift_detected',
                                    getattr(self._detector, 'change_detected', False)))
        else:
            self._detector.add_element(value)
            detected = bool(self._detector.detected_change())
        self.n_drifts += int(detected)
        return detected

    @property
    def warning_detected(self):
        if self.backend == 'river':
            return bool(getattr(self._detector, 'warning_detected', False))
        return bool(self._detector.detected_warning())

    def reset(self):
        """Rebuild the detector, discarding its accumulated state."""
        self._detector = self._build()
        return self

    def describe(self):
        return {
            'spec': self.spec,
            'backend': self.backend,
            'name': self.name,
            'params': {k: v for k, v in self._params.items() if k not in self._dropped},
            'ignored_params': list(self._dropped),
            'n_updates': int(self.n_updates),
            'n_drifts': int(self.n_drifts),
        }


def make_detector(spec=DEFAULT_SPEC, **params):
    return DriftDetector(spec, **params)
