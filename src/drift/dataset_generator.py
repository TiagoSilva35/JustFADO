import pandas as pd
import numpy as np
from src.drift.inject_drift import DriftSimulator, DRIFT_CONFIGS
from capymoa.stream import NumpyStream

config = DRIFT_CONFIGS["x_permutations"]

class Generator:
    def __init__(self, x: np.ndarray, y: np.ndarray, stream: NumpyStream):
        self.x = x
        self.y = y
        self.stream = stream
        self.drift_simulator = DriftSimulator(
            schema=stream.get_schema(),
            **config,
            width=0,
            drift_region=(0.5, 0.7)
        )
        self.stream_size = x.size

    def generate_drifted_dataset(self) -> NumpyStream:
        self.drift_simulator.fit(self.x.size)
        X, Y = np.array(self.x), np.array(self.y)
        for idx in range(self.stream_size):
            instance = self.stream.next_instance()
            if self.drift_simulator.apply_drift(idx):
                X[idx] = self.drift_simulator.transform(instance.X)
                Y[idx] = self.drift_simulator.transform(instance.y)
        drifted_stream = NumpyStream(X, Y, dataset_name=self.stream.get_schema().dataset_name, feature_names=[
            self.stream.get_schema()._moa_header.attribute(i) for i in range(self.stream.get_schema().get_num_attributes())
        ])
        return drifted_stream


        
