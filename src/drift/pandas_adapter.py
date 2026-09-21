import pandas as pd
from capymoa.stream import NumpyStream


class PandasCapyMOAAdapter:
    """Adapt a labeled pandas DataFrame to a CapyMOA stream."""

    def __init__(
        self, 
        data: pd.DataFrame, 
        target_column: str = "target", 
        dataset_name: str = "pandas_dataset",
    ):
        self.data = data.copy()
        self.target_column = target_column
        self.dataset_name = dataset_name
        if target_column not in self.data.columns:
            raise ValueError(f"Missing target column: {target_column}")

        self.feature_columns = [
            column for column in self.data.columns if column != target_column
        ]
        if not self.feature_columns:
            raise ValueError("data must contain at least one feature column")

        if not all(
            pd.api.types.is_numeric_dtype(self.data[column])
            for column in self.feature_columns
        ):
            raise TypeError("All feature columns must be numeric")

        self._stream = self._create_stream()

    def _create_stream(self) -> NumpyStream:
        return NumpyStream(
            X=self.data[self.feature_columns].to_numpy(),
            y=self.data[self.target_column].to_numpy(),
            dataset_name=self.dataset_name,
            feature_names=self.feature_columns,
        )

    def get_schema(self):
        """Return the CapyMOA schema generated from the DataFrame."""
        return self._stream.get_schema()

    def next_instance(self):
        """Return the next CapyMOA instance."""
        return self._stream.next_instance()

    def has_more_instances(self) -> bool:
        """Return whether another instance is available."""
        return self._stream.has_more_instances()

    def reset(self) -> None:
        """Restart iteration from the first DataFrame row."""
        self._stream = self._create_stream()
