import pandas as pd
from src.drift.pandas_adapter import PandasCapyMOAAdapter
from src.helpers.data import read_compas
DATASET_PATH = "data/compas/compas-scores-two-years.csv"

if __name__ == "__main__":
    # Load the dataset
    encoded_features, y, _ = read_compas(
        DATASET_PATH, return_dataframe=True
    )
    df = pd.DataFrame(encoded_features).copy()
    df["two_year_recid"] = y
    # Create an instance of the PandasCapyMOAAdapter
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    adapter = PandasCapyMOAAdapter(data=df, target_column="two_year_recid")

    # # Get the CapyMOA schema
    # schema = adapter.get_schema()
    # print("Schema:", schema)

    # # Iterate through the instances in the stream
    # while adapter.has_more_instances():
    #     instance = adapter.next_instance()
    #     print("Instance:", instance)
    