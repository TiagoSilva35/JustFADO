import pandas as pd
from src.drift.pandas_adapter import PandasCapyMOAAdapter

DATASET_PATH = "data/compas/compas-scores-two-years.csv"

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    # Create an instance of the PandasCapyMOAAdapter
    adapter = PandasCapyMOAAdapter(data=df, target_column="two_year_recid")

    # Get the CapyMOA schema
    schema = adapter.get_schema()
    print("Schema:", schema)

    # Iterate through the instances in the stream
    while adapter.has_more_instances():
        instance = adapter.next_instance()
        print("Instance:", instance)