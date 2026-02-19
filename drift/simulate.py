import pandas as pd
import os

columns = [
    "age", 
    "workclass", 
    "fnlwgt", 
    "education", 
    "education-num", 
    "marital-status", 
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

def open_file(path, mode='r'):
    with open(os.path(path), mode) as f:
        df = pd.read_csv(f, names=columns)
    return df

def print_top_10(path="../data/adult"):
    positive = []
    related_features = {}
    with open(os.path.join(path, "adult.data"), 'r') as f:
        df = pd.read_csv(f, names=columns)
        for row in df.iterrows():
            if row[1]["income"] == " >50K":
                for col in columns[:-1]: 
                    if row[1][col] not in related_features:
                        related_features[row[1][col]] = 0
                    related_features[row[1][col]] += 1
    print(list(top10 for top10 in sorted(related_features.items(), key=lambda item: item[1], reverse=True)[:10]))
    
    
    

print_top_10()
  

