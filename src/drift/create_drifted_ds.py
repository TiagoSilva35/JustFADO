import pandas as pd
import random

seed = 42
def generate_drifted_dataset(X):
    """
    Drifts the dataset X according to the specified split percentages. 
    
    Current implementation (5 splits):
    -> 20% warmup (original distribution)
    -> 15% abrupt drift (drifted distribution)
    -> 25% recovery (original distribution)
    -> 15% slow drift (drifted distribution)
    -> 25% recovery (original distribution)
    
    Args:
        X (pandas.DataFrame): Dataframe to be drifted.
        split (list): A list of percentages indicating how much of the dataset should be drifted. 
    """
    random.seed(seed)
    split = [0.2, 0.35, 0.6, 0.75, 1]
    
    X = X.values.tolist()
    
    n_samples = len(X)
    warmup = X[0:int(split[0] * n_samples)]
    abrupt_drift = X[int(split[0] * n_samples):int((split[1]) * n_samples)]
    recovery_1 = X[int((split[1]) * n_samples):int((split[2]) * n_samples)]
    slow_drift = X[int((split[2]) * n_samples):int((split[3]) * n_samples)]
    recovery_2 = X[int((split[3]) * n_samples):]
    
    count1, count2 = 0, 0
    for sample in abrupt_drift:
        if str(sample[-1]).strip() == "<=50K" and str(sample[9]).strip() == 'Female':
            sample[9] = "Male"
    
    temperature = 0.999
    
    for sample in slow_drift:
        if str(sample[-1]).strip() == "<=50K" and str(sample[9]).strip() == 'Female':
            r = random.random()
            if r < temperature:
                sample[9] = "Male"
                count1 += 1
            temperature *= 0.999
            count2 += 1
            
    return warmup + abrupt_drift + recovery_1 + slow_drift + recovery_2
    
        

    
    
    
    