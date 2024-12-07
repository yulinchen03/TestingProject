import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


def evaluate_importance(X, y, 
                        model=None, 
                        n_repeats=5, 
                        n_jobs=-1, 
                        random_state=None, 
                        cache=True):

    if model is None:
        model = RandomForestClassifier(random_state=random_state)
    
    model.fit(X, y)
    
    feature_importance = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
    )

    if cache:
        with open("feature_importance.pkl", "wb") as f:
            pickle.dump(feature_importance, f)
    
    return feature_importance


def normalize_importance(importance):
    return importance / importance.sum()
