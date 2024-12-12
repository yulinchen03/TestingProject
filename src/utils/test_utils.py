import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
import onnxruntime as rt


def add_checked(df):
    df['checked'] = df['Ja'].apply(lambda x: 1 if x > 0.7 else 0)
    return df


def test_bias(data_path, model_path, feature, new_val):
    model = rt.InferenceSession(model_path)

    df = add_checked(pd.read_csv(data_path))
    df_change = df.copy()

    for feature, val in zip(feature, new_val):
        df_change[feature] = val  # change value of feature here

    X_1 = df.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_1 = df['checked']
    X_2 = df_change.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_2 = df_change['checked']

    # Let's evaluate the model
    y_1_pred = model.run(None, {'X': X_1.values.astype(np.float32)})[0]
    y_2_pred = model.run(None, {'X': X_2.values.astype(np.float32)})[0]

    y_1_checked = np.sum(y_1_pred)
    y_2_checked = np.sum(y_2_pred)

    acc_1 = accuracy_score(y_1, y_1_pred)
    acc_2 = accuracy_score(y_2, y_2_pred)

    _, p_value = ttest_ind(y_1_pred, y_2_pred)
    return acc_1, acc_2, p_value, df.shape[0], y_1_checked, y_2_checked


def test_bias_with_range(data_path, model_path, feature, new_vals_range):
    model = rt.InferenceSession(model_path)

    df = add_checked(pd.read_csv(data_path))
    df_change = df.copy()
    high = new_vals_range[1] + 1
    df_change[feature] = np.random.randint(new_vals_range[0], high , size=df_change.shape[0])

    X_1 = df.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_1 = df['checked']
    X_2 = df_change.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_2 = df_change['checked']

    y_1_pred = model.run(None, {'X': X_1.values.astype(np.float32)})[0]
    y_2_pred = model.run(None, {'X': X_2.values.astype(np.float32)})[0]

    y_1_checked = np.sum(y_1_pred)
    y_2_checked = np.sum(y_2_pred)

    acc_1 = accuracy_score(y_1, y_1_pred)
    acc_2 = accuracy_score(y_2, y_2_pred)

    _, p_value = ttest_ind(y_1_pred, y_2_pred)
    return acc_1, acc_2, p_value, df.shape[0], y_1_checked, y_2_checked
