import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
import onnxruntime as rt


def add_checked(df):
    df['checked'] = df['Ja'].apply(lambda x: 1 if x > 0.7 else 0)
    return df


def show_stats(dataset_size, acc_original, f_name_original, acc_changed, f_name_changed, original_checked_cnt, changed_checked_cnt, p_val):
    print(f'Accuracy for sample of {dataset_size} {f_name_original}: {acc_original * 100:.1f}%')
    print(f'Accuracy for sample of {dataset_size} {f_name_changed}: {acc_changed * 100:.1f}%')
    print(f'Percentage checked amongst {f_name_original}: {original_checked_cnt * 100 / dataset_size:.1f}%')
    print(f'Percentage checked changed to {f_name_changed}: {changed_checked_cnt * 100 / dataset_size:.1f}%')
    print(f'P value: {p_val}')


def run_bias_test(data_path, model_path, modified_model_path, feature, new_val, desc_original, desc_changed):
    # Run bias test on the original model
    acc_original, acc_changed, p_value_1, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
        data_path, model_path, feature, new_val)

    # Run bias test on the modified model
    acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
        data_path, modified_model_path, feature, new_val)

    print("=== Model 1 Results ===")
    show_stats(dataset_size, acc_original, desc_original, acc_changed, desc_changed, original_checked_cnt, changed_checked_cnt, p_value_1)
    print("\n=== Model 2 Results ===")
    show_stats(dataset_size_2, acc_original_2, desc_original, acc_changed_2, desc_changed, original_checked_cnt_2, changed_checked_cnt_2, p_value_2)
    print("\n")

    return p_value_2


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
