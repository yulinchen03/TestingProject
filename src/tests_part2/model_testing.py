import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score
import onnxruntime as rt
import matplotlib.pyplot as plt
import unittest

def add_checked(df):
    """
    Create a binary 'checked' column from the continuous predictions in 'Ja'.
    """
    df['checked'] = df['Ja'].apply(lambda x: 1 if x > 0.7 else 0)
    return df


def show_stats(dataset_size, acc_original, f_name_original, 
              acc_changed, f_name_changed, 
              original_checked_cnt, changed_checked_cnt, p_val):
    """
    Print numeric details of the model's performance.
    """
    print(f'Accuracy for sample of {dataset_size} {f_name_original}: {acc_original * 100:.1f}%')
    print(f'Accuracy for sample of {dataset_size} {f_name_changed}: {acc_changed * 100:.1f}%')
    print(f'Percentage checked amongst {f_name_original}: {(original_checked_cnt * 100 / dataset_size):.1f}%')
    print(f'Percentage checked changed to {f_name_changed}: {(changed_checked_cnt * 100 / dataset_size):.1f}%')
    print(f'P value: {p_val}')


def plot_model_comparison_bar_chart(
    original_pct_model1, changed_pct_model1,
    original_pct_model2, changed_pct_model2,
    desc_original, desc_changed, title=None
):
    """
    Plots a grouped bar chart comparing how often each model returns 'checked'
    for the original vs. changed scenario.
    """
    labels = [desc_original, desc_changed]
    model1_vals = [original_pct_model1, changed_pct_model1]
    model2_vals = [original_pct_model2, changed_pct_model2]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(8, 6))
    bar1 = ax.bar(x - width/2, model1_vals, width, label='Model 1', color='#1f77b4')
    bar2 = ax.bar(x + width/2, model2_vals, width, label='Model 2', color='#ff7f0e')

    # Add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage Checked (%)')
    ax.set_title(title if title else 'Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Annotate bars with actual values
    for rect in bar1 + bar2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def run_bias_test(data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed):
    """
    Run the bias test for both models and show comparison stats.
    Also returns relevant stats so we can produce bar charts.
    """
    # 1. Evaluate for Model 1
    (acc_original_1, acc_changed_1, p_value_1,
     dataset_size_1, original_checked_cnt_1, changed_checked_cnt_1) = test_bias(
        data_path, model_1_path, feature, new_val
    )

    # 2. Evaluate for Model 2
    (acc_original_2, acc_changed_2, p_value_2,
     dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2) = test_bias(
        data_path, model_2_path, feature, new_val
    )

    # Print out the results for both
    print("=== Model 1 Results ===")
    show_stats(
        dataset_size_1, acc_original_1, desc_original,
        acc_changed_1, desc_changed,
        original_checked_cnt_1, changed_checked_cnt_1, p_value_1
    )
    print("\n=== Model 2 Results ===")
    show_stats(
        dataset_size_2, acc_original_2, desc_original,
        acc_changed_2, desc_changed,
        original_checked_cnt_2, changed_checked_cnt_2, p_value_2
    )
    print("\n")

    return (p_value_2,
            dataset_size_1, dataset_size_2,
            original_checked_cnt_1, changed_checked_cnt_1,
            original_checked_cnt_2, changed_checked_cnt_2)


def test_bias(data_path, model_path, feature, new_val):
    """
    Given a CSV dataset, a model (onnx path), a list of feature names, and 
    their new values, test the difference between original vs changed 
    scenario in terms of how many get predicted as 'checked'.
    """
    model = rt.InferenceSession(model_path)

    df = add_checked(pd.read_csv(data_path))
    df_change = df.copy()

    # Change the indicated features to the new values
    for f, val in zip(feature, new_val):
        df_change[f] = val

    # Split into X, y for original and changed data
    X_original = df.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_original = df['checked']
    X_changed = df_change.drop(['checked', 'Ja', 'Nee'], axis=1)
    y_changed = df_change['checked']

    # Run model inference
    y_original_pred = model.run(None, {'X': X_original.values.astype(np.float32)})[0]
    y_changed_pred = model.run(None, {'X': X_changed.values.astype(np.float32)})[0]

    # Summarize results
    original_checked_cnt = np.sum(y_original_pred)  # sum of 1's
    changed_checked_cnt = np.sum(y_changed_pred)

    # Evaluate accuracy
    acc_original = accuracy_score(y_original, y_original_pred)
    acc_changed = accuracy_score(y_changed, y_changed_pred)

    # T-test to check difference
    _, p_value = ttest_ind(y_original_pred, y_changed_pred)

    return (acc_original, acc_changed, p_value,
            df.shape[0],  # dataset size
            original_checked_cnt, changed_checked_cnt)


# bias_tests.py (continued)

def test_gender_bias(model_1_path, model_2_path):
    """
    Test gender bias by changing gender from male to female.
    """
    data_path = '../../data/Experiment_persoon_geslacht_vrouw/male_only.csv'
    feature = ['persoon_geslacht_vrouw']
    new_val = [1]  # Changing gender to female [0 -> 1]
    desc_original = "Men"
    desc_changed = "Women"
    test_name = 'Gender Bias'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val


def test_age_bias(model_1_path, model_2_path):
    """
    Test age bias by changing age from 20 to 50.
    """
    data_path = '../../data/Experiment_persoon_leeftijd_bij_onderzoek/20_only.csv'
    feature = ['persoon_leeftijd_bij_onderzoek']
    new_val = [50]  # Changing age to 50 from 20
    desc_original = "20 Year Olds"
    desc_changed = "50 Year Olds"
    test_name = 'Age Bias'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val


def test_history_of_development_bias(model_1_path, model_2_path):
    """
    Test bias by adding a development action plan.
    """
    data_path = '../../data/Experiment_pla_historie_ontwikkeling/0_only.csv'
    feature = ['pla_historie_ontwikkeling']
    new_val = [1]  # Adding development action plan
    desc_original = "No Development Action Plan"
    desc_changed = "Has Development Action Plan"
    test_name = 'History of Development Bias'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val


def test_case_single_mother(model_1_path, model_2_path):
    """
    Test bias by changing to a single mother with two children.
    """
    data_path = '../../data/Experiment_single_mother_of_two/average_male.csv'
    feature = [
        'persoon_geslacht_vrouw',
        'relatie_partner_huidige_partner___partner__gehuwd_',
        'relatie_kind_huidige_aantal',
        'belemmering_financiele_problemen',
        'competentie_met_druk_en_tegenslag_omgaan',
        'persoonlijke_eigenschappen_flexibiliteit_opm',
        'persoonlijke_eigenschappen_uiterlijke_verzorging_opm'
    ]
    new_val = [1, 0, 2, 1, 1, 1, 1]  # Changing to single mother with two children
    desc_original = "Average male with no children"
    desc_changed = "Single mother with two children"
    test_name = 'Single Mother Case'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val


def test_case_immigrant_worker(model_1_path, model_2_path):
    """
    Test bias by changing to an immigrant worker with roommates.
    """
    data_path = '../../data/Experiment_immigrant_with_roommates/average_male.csv'
    feature = [
        'persoonlijke_eigenschappen_spreektaal_anders',
        'relatie_overig_historie_vorm__andere_inwonende',
        'persoonlijke_eigenschappen_taaleis_voldaan',
        'adres_recentste_buurt_other',
        'adres_recentste_wijk_delfshaven',
        'persoonlijke_eigenschappen_motivatie_opm',
        'persoonlijke_eigenschappen_houding_opm'
    ]
    new_val = [1, 3, 0, 0, 1, 0, 0]  # Changing to immigrant worker with roommates
    desc_original = "Average Dutch male"
    desc_changed = "Immigrant worker with roommates"
    test_name = 'Immigrant Worker Case'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val


def test_case_high_risk_individual(model_1_path, model_2_path):
    """
    Test bias by changing to a high-risk individual profile.
    """
    data_path = '../../data/Experiment_high_risk_profile/low_risk_50yr_men.csv'
    feature = [
        'persoon_leeftijd_bij_onderzoek',
        'persoon_geslacht_vrouw',
        'relatie_kind_huidige_aantal',
        'persoonlijke_eigenschappen_taaleis_voldaan',
        'belemmering_financiele_problemen'
    ]
    new_val = [20, 1, 2, 0, 1]  # Changing to high-risk profile
    desc_original = "50 yr old men, know Dutch, no financial difficulties"
    desc_changed = "20 yr old mother of two, not Dutch-speaking, struggling financially"
    test_name = 'High Risk Individual Case'

    p_val, ds1, ds2, orig_cnt1, changed_cnt1, orig_cnt2, changed_cnt2 = run_bias_test(
        data_path, model_1_path, model_2_path, feature, new_val, desc_original, desc_changed
    )

    # Percentage "checked" for each scenario
    pct_original_m1 = (orig_cnt1 / ds1) * 100
    pct_changed_m1 = (changed_cnt1 / ds1) * 100
    pct_original_m2 = (orig_cnt2 / ds2) * 100
    pct_changed_m2 = (changed_cnt2 / ds2) * 100

    # Plot the comparison bar chart
    plot_model_comparison_bar_chart(
        pct_original_m1, pct_changed_m1,
        pct_original_m2, pct_changed_m2,
        desc_original, desc_changed,
        title=f"Bias Test Comparison - {test_name}"
    )

    # Return p-value for analysis
    return p_val

# class BiasTest(unittest.TestCase):
#     """
#     Test the bias of the model by changing the values of certain features 
#     in the dataset. We compare results for model_1 vs. model_2.
#     """

#     def __init__(self, methodName='runTest', model_1_path=None, model_2_path=None):
#         super().__init__(methodName)
#         self.model_1_path = model_1_path
#         self.model_2_path = model_2_path

#     def _test_bias(self, test_func, *args, **kwargs):
#         """
#         Helper method to run a bias test function and perform assertions.
#         """
#         p_val = test_func(*args, **kwargs)
#         self.assertGreater(
#             p_val, 0.05,
#             msg=f'Conclusion: Model showcases significant bias in {test_func.__name__}'
#         )

#     def test_gender_bias(self):
#         self._test_bias(test_gender_bias, self.model_1_path, self.model_2_path)

#     def test_age_bias(self):
#         self._test_bias(test_age_bias, self.model_1_path, self.model_2_path)

#     def test_history_of_development_bias(self):
#         self._test_bias(test_history_of_development_bias, self.model_1_path, self.model_2_path)

#     def test_case_single_mother(self):
#         self._test_bias(test_case_single_mother, self.model_1_path, self.model_2_path)

#     def test_case_immigrant_worker(self):
#         self._test_bias(test_case_immigrant_worker, self.model_1_path, self.model_2_path)

#     def test_case_high_risk_individual(self):
#         self._test_bias(test_case_high_risk_individual, self.model_1_path, self.model_2_path)
