import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import unittest
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from src.utils.test_utils import *


'''
IMPORTANT! READ BEFORE WRITING TESTS

Naming convention:
    Tests should be named as follows: test_root_featureA_featureB_.... The root is the feature you want to test.
    The features A, B, ... are the features that you want to test in relation to the root feature (tree branches).
    If you still have no idea what I'm talking about, check the example below.

How to write tests (WTF is going on):
    Each test tests the bias in specific subset of features. For more information,
    check this: https://www.wired.com/story/welfare-state-algorithms/.
    What they do is show how rating of a person will change if some/multiple parameters
    will change. So, this is what our tests should do. For example, if we choose subset
    that contains gender with particular additional feature (like haschildren), we can test
    how the rating will change if we swap the gender in a subset of haschildren=true.
    You root is gender, and feature_to_test is has_children. You can also test only root features,
    then the test name will be test_root (e.g. test_gender).
    Example test provided below.

Data generation:
    Before writing tests, you need to generate data. You can use the data generation script
    provided here: https://github.com/abuszydlik/Social-Welfare-Dataset/blob/main/DataManual.md.
    Essentially, you can choose specific features and their values, and generate data
    from distribution. Use the same naming convention for tests like "data_root_featureA_featureB_...".

Run tests:
    Just in case if you forgot how to do it: python -m unittest <relative_test_path>
    (or python3 if you use MacOS).

GOOD FUCKING LUCK!
'''


def _show_stats(dataset_size, acc_original, f_name_original, acc_changed, f_name_changed, original_checked_cnt, changed_checked_cnt, p_val):
    print(f'Accuracy for sample of {dataset_size} {f_name_original}: {acc_original * 100:.1f}%')
    print(f'Accuracy for sample of {dataset_size} {f_name_changed}: {acc_changed * 100:.1f}%')
    print(f'Percentage checked amongst {f_name_original}: {original_checked_cnt * 100 / dataset_size:.1f}%')
    print(f'Percentage checked changed to {f_name_changed}: {changed_checked_cnt * 100 / dataset_size:.1f}%')
    print(f'P value: {p_val}')


def _run_bias_test(data_path, model_path, modified_model_path, feature, new_val, desc_original, desc_changed):
    # Run bias test on the original model
    acc_original, acc_changed, p_value_1, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
        data_path, model_path, feature, new_val)

    # Run bias test on the modified model
    acc_original_2, acc_changed_2, p_value_2, dataset_size_2, original_checked_cnt_2, changed_checked_cnt_2 = test_bias(
        data_path, modified_model_path, feature, new_val)

    print("=== Original Model Results ===")
    _show_stats(dataset_size, acc_original, desc_original, acc_changed, desc_changed, original_checked_cnt, changed_checked_cnt, p_value_1)
    print("\n=== Modified Model Results ===")
    _show_stats(dataset_size_2, acc_original_2, desc_original, acc_changed_2, desc_changed, original_checked_cnt_2, changed_checked_cnt_2, p_value_2)

    return p_value_2


class BiasTest(unittest.TestCase):

    def setUp(self):
        self.model_path = "../../../model/gboost1_v0.onnx"  # replace with gboost2.onnx if you are working on the bad model
        self.modified_model_path = "../../../model/gboost1_v2.onnx"

    def _test_bias(self, data_path, feature, new_val, desc_original, desc_changed):
        p_value_2 = _run_bias_test(data_path, self.model_path, self.modified_model_path, feature, new_val,
                                   desc_original, desc_changed)
        self.assertGreater(p_value_2, 0.05,
                           msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')

    def test_gender_bias(self):
        self._test_bias(
            data_path='../../../data/Experiment_persoon_geslacht_vrouw/male_only.csv',
            feature=['persoon_geslacht_vrouw'],
            new_val=[1],  # Changing gender to female [0 -> 1]
            desc_original="Men",
            desc_changed="Women"
        )

    def test_age_bias(self):
        self._test_bias(
            data_path='../../../data/Experiment_persoon_leeftijd_bij_onderzoek/20_only.csv',
            feature=['persoon_leeftijd_bij_onderzoek'],
            new_val=[50],  # Changing age to 50 from 20
            desc_original="20 Year Olds",
            desc_changed="50 Year Olds"
        )

    def test_history_of_development_bias(self):
        self._test_bias(
            data_path='../../../data/Experiment_pla_historie_ontwikkeling/0_only.csv',
            feature=['pla_historie_ontwikkeling'],
            new_val=[1],  # Adding development action plan
            desc_original="People with no development action plan",
            desc_changed="People with a development action plan"
        )

    def test_case_single_mother(self):
        """Test case for a single mother of two."""
        self._test_bias(
            data_path='../../../data/Experiment_single_mother_of_two/average_male.csv',
            feature=['persoon_geslacht_vrouw', 'relatie_partner_huidige_partner___partner__gehuwd_',
                     'relatie_kind_huidige_aantal', 'belemmering_financiele_problemen',
                     'competentie_met_druk_en_tegenslag_omgaan', 'persoonlijke_eigenschappen_flexibiliteit_opm',
                     'persoonlijke_eigenschappen_uiterlijke_verzorging_opm'],
            new_val=[1, 0, 2, 1, 1, 1, 1],  # Changing to single mother with two children
            desc_original="Average male with no children",
            desc_changed="Single mother with two children"
        )

    def test_case_immigrant_worker(self):
        """Test case for an immigrant worker with roommates."""
        self._test_bias(
            data_path='../../../data/Experiment_immigrant_with_roommates/average_male.csv',
            feature=['persoonlijke_eigenschappen_spreektaal_anders', 'relatie_overig_historie_vorm__andere_inwonende',
                     'persoonlijke_eigenschappen_taaleis_voldaan', 'adres_recentste_buurt_other', 'adres_recentste_wijk_delfshaven',
                     'persoonlijke_eigenschappen_motivatie_opm', 'persoonlijke_eigenschappen_houding_opm'],
            new_val=[1, 3, 0, 0, 1, 0, 0],  # Changing to immigrant worker with roommates
            desc_original="Average Dutch male",
            desc_changed="Immigrant worker with roommates"
        )

    def test_case_high_risk_individual(self):
        """Test case for a high-risk individual."""
        self._test_bias(
            data_path='../../../data/Experiment_high_risk_profile/low_risk_50yr_men.csv',
            feature=['persoon_leeftijd_bij_onderzoek', 'persoon_geslacht_vrouw',
                     'relatie_kind_huidige_aantal', 'persoonlijke_eigenschappen_taaleis_voldaan',
                     'belemmering_financiele_problemen'],
            new_val=[20, 1, 1, 0, 1],  # Changing to high-risk profile
            desc_original="50 year old men who know Dutch and have no financial difficulties",
            desc_changed="20 year old mother of two who does not know Dutch and struggles to pay bills"
        )
