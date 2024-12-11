import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import unittest
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from src.utils.test_utils import *


class BiasTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_young_age_bias(self):
        # change this with your dataset's path
        data_path = '../../../data/Experiment_persoon_leeftijd_bij_onderzoek/20_only.csv'
        model_path = "../../../model/gboost2.onnx" # replace with gboost2.onnx if you are working on the bad model

        # change this when you want to test a different feature
        feature = 'persoon_leeftijd_bij_onderzoek'

        new_vals_range = (26, 68)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias_with_range(
            data_path, model_path, feature, new_vals_range)  # DO NOT CHANGE

        # Results messages (replace it with your own)
        print(f'Accuracy for sample of {dataset_size} Men: {acc_original * 100:.1f}%')
        print(f'Accuracy for sample of {dataset_size} Women: {acc_changed * 100:.1f}%')
        print(f'Percentage checked amongst 1000 men: {original_checked_cnt * 100 / dataset_size:.1f}%')
        print(f'Percentage checked when gender is changed to female: {changed_checked_cnt * 100 / dataset_size:.1f}%')

        self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')