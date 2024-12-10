import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import unittest
import onnxruntime as ort
from sklearn.metrics import accuracy_score
from src.tests.test_utils import *


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

class BiasTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_gender_bias(self):
        # change this with your dataset's path
        data_path = '../../../data/Experiment_persoon_geslacht_vrouw/male_only.csv'
        model_path = "../../../model/gboost.onnx"

        # change this when you want to test a different feature
        feature = 'persoon_geslacht_vrouw'

        new_val = 1  # 0 -> 1 for women, set this to the value you want to test for (e.g Age 40 -> Age 20)

        acc_original, acc_changed, p_value, dataset_size, original_checked_cnt, changed_checked_cnt = test_bias(
            data_path, model_path, feature, new_val)  # DO NOT CHANGE

        # Results messages (replace it with your own)
        print(f'Accuracy for sample of {dataset_size} Men: {acc_original * 100:.1f}%')
        print(f'Accuracy for sample of {dataset_size} Women: {acc_changed * 100:.1f}%')
        print(f'Percentage checked amongst 1000 men: {original_checked_cnt * 100 / dataset_size:.1f}%')
        print(f'Percentage checked when gender is changed to female: {changed_checked_cnt * 100 / dataset_size:.1f}%')

        self.assertGreater(p_value, 0.05, msg=f'Conclusion: Model showcases significant bias towards feature - {feature}')