import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import unittest
import onnxruntime as ort

from src.tests.test_utils import add_checked

'''
Run tests:
    Just in case if you forgot how to do it: python -m unittest <relative_test_path>
    (or python3 if you use MacOS).
'''


class BiasTest(unittest.TestCase):

    def setUp(self):
        self.new_session = ort.InferenceSession('path_to_your_model.onnx')

    def test_gender_haschildren(self):
        # Load the data
        df_gender = add_checked(pd.read_csv('data/examples/address_change.csv'))

        # # Predict for women
        women_pred = df_gender[df_gender['persoon_geslacht_vrouw'] == 1]
        X_women, y_women = women_pred.drop(['checked', 'Ja', 'Nee'], axis=1), women_pred['checked']
        y_women_pred = self.new_session.run(None, {'X': X_women.values.astype(np.float32)})[0]

        # # Predict for men
        X_men, y_men = women_pred.drop(['checked', 'Ja', 'Nee'], axis=1), women_pred['checked']
        X_men['persoon_geslacht_vrouw'] = 0
        y_men_pred = self.new_session.run(None, {'X': X_men.values.astype(np.float32)})[0]

        # acc_women = accuracy_score(y_women, y_women_pred)
        # acc_men = accuracy_score(y_men, y_men_pred)

        _, p_value = ttest_ind(y_women_pred, y_men_pred)
        self.assertGreater(p_value, 0.05, msg=f"Significant difference was found!")