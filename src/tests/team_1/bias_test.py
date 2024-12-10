import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import unittest
import onnxruntime as ort
from sklearn.metrics import accuracy_score


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


def add_checked(df):
    df['checked'] = df['Ja'].apply(lambda x: 1 if x > 0.7 else 0)
    return df


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
