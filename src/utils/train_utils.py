import onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, accuracy_score
import os
import re


def get_versioned_name(dir_path, filename_prefix, file_extension, create_new=False):
    highest_id = None
    id_pattern = re.compile(rf"^{re.escape(filename_prefix)}(\d+){re.escape(file_extension)}$")

    for filename in os.listdir(dir_path):
        match = id_pattern.match(filename)
        if match:
            file_id = int(match.group(1))
            if highest_id is None or file_id > highest_id:
                highest_id = file_id

    if create_new:
        highest_id += 1

    return filename_prefix + str(highest_id) + file_extension

def filter_features(features, keywords=None):
    '''
    Filter out certain biased/useless features of the dataset
    Namely:
    addres-recentste: recent neighborhood of the customer
    geslacht: Gender
    taal: Language
    persoonlijke_eigenschappen: Personal qualities
    :param features: list of features from the dataset
    :param keywords: specific keywords to filter out features
    :return: list of features that need to be filtered out
    '''
    if keywords is None:
        keywords = ["adres_recentste", "geslacht", "_taal", "persoonlijke_eigenschappen"]
    return [feature for feature in features if any(keyword in feature for keyword in keywords)]


def train(X, y, model_path):
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'learning_rate': [0.1, 0.3, 0.5],
        'max_depth': [1, 3, 5],
    }

    selector = VarianceThreshold()
    classifier = GradientBoostingClassifier(n_estimators=500, random_state=1, verbose=1)

    # Define the cross-validation method
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=1)

    # Perform Grid Search
    grid_search.fit(X, y)

    # Get the best parameters
    best_params = grid_search.best_params_

    # Get the mean and standard deviation of accuracy from the cross-validation
    mean_accuracy = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    std_accuracy = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    # Output the results
    print(f"Best Parameters: {best_params}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

    # pipeline = Pipeline(steps=[('feature selection', selector), ('classification', classifier)])
    # pipeline.set_params(**best_params)
    #
    # # Let's train a simple model
    # pipeline.fit(X, y)

    # # Let's convert the model to ONNX
    # onnx_model = convert_sklearn(
    #     pipeline, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
    #     target_opset=12)

    # # Let's save the model
    # onnx.save(onnx_model, model_path)
    # print(f'Model successfully saved to {model_path}.')


def run(X, y, model_path):
    # retrain model
    print(f'Training model.....')
    train(X, y, model_path)

