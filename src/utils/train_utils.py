import os
import onnx
import re
from onnxconverter_common import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from skl2onnx import convert_sklearn


def get_versioned_name(dir_path, filename_prefix, file_extension, create_new=False):
    """
    Generate a versioned filename in the given directory by incrementing an integer suffix.
    :param dir_path: Path to the directory where the file is located.
    :param filename_prefix: Prefix of the filename.
    :param file_extension: File extension, e.g., '.onnx'.
    :param create_new: Whether to create a new versioned file.
    :return: The versioned filename.
    """
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' does not exist.")

    highest_id = -1
    id_pattern = re.compile(rf"^{re.escape(filename_prefix)}(\d+){re.escape(file_extension)}$")

    for filename in os.listdir(dir_path):
        match = id_pattern.match(filename)
        if match:
            highest_id = max(highest_id, int(match.group(1)))

    if create_new:
        highest_id += 1

    return f"{filename_prefix}{highest_id}{file_extension}"

def filter_features(features, keywords=None):
    """
    Filter out features based on specified keywords.
    :param features: List of feature names from the dataset.
    :param keywords: Specific keywords to filter out features. Defaults to a predefined list.
    :return: A list of features that match the keywords.
    """
    if keywords is None:
        keywords = ["adres_recentste", "geslacht", "_taal", "persoonlijke_eigenschappen"]
    return [feature for feature in features if any(keyword in feature for keyword in keywords)]


def train(X, y, model_path, random_state=42, verbose=True):
    """
    Train a Gradient Boosting Classifier using GridSearchCV and save the model in ONNX format.
    :param X: Feature matrix.
    :param y: Target vector.
    :param model_path: Path to save the ONNX model.
    :param random_state: Random seed for reproducibility.
    :param verbose: Whether to print model statistics.
    """
    param_grid = {
        'classification__learning_rate': [0.3],  # Adjust as needed
        'classification__max_depth': [3],
    }

    pipeline = Pipeline([
        ('feature_selection', VarianceThreshold()),
        ('classification', GradientBoostingClassifier(n_estimators=500, random_state=1, verbose=1))
    ])

    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    mean_accuracy = grid_search.cv_results_['mean_test_score'][grid_search.best_index_]
    std_accuracy = grid_search.cv_results_['std_test_score'][grid_search.best_index_]

    if verbose:
        print("=== Model Summary ===")
        print(f"Best Parameters: {best_params}")
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        print(f"Standard Deviation of Accuracy: {std_accuracy:.4f}")

    return grid_search


def run(X, y, model_path):
    print(f'Training model...')
    trained_model = train(X, y, model_path)

    print("\nConverting to ONNX...")
    initial_type = [('X', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(trained_model, initial_types=initial_type, target_opset=12)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
    onnx.save(onnx_model, model_path)
    print(f"\nONNX model successfully saved to {model_path}.")
