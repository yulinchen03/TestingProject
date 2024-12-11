import onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from src.utils.test_utils import *


def load_data(path):
    # specify existing model path
    model_path = "../../../model/gboost2.onnx"
    df = add_checked(pd.read_csv(path))
    X = df.drop(['checked', 'Ja', 'Nee'], axis=1)
    X = X.astype(np.float32)
    y = df['checked']

    return X, y


# Manipulate the data to reduce/increase bias
def data_manipulator(data, feature, manipulation, fraction,  overwrite_value=None, remove_condition=None):
    """
    :param data: The data to be manipulated
    :param feature: The feature to be manipulated
    :param manipulation: manipulation type:
        - 'overwrite': Overwrites values in specified feature and ensures the fraction is satisfied.
        - 'remove': Remove a fraction of rows where the feature has the specified value
    :param fraction: Fraction of data to be manipulated
    :param overwrite_value: The value to be overwritten
    :param remove_condition: The condition needed to be satisfied to be removed
    """

    if not (0 < fraction < 1):
        raise ValueError('fraction must be between 0 and 1')

    manipulated_data = data.copy()

    if manipulation == 'overwrite':
        # calculate how many rows are needed to overwrite to satisfy fraction

        target_count = int(len(manipulated_data) * fraction)
        curr_count = (manipulated_data[feature] == overwrite_value).sum()

        if curr_count < target_count:
            # how many more to overwrite
            rows_to_change = target_count - curr_count
            idx_to_change = manipulated_data[manipulated_data[feature] != overwrite_value].sample(
                n=rows_to_change, random_state=42).index
            manipulated_data.loc[idx_to_change, feature] = overwrite_value

    elif manipulation == 'remove':
        # rows that match condition
        matching_rows = manipulated_data[manipulated_data.apply(remove_condition, axis=1)]

        # how many to remove
        rows_to_remove = int(len(matching_rows) * fraction)

        if rows_to_remove > 0:
            idx_to_remove = matching_rows.sample(n=rows_to_remove, random_state=42).index
            manipulated_data = manipulated_data.drop(idx_to_remove)

    return manipulated_data

def retrain(X, y):
    model_path = "model/gboost2.onnx" # replace with gboost2.onnx if you are working on the bad model
    selector = VarianceThreshold()
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
    pipeline = Pipeline(steps=[('feature selection', selector), ('classification', classifier)])

    # Let's train a simple model
    pipeline.fit(X, y)

    # Let's convert the model to ONNX
    onnx_model = convert_sklearn(
        pipeline, initial_types=[('X', FloatTensorType((None, X.shape[1])))],
        target_opset=12)

    # Let's save the model
    onnx.save(onnx_model, model_path)
    print(f'Model successfully saved to {model_path}')

def run(path):
    # load data
    print(f'Loading dataset from {path}.....')
    X, y = load_data(path)

    # manipulate data
    print('Manipulating data.....')
    # data_manipulator()

    # retrain model
    print(f'Retraining model.....')
    retrain(X, y)


# path = '../../../data/synth_data_train_labeled.csv'
# run(path)

