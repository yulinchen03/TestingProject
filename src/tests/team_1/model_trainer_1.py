import onnx
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from onnxconverter_common import FloatTensorType
from skl2onnx import convert_sklearn
from src.tests.test_utils import *


def load_data(path):
    # specify existing model path
    model_path = "../../../model/gboost1.onnx"
    df = add_checked(pd.read_csv(path))
    X = df.drop(['checked', 'Ja', 'Nee'], axis=1)
    X = X.astype(np.float32)
    y = df['checked']

    return X, y


# Manipulate the data to reduce/increase bias
def data_manipulator():
    ########INSERT DATA MANIPULATION CODE HERE##########

    pass

    ####################################################


def retrain(X, y):
    model_path = "../../../model/gboost1.onnx" # replace with gboost2.onnx if you are working on the bad model
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


path = '../../../data/synth_data_train_labeled.csv'
run(path)

