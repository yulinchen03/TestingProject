import os
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Optional: If xgboost is available
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

a = time.time()

# Load the dataset
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir, '..', '..', 'data',
                        'synth_data_for_training.csv')

# Load the data
df = pd.read_csv(data_dir, encoding='ISO-8859-1')

# Load and map feature names
english_names_dir = os.path.join(
    curr_dir, '..', '..', 'data', 'data_description.csv')
names_df = pd.read_csv(english_names_dir, encoding='ISO-8859-1')
name_mapping = dict(zip(names_df['Feature (nl)'], names_df['Feature (en)']))

df.rename(columns=name_mapping, inplace=True)

# Define features and target
y = df['checked']
X = df.drop(['checked'], axis=1).astype(np.float32)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

selector = VarianceThreshold()

# Define pipelines
models = {
    'LogisticRegression': Pipeline([
        ('feature_selection', selector),
        ('scaler', StandardScaler()),
        ('classification', LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'SVC': Pipeline([
        ('feature_selection', selector),
        ('scaler', StandardScaler()),
        ('classification', SVC(probability=True, random_state=42))
    ]),
    'RandomForest': Pipeline([
        ('feature_selection', selector),
        ('classification', RandomForestClassifier(random_state=42))
    ]),
    'GradientBoosting': Pipeline([
        ('feature_selection', selector),
        ('classification', GradientBoostingClassifier(random_state=42))
    ])
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = Pipeline([
        ('feature_selection', selector),
        ('classification', XGBClassifier(
            use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])

# Define hyperparameter grids
param_grids = {
    'LogisticRegression': {
        'classification__C': np.logspace(-4, 4, 10),
        'classification__solver': ['liblinear', 'lbfgs'],
        'classification__penalty': ['l2']
    },
    'SVC': {
        'classification__C': [0.1, 1, 10],
        'classification__kernel': ['linear'],
        'classification__gamma': ['scale']
    },
    'RandomForest': {
        'classification__n_estimators': [50, 100, 150],
        'classification__max_depth': [3, 5, 7, None],
        'classification__max_features': ['auto', 'sqrt', 'log2']
    },
    'GradientBoosting': {
        'classification__n_estimators': [50, 100, 150],
        'classification__learning_rate': [0.01, 0.1, 0.2, 0.3],
        'classification__max_depth': [1, 3, 5]
    }
}

# Perform RandomizedSearchCV for each model, except SVC (takes a while...)
best_models = {}
for model_name, model_pipeline in models.items():
    if model_name in param_grids:
        if model_name == "SVC":
            continue  # With SVC This takes a long time, uncomment this if you want to try
        print(f"Performing RandomizedSearchCV for {model_name}...")
        search = RandomizedSearchCV(
            model_pipeline,
            param_distributions=param_grids[model_name],
            n_iter=3,  # Tune this for more exhaustive searches
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        print(f"Best params for {model_name}: {search.best_params_}\n")
    else:
        # Directly fit models without search if no params defined
        print(f"Training {model_name} with default params...")
        model_pipeline.fit(X_train, y_train)
        best_models[model_name] = model_pipeline


print("\nEvaluating all models on the test set...\n")

# Use SMOTE if desired
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

model_scores = {}

for model_name, model_pipeline in best_models.items():
    model_pipeline.fit(X_res, y_res)
    y_pred = model_pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[model_name] = acc
    print(f"{model_name}: Accuracy = {acc:.4f}")

# Determine the best model
best_model_name = max(model_scores, key=model_scores.get)
print("\nBest model based on test set accuracy:", best_model_name,
      "with accuracy:", model_scores[best_model_name])

b = time.time()
print("Total Runtime (seconds):", b - a)
