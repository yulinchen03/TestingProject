import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler


def filter_features(features, keywords):
    return [feature for feature in features if any(keyword in feature for keyword in keywords)]


class FeatureAnalyzer:

    def __init__(self):
        self.X = None
        self.y = None
        self.feature_importance = None

    def load_importance(self, filepath="feature_importance.pkl"):
        with open(filepath, "rb") as f:
            self.feature_importance = pickle.load(f)

    def evaluate_importance(self,
                            dataframe,
                            target,
                            add_drop=[],
                            model=None,
                            n_repeats=5,
                            n_jobs=-1,
                            random_state=None,
                            cache=True,
                            filename="feature_importance.pkl"):

        scaler = MinMaxScaler()
        dataframe_norm = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)

        self.X = dataframe_norm.drop([target] + add_drop, axis=1)
        self.y = dataframe_norm[target]

        if model is None:
            model = GradientBoostingClassifier(random_state=random_state)
        model.fit(self.X, self.y)

        feature_importance = permutation_importance(
            model, self.X, self.y, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )

        if cache:
            with open(filename, "wb") as f:
                pickle.dump(feature_importance, f)

        self.feature_importance = feature_importance

    def feature_importance_as_dict(self, column_names=None, normalize=False):
        if self.X is not None:
            column_names = self.X.columns
        elif column_names is None:
            raise ValueError("column_names must be provided")
        else:
            features_mean = self.feature_importance.importances_mean
            # features_std = self.feature_importance.importances_std

            if normalize:
                scaler = MinMaxScaler()
                features_mean = scaler.fit_transform(features_mean.reshape(-1, 1)).flatten()
                # features_std = scaler.fit_transform(features_std.reshape(-1, 1)).flatten()

            # return {col: [mean, std] for col, mean, std in zip(column_names, features_mean, features_std)}
            return {col: mean for col, mean in zip(column_names, features_mean)}

    def plot_importance(self, column_names=None, min_val=0):
        if self.X is not None:
            column_names = self.X.columns
        elif column_names is None:
            raise ValueError("column_names must be provided")
        else:
            scaler = MinMaxScaler()
            features_mean = scaler.fit_transform(self.feature_importance.importances_mean.reshape(-1, 1)).flatten()
            features_std = scaler.fit_transform(self.feature_importance.importances_std.reshape(-1, 1)).flatten()

            threshold = features_mean > min_val
            filtered_features_mean = features_mean[threshold]
            filtered_features_std = features_std[threshold]
            filtered_features_names = column_names[threshold]

            importances_mean = pd.Series(filtered_features_mean, index=filtered_features_names)

            fig, ax = plt.subplots()
            importances_mean.plot.bar(yerr=filtered_features_std, ax=ax)
            importances_mean.plot.bar(ax=ax)
            ax.set_title("Feature importances using permutation on full model")
            ax.set_ylabel("Mean accuracy decrease")
            fig.tight_layout()
            plt.show()
