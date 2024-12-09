import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import skew


class DataProcessor:
    def __init__(self, X, y, most_important_features):
        self.X = X
        self.X_reduced = None
        self.X_balanced = None
        self.y_balanced = None
        self.y = y
        self.most_important_features = most_important_features


    def count_distribution(self, col):
        col = col.sort_values()
        value_counts = col.value_counts(ascending=True)

        values = value_counts.index.tolist()  # Unique values
        counts = value_counts.values.tolist()  # Counts of each value

        sorted_values_counts = sorted(zip(values, counts))  # Sort pairs by values
        sorted_values, sorted_counts = zip(*sorted_values_counts)

        sorted_values = list(sorted_values)
        sorted_counts = list(sorted_counts)

        if (len(sorted_values) >= 10):
            # Combine values and counts into a DataFrame
            data = pd.DataFrame({'Values': sorted_values, 'Counts': sorted_counts})

            # Create 10 bins
            bins = np.linspace(min(values), max(values), 11)  # 10 bins means 11 edges
            bins = [int(round(num)) for num in bins]
            data['Bins'] = pd.cut(data['Values'], bins=bins, include_lowest=False)

            # Aggregate counts by bins
            binned_data = data.groupby('Bins', observed=False)['Counts'].sum()

            # Convert binned data for plotting
            binned_values = [f"{int(round(interval.left))}-{int(round(interval.right))}" for interval in
                             binned_data.index]
            binned_counts = binned_data.values

            return binned_values, binned_counts

        return sorted_values, sorted_counts


    def plot_distributions(self, feature, values, counts, title='Distribution of feature: ', color='skyblue'):
        # Create the bar chart
        plt.figure(figsize=(10, 5))  # Width: 10, Height: 5
        plt.bar(values, counts, width=0.5, color=color)
        plt.xticks(values)  # This ensures only the existing values are displayed

        # Overlay count labels
        for i, count in enumerate(counts):
            plt.text(values[i], count + 1,  # Position slightly above the bar
                     str(count),  # Text to display
                     ha='center',  # Center the text horizontally
                     va='bottom',  # Align text to the bottom
                     fontsize=10,  # Font size
                     color='black')  # Text color

        # Add labels and title
        plt.xlabel(feature)
        plt.ylabel('Counts')
        plt.title(title + feature)

        # Show the plot
        plt.show()

    def entropy(self, counts):
        total = sum(counts)
        proportions = [count / total for count in counts]
        return -sum(p * np.log2(p) for p in proportions if p > 0)

    def coefficient_of_variation(self, counts):
        mean = np.mean(counts)
        std_dev = np.std(counts)
        return std_dev / mean

    def compute_stats(self, counts):
        entropy = self.entropy(counts)
        cov = self.coefficient_of_variation(counts)
        return entropy, cov


    def preprocess(self, remove_features=True, class_balancing=True, visualize=False):
        # feature reduction
        if remove_features:
            print(f"Dataframe before dimensionality reduction: {self.X.shape}")
            self.X_reduced = self.X.loc[:, self.X.columns.isin(self.most_important_features)]
            print(f"Dataframe after dimensionality reduction: {self.X_reduced.shape}\n")
        else:
            self.X_reduced = self.X

        # class balancing
        if class_balancing:
            # Step 1: Undersample the majority class
            rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)  # Keep 50% of the majority
            self.X_balanced, self.y_balanced = rus.fit_resample(self.X_reduced, self.y)

            # Step 2: Oversample the minority class
            smote = SMOTE(sampling_strategy=0.75, random_state=42)  # Minority class becomes 75% of majority
            self.X_balanced, self.y_balanced = smote.fit_resample(self.X_balanced, self.y_balanced)

            print("\nAfter resampling:\n")
            y_values, y_counts = self.count_distribution(self.y)
            y_balanced_values, y_balanced_counts = self.count_distribution(self.y_balanced)

            if visualize:
                self.plot_distributions('checked', y_values, y_counts,
                                        title='Distribution of class labels before balancing: ')
                self.plot_distributions('checked', y_balanced_values, y_balanced_counts,
                                        title='Distribution of class labels after balancing: ', color='gold')

            y_entropy, y_cov = self.compute_stats(y_counts)
            y_balanced_entropy, y_balanced_cov = self.compute_stats(y_balanced_counts)

            print(f'-----------------------------------------------------------\n'
                  f'Measurement of class balance before and after balancing:\n'
                  f'Before\n'
                  f'Entropy: {y_entropy}\n'
                  f'Coefficient of variation: {y_cov}\n'
                  f'\n'
                  f'After\n'
                  f'Entropy: {y_balanced_entropy}\n'
                  f'Coefficient of variation: {y_balanced_cov}\n'
                  f'-----------------------------------------------------------\n')

            for feature in self.most_important_features:
                feature_values, feature_counts = self.count_distribution(self.X_reduced[feature])
                feature_balanced_values, feature_balanced_counts = self.count_distribution(self.X_balanced[feature])

                if visualize:
                    self.plot_distributions(feature, feature_values, feature_counts,
                                            title='Distribution of feature values before balancing: ')
                    self.plot_distributions(feature, feature_balanced_values, feature_balanced_counts,
                                            title='Distribution of feature values after balancing: ', color='gold')

                entropy, cov = self.compute_stats(feature_counts)
                balanced_entropy, balanced_cov = self.compute_stats(feature_balanced_counts)

                print(f'-----------------------------------------------------------\n'
                      f'Measurement of feature balance ({feature}) before and after balancing:\n'
                      f'Before\n'
                      f'Entropy: {entropy}\n'
                      f'Coefficient of variation: {cov}\n'
                      f'\n'
                      f'After\n'
                      f'Entropy: {balanced_entropy}\n'
                      f'Coefficient of variation: {balanced_cov}\n'
                      f'-----------------------------------------------------------\n')

        else:
            self.X_balanced = self.X_reduced
            self.y_balanced = self.y

        return self.X_balanced, self.y_balanced
