import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def translate_cols(df, top_features):
    english_names_dir = "../../data/data_description.csv"
    names_df = pd.read_csv(english_names_dir, encoding='ISO-8859-1')
    name_mapping = dict(zip(names_df['Feature (nl)'], names_df['Feature (en)']))
    top_features_en = dict()

    # Rename the columns in the `df` DataFrame
    df.rename(columns=name_mapping, inplace=True)

    for feature, importance in top_features.items():
        feature_en = names_df[names_df['Feature (nl)'] == feature]['Feature (en)'].iloc[0]
        top_features_en[feature_en] = importance / sum(list(top_features.values()))

    return df, top_features_en


'''
calculate the difference between the mean of values of a feature 
and the current count of a specific value of that feature (for re-balancing)
'''
def calc(sorted_values, sorted_counts):
    to_add = {}
    maximum = np.max(sorted_counts)
    for value, count in zip(sorted_values, sorted_counts):
        if count < maximum:
            to_add[value] = int(math.floor((maximum - count) / 100) * 100)
    return to_add


def plot_distribution(col, feature):
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
        binned_data = data.groupby('Bins', observed=True)['Counts'].sum()

        # Convert binned data for plotting
        binned_values = [f"{int(round(interval.left))}-{int(round(interval.right))}" for interval in binned_data.index]
        binned_counts = binned_data.values

        print(calc(binned_values, binned_counts))

        # Create the bar chart
        plt.figure(figsize=(12, 5))  # Width: 10, Height: 5
        plt.bar(binned_values, binned_counts, color='skyblue')
        plt.xticks(binned_values)  # This ensures only the existing values are displayed

        # Overlay count labels
        for i, count in enumerate(binned_counts):
            plt.text(binned_values[i], count + 1,  # Position slightly above the bar
                     str(count),  # Text to display
                     ha='center',  # Center the text horizontally
                     va='bottom',  # Align text to the bottom
                     fontsize=10,  # Font size
                     color='black')  # Text color
    else:
        print(calc(sorted_values, sorted_counts))

        # Create the bar chart
        plt.figure(figsize=(12, 5))  # Width: 10, Height: 5
        plt.bar(sorted_values, sorted_counts, color='skyblue')
        plt.xticks(sorted_values)  # This ensures only the existing values are displayed

        # Overlay count labels
        for i, count in enumerate(sorted_counts):
            plt.text(sorted_values[i], count + 1,  # Position slightly above the bar
                     str(count),  # Text to display
                     ha='center',  # Center the text horizontally
                     va='bottom',  # Align text to the bottom
                     fontsize=10,  # Font size
                     color='black')  # Text color

    # Add labels and title
    plt.xlabel(feature)
    plt.ylabel('Counts')
    plt.title('Distribution of feature: ' + feature)

    # Show the plot
    plt.show()