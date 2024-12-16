import math
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

def plot_dist_checked(df, feature):
    """
    Plots the distribution of a feature's values, segmented by the 'checked' status.
    Also shows the counts of 'Checked True' and 'Checked False' within their corresponding bins.
    """
    # Ensure the feature exists in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in the DataFrame.")
    
    if 'checked' not in df.columns:
        raise ValueError("The DataFrame does not contain a 'checked' column.")
    
    # Drop rows with missing values in the feature or 'checked' columns
    df = df.dropna(subset=[feature, 'checked'])
    
    # Sort the feature column
    sorted_col = df[feature].sort_values()
    
    # Get value counts
    value_counts = sorted_col.value_counts(ascending=True)
    
    values = value_counts.index.tolist()  # Unique values
    counts = value_counts.values.tolist()  # Counts of each value
    
    sorted_values_counts = sorted(zip(values, counts))  # Sort pairs by values
    sorted_values, sorted_counts = zip(*sorted_values_counts)  # Unzip
    
    sorted_values = list(sorted_values)
    sorted_counts = list(sorted_counts)
    
    # Determine if binning is needed
    if len(sorted_values) >= 10:
        # Binning the feature into 10 equal-width bins
        num_bins = 10
        bins = np.linspace(min(sorted_values), max(sorted_values), num_bins + 1)
        bins = [int(round(num)) for num in bins]  # Convert bin edges to integers
        
        # Assign each value to a bin
        df['Bins'] = pd.cut(df[feature], bins=bins, include_lowest=True)
        
        # Aggregate counts by bins and 'checked' status
        binned_data = df.groupby(['Bins', 'checked']).size().unstack(fill_value=0)
        
        # Prepare labels and counts for plotting
        binned_values = [f"{int(round(interval.left))}-{int(round(interval.right))}" for interval in binned_data.index]
        checked_true = binned_data.get(True, pd.Series([0]*len(binned_data))).values
        checked_false = binned_data.get(False, pd.Series([0]*len(binned_data))).values
        
        # Example calc call (adjust as needed)
        total_counts = checked_true + checked_false
        # print(calc(binned_values, total_counts))  # Uncomment if calc is defined elsewhere
        
        # Create the stacked bar chart
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        indices = np.arange(len(binned_values))
        
        # Plot 'checked' == True
        plt.bar(indices, checked_true, bar_width, label='Checked True', color='skyblue')
        
        # Plot 'checked' == False on top of 'checked' == True
        plt.bar(indices, checked_false, bar_width, bottom=checked_true, label='Checked False', color='salmon')
        
        # Set x-axis labels and positions
        plt.xticks(indices, binned_values, rotation=45)
        
        # Overlay count labels for each segment
        for i in range(len(binned_values)):
            true_count = checked_true[i]
            false_count = checked_false[i]
            total = true_count + false_count

            # Total count above the stack
            plt.text(i, total + max(total_counts)*0.01,
                     str(total), 
                     ha='center', 
                     va='bottom', 
                     fontsize=9, 
                     color='black')

            # Checked True count (inside the bottom bar if > 0)
            if true_count > 0:
                plt.text(i, true_count/2,
                         str(true_count),
                         ha='center',
                         va='center',
                         fontsize=9,
                         color='black')
            
            # Checked False count (inside the top bar if > 0)
            if false_count > 0:
                plt.text(i, true_count + false_count/2,
                         str(false_count),
                         ha='center',
                         va='center',
                         fontsize=9,
                         color='black')
        
    else:
        # No binning needed; plot each unique value separately
        grouped_data = df.groupby([feature, 'checked']).size().unstack(fill_value=0)
        grouped_data = grouped_data.reindex(sorted_values, fill_value=0)
        
        checked_true = grouped_data.get(True, pd.Series([0]*len(grouped_data))).values
        checked_false = grouped_data.get(False, pd.Series([0]*len(grouped_data))).values
        
        total_counts = checked_true + checked_false
        # print(calc(sorted_values, total_counts))  # Uncomment if calc is defined
        
        # Create the stacked bar chart
        plt.figure(figsize=(12, 6))
        bar_width = 0.6
        indices = np.arange(len(sorted_values))
        
        plt.bar(indices, checked_true, bar_width, label='Checked True', color='skyblue')
        plt.bar(indices, checked_false, bar_width, bottom=checked_true, label='Checked False', color='salmon')
        
        # Set x-axis labels and positions
        plt.xticks(indices, sorted_values, rotation=45)
        
        # Overlay count labels for each segment
        for i in range(len(sorted_values)):
            true_count = checked_true[i]
            false_count = checked_false[i]
            total = true_count + false_count

            # Total count above the stack
            plt.text(i, total + max(total_counts)*0.01,
                     str(total), 
                     ha='center', 
                     va='bottom', 
                     fontsize=9, 
                     color='black')

            # Checked True count (inside the bottom bar if > 0)
            if true_count > 0:
                plt.text(i, true_count/2,
                         str(true_count),
                         ha='center',
                         va='center',
                         fontsize=9,
                         color='black')
            
            # Checked False count (inside the top bar if > 0)
            if false_count > 0:
                plt.text(i, true_count + false_count/2,
                         str(false_count),
                         ha='center',
                         va='center',
                         fontsize=9,
                         color='black')
    
    # Add labels and title
    plt.xlabel(feature)
    plt.ylabel('Counts')
    plt.title(f'Distribution of {feature} by Checked Status')
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    plt.show()