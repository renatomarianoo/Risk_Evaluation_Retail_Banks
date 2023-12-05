"""
Created on 26/11/2023

@author: renato.mariano
"""

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def custom_info(df, top_n=5):
    """
    Display custom information about the DataFrame including data types, null values, basic statistics,
    and top values for each column.

    Parameters:
    - data_frame (pd.DataFrame): The input DataFrame.
    - top_n (int): Number of top values to display for each column.
    """
    print(f"Shape: {df.shape}\n")

    df_info = df.dtypes.to_frame()
    df_info.columns = ["dtype"]
    df_info["%Nulls"] = df.isna().sum() * 100 / df.shape[0]
    df_info["#Unique"] = df.nunique()

    # Basic statistics
    df_info["Min"] = df.min(numeric_only=True)
    df_info["Mean"] = df.mean(numeric_only=True)
    df_info["Max"] = df.max(numeric_only=True)
    df_info["Std"] = df.std(numeric_only=True)

    # Top N values
    df_info["top_values"] = ""
    df_info["top_counts"] = ""
    df_info["top_ratios"] = ""
    for col in df_info.index:
        value_counts = df[col].value_counts().head(top_n)

        values = list(value_counts.index)
        counts = list(value_counts.values)
        ratios = list((value_counts.values / df.shape[0]).round(2))
        df_info.loc[col, "top_values"] = str(values)
        df_info.loc[col, "top_counts"] = str(counts)
        df_info.loc[col, "top_ratios"] = str(ratios)

    return df_info


def plot_categories(
    df, cat_features, ncols=4, fig_x=15, fig_y=18, top_n=10, color="tab:blue"
):
    nrows = (len(cat_features) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_x, fig_y))
    axes = axes.flatten()

    for i, col in enumerate(cat_features):
        order = df[col].value_counts(ascending=False).head(top_n).index

        sns.countplot(data=df, x=col, ax=axes[i], alpha=0.7, order=order, color=color)
        plt.xticks(rotation=90)

        # Calculate and display percentages on the bars
        total = len(df[col])
        for p in axes[i].patches:
            percentage = "{:.1f}%".format(100 * p.get_height() / total)
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            axes[i].annotate(percentage, (x, y), ha="center", va="bottom")

        axes[i].set(title=f"{col}\n", ylabel="", xlabel="")
        axes[i].tick_params(axis="both", which="both", length=0, rotation=90)
        axes[i].set_yticklabels("")
        sns.despine(left=True)

    # Remove any unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()


def plot_continuous(df, cont_features, ncols=4, fig_x=15, fig_y=35, color="tab:blue"):
    num_features = len(cont_features)
    nrows = (num_features + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_x, fig_y))
    axes = axes.flatten()

    for i, col in enumerate(cont_features):
        sns.kdeplot(df[col], ax=axes[i])
        axes[i].set(title=f"{col}\n", xlabel="")
        axes[i].set_yticklabels("")
        sns.despine()

    # Remove any unused subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(axes[j])

    plt.tight_layout()
