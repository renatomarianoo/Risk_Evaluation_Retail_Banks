'''
Created on 30/11/2023

@author: renato.mariano
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve


def plot_feature_importance(model, X, y, ax, top_n=10, color='tab:blue', model_name=""):
    model.fit(X, y)

    feature_importance = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color=color, alpha=0.7, ax=ax)
    ax.set(title = f'Top {top_n} Feature Importance\n{model_name}', xlabel='Importance', ylabel='')
    sns.despine()


def plot_pr_curve(model, X, y, model_name, ax):
    '''Function to perform plot precision-recall curve'''
    y_probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_probs)
    fscore = (2 * precision * recall) / (precision + recall)
    ix = np.nanargmax(fscore)

    ax.plot([0,1], [0,0], linestyle='--', label='No Skill', color='tab:blue')
    ax.plot(recall, precision, label=f'{model_name}', color='orange', linewidth=0.9)

    ax.set(title=f"Precision-Recall Curve\n{model_name}", ylabel="Precision", xlabel="Recall")  
    ax.legend()
    text = f'Best Threshold={thresholds[ix]:.3f}\nF1-Score={fscore[ix]:.3f}\nRecall={recall[ix]:.3f}'
    ax.text(1, 0.7, text, fontsize=8, ha='right', va='bottom', color='black')
    ax.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')


def find_threshold_for_recall(model, X, y, target_recall):
    '''Function to recieve a target Recall and return the PR threshold'''
    y_probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, y_probs)
    ix = (np.abs(recall - target_recall)).argmin()
    chosen_threshold = thresholds[ix]
    
    print(f"Target Recall {target_recall}\nThreshold: {chosen_threshold:.3f}; Precision: {precision[ix]:.3f}")
    return chosen_threshold


def apply_threshold(model, X, threshold):
    y_probs = model.predict_proba(X)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    return y_pred
