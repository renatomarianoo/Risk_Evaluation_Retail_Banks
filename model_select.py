'''
Created on 30/11/2023

@author: renato.mariano
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, X, y, top_n=10, color='tab:blue'):
    model.fit(X, y)

    feature_importance = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(6, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color=color, alpha=0.7)
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('')
    sns.despine()
    plt.show()

