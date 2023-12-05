"""
Created on 30/11/2023

@author: renato.mariano
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from feature_engine.encoding import RareLabelEncoder
import feat_eng


def create_preprocess_pipeline(num_feats, binary_feats, highcard_feats):
    num_transformer = Pipeline(
        steps=[
            ("robust_scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )

    binary_transformer = Pipeline(
        steps=[
            (
                "ordinal_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    highcard_transformer = Pipeline(
        steps=[
            (
                "rare_encoder",
                RareLabelEncoder(tol=0.02, n_categories=1, missing_values="ignore"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocess_pipe = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_feats),
            ("bin", binary_transformer, binary_feats),
            ("cat", highcard_transformer, highcard_feats),
        ]
    )

    preprocess_pipe.set_output(transform="pandas")

    return preprocess_pipe
