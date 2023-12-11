"""
Created on 26/11/2023

@author: renato.mariano
"""

import pandas as pd
import numpy as np
from rapidfuzz import process
from sklearn.base import BaseEstimator, TransformerMixin

################# First Employed set of Feature Engineering - Application Dataframe ###############

class ZeroToNullTransformer(BaseEstimator, TransformerMixin):
    """
    Replace values equal to zero with np.nan for specified columns.

    Parameters:
        columns (list or str): The column or list of columns to transform.
    """

    def __init__(self, columns):
        self.columns = columns if isinstance(columns, list) else [columns]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[self.columns] = X[self.columns].replace(0, np.nan)
        return X


class MultiplyByNeg1(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            if col in X.columns:
                X[col] *= -1
        return X


class HandleDaysEmployedAnomaly(BaseEstimator, TransformerMixin):
    """
    Transformer for handling anomalous values in the 'DAYS_EMPLOYED' column.

    This transformer performs the following tasks:
    1. Creates a new binary column 'DAYS_EMPLOYED_anom' to identify anomalous values.
    2. Replaces anomalous values with NaN in the 'DAYS_EMPLOYED' column.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["DAYS_EMPLOYED_anom"] = X["DAYS_EMPLOYED"] < 0
        X["DAYS_EMPLOYED_anom"] = X["DAYS_EMPLOYED_anom"].map({True: 1, False: 0})
        X.loc[X["DAYS_EMPLOYED"] < 0, "DAYS_EMPLOYED"] = np.nan

        return X


class ApplyMapToOrganization:
    """
    Groups together elements with similar names based on the provided mapping.
    Orginal column is dropped.
    """

    def __init__(self, column, mapping=None, similarity_threshold=60):
        self.column = column
        self.mapping = mapping or {
            "Business Entity": "Business Entity Type",
            "xna": "XNA",
            "Self-employed": "Self-employed",
            "Other": "Other",
            "Medicine": "Medicine",
            "Government": "Government",
            "School": "School",
            "Trade": "Trade: type",
            "Kindergarten": "Kindergarten",
            "Construction": "Construction",
            "Transport": "Transport: type",
            "Industry": "Industry: type",
            "Security": "Security",
            "Housing": "Housing",
            "Military": "Military",
            "Bank": "Bank",
        }
        self.similarity_threshold = similarity_threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        new_column = (
            X[self.column]
            .str.lower()
            .str.strip()
            .str.replace("-", " ")
            .str.replace("_", " ")
        )
        new_column = new_column.apply(
            lambda x: process.extractOne(x, self.mapping.keys())
        )
        new_column = new_column.apply(
            lambda x: self.mapping[x[0]]
            if x and x[1] >= self.similarity_threshold
            else "Other"
        )

        X = X.drop(self.column, axis=1)
        X["ORGANIZATION_TYPE_Grouped"] = new_column
        return X


################# Second Employed set of Feature Engineering - Application Dataframe ###############


class ExternalSourcesTransformer(BaseEstimator, TransformerMixin):
    """Calculate new parameters based on the external sources"""

    def __init__(self, sources=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]):
        self.sources = sources

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["EXT_SOURCES_prod"] = X[self.sources].product(axis=1)
        X["EXT_SOURCES_sum"] = X[self.sources].sum(axis=1)
        X["EXT_SOURCES_mean"] = X[self.sources].mean(axis=1)

        X["EXT_SOURCE_1_2_mean"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2"]].mean(axis=1)
        X["EXT_SOURCE_2_3_mean"] = X[["EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        X["EXT_SOURCE_1_3_mean"] = X[["EXT_SOURCE_1", "EXT_SOURCE_3"]].mean(axis=1)

        return X


class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """Calculate financial ratios based on specified columns."""

    def __init__(self, base_name="amt"):
        self.base_name = base_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        ratios = {
            "credit_income_ratio": "AMT_CREDIT / AMT_INCOME_TOTAL",
            "annuity_income_ratio": "AMT_ANNUITY / AMT_INCOME_TOTAL",
            "goods_price_income_ratio": "AMT_GOODS_PRICE / AMT_INCOME_TOTAL",
            "credit_annuity_ratio": "AMT_CREDIT / AMT_ANNUITY",
            "goods_price_annuity_ratio": "AMT_GOODS_PRICE / AMT_ANNUITY",
            "goods_price_credit_ratio": "AMT_GOODS_PRICE / AMT_CREDIT",
            "income_children_ratio": "AMT_INCOME_TOTAL / (CNT_CHILDREN + 1)",
            "credit_children_ratio": "AMT_CREDIT / (CNT_CHILDREN + 1)",
            "annuity_children_ratio": "AMT_ANNUITY / (CNT_CHILDREN + 1)",
            "goods_price_children_ratio": "AMT_GOODS_PRICE / (CNT_CHILDREN + 1)",
            "income_family_members_ratio": "AMT_INCOME_TOTAL / CNT_FAM_MEMBERS",
            "credit_family_members_ratio": "AMT_CREDIT / CNT_FAM_MEMBERS",
            "annuity_family_members_ratio": "AMT_ANNUITY / CNT_FAM_MEMBERS",
            "goods_price_family_members_ratio": "AMT_GOODS_PRICE / CNT_FAM_MEMBERS",
        }

        for ratio_name, formula in ratios.items():
            column_name = f"{self.base_name}_{ratio_name}"
            X[column_name] = X.eval(formula)

        return X


class AgeAndEmploymentTransformer(BaseEstimator, TransformerMixin):
    """Calculate age, age bins, years of work and percent of worked days."""

    def __init__(self, birth="DAYS_BIRTH", employed="DAYS_EMPLOYED"):
        self.birth = birth
        self.employed = employed

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X["age"] = X[self.birth] // 365
        X["years_worked"] = X[self.employed] // 365
        X["age_binned"] = pd.cut(X["age"], bins=np.linspace(20, 70, num=11))
        X["days_employed_percent"] = X[self.employed] / X[self.birth]

        return X


################# Third Employed set of Feature Engineering - Previous Appl and Installments ###############

class PreviousApplicationTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for processing previous application data.

    This transformer performs the following tasks:
    1. Counts the number of previous loans for each SK_ID_CURR.
    2. Creates a pivot table for contract status counts for each SK_ID_CURR.
    3. Aggregates continous features (float64), calculating median, max, and min values for each SK_ID_CURR.

    Returns:
    DataFrame: Transformed DataFrame with the new features - 1 row per client id.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        previous_loan_counts = self.calculate_previous_loan_counts(X)
        pivot_table = self.create_contract_status_pivot_table(X)
        numerical_agg = self.aggregate_numerical_features(X)

        X_transformed = previous_loan_counts.merge(
            pivot_table, on="SK_ID_CURR", how="left"
        )
        X_transformed = X_transformed.merge(
            numerical_agg, on="SK_ID_CURR", how="left"
        )

        return X_transformed

    def calculate_previous_loan_counts(self, X):
        previous_loan_counts = (
            X.groupby("SK_ID_CURR").size().reset_index(name="PREV_previous_loan_counts")
        )
        return previous_loan_counts

    def create_contract_status_pivot_table(self, X):
        pivot_table = X[["SK_ID_CURR", "NAME_CONTRACT_STATUS"]].pivot_table(
            index="SK_ID_CURR", columns="NAME_CONTRACT_STATUS", aggfunc="size"
        )
        pivot_table.columns = [
            f"PREV_name_contract_{status.lower().replace(' ', '_')}"
            for status in pivot_table.columns
        ]
        pivot_table.reset_index(inplace=True)
        return pivot_table
    
    def aggregate_numerical_features(self, X):
        numerical_columns = X.select_dtypes(include='float64').columns
        aggregation_functions = ["median", "mean", "max", "min"]

        agg_dict = {col: aggregation_functions for col in numerical_columns}
        numerical_agg = (
            X.groupby("SK_ID_CURR")
            .agg(agg_dict)
            .reset_index()
        )

        numerical_agg.columns = [
            f"PREV_{col.lower()}_{agg_func}" if col != "SK_ID_CURR" else col
            for col, agg_func in numerical_agg.columns
        ]
        
        return numerical_agg
    

class InstallmentsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for processing installments data.

    This transformer performs the following tasks:
    1. Calculates the ratio of paid value by installment value (aggregated by median, mean, max, min).
    2. Flags if the client ever paid less (aggregated by mode and existence flag).
    3. Calculates delayed days of payment (aggregated by median, mean, max, min).
    4. Flags if a client has delayed a payment (aggregated by mode and existence flag).
    5. Calculates delayed days of payment in the 1st year of loan (aggregated by median, mean, max, min).

    Returns:
    DataFrame: Transformed DataFrame with the new features - 1 row per client id.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ratio_aggregated = self.calculate_ratio_aggregations(X)
        X_paid_less_aggregated = self.flag_paid_less(X)
        X_delayed_days = self.calculate_delayed_days_aggregations(X)
        X_delayed_payment = self.flag_delayed_payment(X)
        X_delayed_days_365 = self.calculate_delayed_days_aggregations_365(X)

        X_aggregated = (
            X_ratio_aggregated.merge(
                X_paid_less_aggregated, on="SK_ID_CURR", how="left"
            )
            .merge(X_delayed_days, on="SK_ID_CURR", how="left")
            .merge(X_delayed_payment, on="SK_ID_CURR", how="left")
            .merge(X_delayed_days_365, on="SK_ID_CURR", how="left")
        )

        return X_aggregated

    def calculate_ratio_aggregations(self, X):
        X["amt_payment_installment_ratio"] = X["AMT_PAYMENT"] / X["AMT_INSTALMENT"]
        X["amt_payment_installment_ratio"].replace(
            [np.inf, -np.inf], np.nan, inplace=True
        )
        aggregations = ["median", "mean", "max", "min"]
        X_ratio_aggregated = (
            X.groupby("SK_ID_CURR")["amt_payment_installment_ratio"]
            .agg(aggregations)
            .reset_index()
        )
        X_ratio_aggregated = X_ratio_aggregated.rename(
            columns=lambda col: f"INST_amt_payment_installment_ratio_{col}"
            if col != "SK_ID_CURR"
            else col
        )
        return X_ratio_aggregated

    def flag_paid_less(self, X):
        X["paid_less_flag"] = (X["amt_payment_installment_ratio"] < 0.95).astype(int)
        X_paid_less_aggregated = (
            X.groupby("SK_ID_CURR")["paid_less_flag"]
            .agg([pd.Series.mode, "any"])
            .reset_index()
        )

        X_paid_less_aggregated["any"] = X_paid_less_aggregated["any"].map(
            {True: 1, False: 0}
        )
        X_paid_less_aggregated[
            "mode"
        ] = X_paid_less_aggregated[  # pd.Series.mode returned an array in
            "mode"
        ].apply(
            lambda x: 1 if isinstance(x, np.ndarray) else x
        )
        X_paid_less_aggregated = X_paid_less_aggregated.rename(
            columns=lambda col: f"INST_paid_less_flag_{col}"
            if col != "SK_ID_CURR"
            else col
        )
        return X_paid_less_aggregated

    def calculate_delayed_days_aggregations(self, X):
        X["delayed_days"] = X["DAYS_INSTALMENT"] - X["DAYS_ENTRY_PAYMENT"]
        aggregations = ["median", "mean", "max", "min"]
        X_delayed_days = (
            X.groupby("SK_ID_CURR")["delayed_days"].agg(aggregations).reset_index()
        )
        X_delayed_days = X_delayed_days.rename(
            columns=lambda col: f"INST_delayed_days_{col}"
            if col != "SK_ID_CURR"
            else col
        )
        return X_delayed_days

    def flag_delayed_payment(self, X):
        X["delayed_payment_flag"] = (X["delayed_days"] > 0).astype(int)
        X_delayed_payment = (
            X.groupby("SK_ID_CURR")["delayed_payment_flag"]
            .agg([pd.Series.mode, "any"])
            .reset_index()
        )

        X_delayed_payment["any"] = X_delayed_payment["any"].map({True: 1, False: 0})
        X_delayed_payment["mode"] = X_delayed_payment["mode"].apply(
            lambda x: 1 if isinstance(x, np.ndarray) else x
        )
        X_delayed_payment = X_delayed_payment.rename(
            columns=lambda col: f"INST_delayed_payment_flag_{col}"
            if col != "SK_ID_CURR"
            else col
        )
        return X_delayed_payment

    def calculate_delayed_days_aggregations_365(self, X):
        X_temp = X[X["DAYS_ENTRY_PAYMENT"] >= -365]
        X_temp["delayed_days_365"] = (
            X_temp["DAYS_INSTALMENT"] - X_temp["DAYS_ENTRY_PAYMENT"]
        )
        aggregations = ["median", "mean", "max", "min"]
        X_delayed_days_365 = (
            X_temp.groupby("SK_ID_CURR")["delayed_days_365"]
            .agg(aggregations)
            .reset_index()
        )
        X_delayed_days_365 = X_delayed_days_365.rename(
            columns=lambda col: f"INST_delayed_days_365_{col}"
            if col != "SK_ID_CURR"
            else col
        )
        return X_delayed_days_365


################# Fourth Employed set of Feature Engineering - Bureau Data ###############

class BureauTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms the bureau dataframe by calculating:
     1- aggregations for continous features,
     2- application counts,
     3- creating a credit active pivot table,
     4- financial ratios (median, max, and min values for each SK_ID_CURR).

    Returns:
    Dataframe: Transformed bureau df with the new features + old financial features - 1 id per client
    """

    def __init__(self, base_name="BU"):
        self.base_name = base_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numerical_agg = self.aggregate_numerical_features(X)
        bureau_loan_counts = self.calculate_loan_counts(X)
        pivot_table = self.create_credit_active_pivot_table(X)
        financial_ratios = self.create_financial_ratios(X)

        X_transformed = numerical_agg.merge(
            bureau_loan_counts, on="SK_ID_CURR", how="left"
        ).merge(
            pivot_table, on="SK_ID_CURR", how="left"
        ).merge(
            financial_ratios, on="SK_ID_CURR", how="left"
        )

        return X_transformed
    
    def aggregate_numerical_features(self, X):
        numerical_columns = X.select_dtypes(include='float64').columns
        aggregation_functions = ["median", "mean", "max", "min"]

        agg_dict = {col: aggregation_functions for col in numerical_columns}
        numerical_agg = (
            X.groupby("SK_ID_CURR")
            .agg(agg_dict)
            .reset_index()
        )

        numerical_agg.columns = [
            f"BU_{col.lower()}_{agg_func}" if col != "SK_ID_CURR" else col
            for col, agg_func in numerical_agg.columns
        ]
        
        return numerical_agg

    def calculate_loan_counts(self, X):
        bureau_loan_counts = (
            X.groupby("SK_ID_CURR").size().reset_index(name="BU_application_counts")
        )
        return bureau_loan_counts

    def create_credit_active_pivot_table(self, X):
        pivot_table = X[["SK_ID_CURR", "CREDIT_ACTIVE"]].pivot_table(
            index="SK_ID_CURR", columns="CREDIT_ACTIVE", aggfunc="size"
        )
        pivot_table.columns = [
            f"BU_credit_active_{status.lower().replace(' ', '_')}"
            for status in pivot_table.columns
        ]
        pivot_table.reset_index(inplace=True)
        return pivot_table

    def create_financial_ratios(self, X):
        ratios = {
            "max_overdue_prolong_ratio": "AMT_CREDIT_MAX_OVERDUE / CNT_CREDIT_PROLONG",
            "max_overdue_credit_sum_ratio": "AMT_CREDIT_MAX_OVERDUE / AMT_CREDIT_SUM",
            "max_overdue_debt_ratio": "AMT_CREDIT_MAX_OVERDUE / AMT_CREDIT_SUM_DEBT",
            "max_overdue_overdue_ratio": "AMT_CREDIT_MAX_OVERDUE / AMT_CREDIT_SUM_OVERDUE",
            "annuity_prolong_ratio": "AMT_ANNUITY / CNT_CREDIT_PROLONG",
            "annuity_credit_sum_ratio": "AMT_ANNUITY / AMT_CREDIT_SUM",
            "annuity_update_ratio": "AMT_ANNUITY / DAYS_CREDIT_UPDATE",
            "annuity_overdue_ratio": "AMT_ANNUITY / AMT_CREDIT_SUM_OVERDUE",
            "annuity_debt_limit_ratio": "AMT_ANNUITY / AMT_CREDIT_SUM_DEBT",
            "prolong_ratio": "CNT_CREDIT_PROLONG / AMT_CREDIT_SUM",
            "prolong_credit_sum_ratio": "CNT_CREDIT_PROLONG / AMT_CREDIT_SUM",
            "debt_prolong_ratio": "AMT_CREDIT_SUM_DEBT / CNT_CREDIT_PROLONG",
            "limit_prolong_ratio": "AMT_CREDIT_SUM_LIMIT / CNT_CREDIT_PROLONG",
            "prolong_overdue_ratio": "CNT_CREDIT_PROLONG / AMT_CREDIT_SUM_OVERDUE",
            "limit_debt_ratio": "AMT_CREDIT_SUM_LIMIT / AMT_CREDIT_SUM_DEBT",
            "debt_credit_sum_ratio": "AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM",
            "debt_limit_overdue_ratio": "AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM_OVERDUE",
            "overdue_ratio": "AMT_CREDIT_SUM_OVERDUE / AMT_CREDIT_SUM",
            "credit_sum_limit_ratio": "AMT_CREDIT_SUM_LIMIT / AMT_CREDIT_SUM",
            "credit_sum_overdue_ratio": "AMT_CREDIT_SUM / AMT_CREDIT_SUM_OVERDUE",
            "limit_overdue_ratio": "AMT_CREDIT_SUM_LIMIT / AMT_CREDIT_SUM_OVERDUE",
        }

        columns_for_aggregation = [
            "AMT_CREDIT_MAX_OVERDUE",
            "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_LIMIT",
            "AMT_CREDIT_SUM_OVERDUE",
            "DAYS_CREDIT_UPDATE",
            "AMT_ANNUITY",
        ]
        columns_for_aggregation += [f"{self.base_name}_{col}" for col in ratios.keys()]

        for ratio_name, formula in ratios.items():
            column_name = f"{self.base_name}_{ratio_name}"
            X[column_name] = X.eval(formula)

        financial_ratios = X.groupby("SK_ID_CURR")[columns_for_aggregation].agg(
            ["median", "mean", "max", "min"]
        )
        financial_ratios.columns = [
            f"{col[0]}_{col[1]}" for col in financial_ratios.columns
        ]
        financial_ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
        return financial_ratios
    