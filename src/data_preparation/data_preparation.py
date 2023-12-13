import logging
import pickle
import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from scipy.stats import anderson
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from settings import MODELS_FOLDER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataPreparation:
    def __init__(
        self,
        df: DataFrame,
        target_bank_col: str | None = None,
        to_drop_columns: list[str] | None = None,
        scaler: StandardScaler | None = None,
        ohe_model: OneHotEncoder | None = None,
        to_ohe_columns: list[str] | None = None,
    ):
        """
        :param df: dataframe to be cleaned for one bank
        :param to_drop_columns: columns to be dropped
        :param target_bank_col: expected values: 'BankA', 'BankB' etc.
        """
        self.df = df
        self.to_drop_columns = to_drop_columns
        self.target_bank_col = target_bank_col
        self.cleared_df = None
        self.ohe_df = None
        self.df_no_outliers = None
        self.normalized_df = None
        self.scaler = scaler if scaler else StandardScaler()
        self.ohe_model = ohe_model or OneHotEncoder(sparse_output=False, drop="first", handle_unknown="error")
        self.to_ohe_columns = to_ohe_columns or [
            "education",
            "employment_status",
            "gender",
            "family_status",
            "child_count",
            "loan_term",
            "goods_category",
            "value",
            "snils",
            "merch_code",
        ]

    def drop_na(self):
        """
        Drop na values
        """
        self.df.dropna(inplace=True)

    def index_as_int(self):
        """
        Set index as int
        """
        self.df.index = self.df.index.astype(int)

    def drop_duplicates(self):
        """
        Drop duplicates
        """
        self.df.drop_duplicates(inplace=True)

    def columns_to_lower(self):
        """
        Set columns to lower
        """
        self.df.columns = self.df.columns.str.lower()

    @staticmethod
    def to_snake_case(column):
        column = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", column)
        column = re.sub("__([A-Z])", r"_\1", column)
        column = re.sub("([a-z0-9])([A-Z])", r"\1_\2", column)
        return column.lower()

    def convert_from_camel_case_to_snake_case(self):
        """
        Convert from camel case to snake case
        """
        self.columns_to_snake_case()
        self.df.columns = self.df.columns.to_series().apply(self.to_snake_case)
        self.columns_to_snake_case()

    def columns_to_type(self, columns: list[str], dtype: str):
        """
        Set columns to type
        """
        for column in columns:
            self.df[column] = self.df[column].astype(dtype)

    def columns_to_snake_case(self):
        """
        Set columns to snake case
        """
        self.df.columns = self.df.columns.str.replace(" ", "_")

    def columns_to_datetime(self, columns: list[str]):
        """
        Set columns to datetime
        """
        for column in columns:
            self.df[column] = pd.to_datetime(self.df[column])

    def drop_values(self, column: str, value: str = "error"):
        """
        This method replaces a specified value in a given column with NA (Not Available), and then drops these NA values.
        This is typically used to handle error values in the data that are not useful for analysis or modeling.

        Parameters:
        column (str): The name of the column in the dataframe where the specified value should be replaced.
        value (str, default='error'): The value to be replaced with NA in the specified column.

        Returns:
        None: The method operates on the dataframe in-place and does not return anything.
        """
        self.df[column] = self.df[column].replace(value, pd.NA)
        self.df.dropna(inplace=True)

    def is_normality_distributed(self) -> dict:
        """
        Check the normality of each numeric column in a DataFrame using the Anderson-Darling test.
        Returns:
        - dict: Dictionary of distribution for each numeric column
        """
        # Select numeric columns
        numeric_columns = self.df.select_dtypes(include="number").columns
        distribution = dict.fromkeys(numeric_columns, None)

        for column in numeric_columns:
            # Perform Anderson-Darling test for normality
            logger.info(f"Performing Anderson-Darling test for '{column}'")
            result = anderson(self.df[column])

            logger.info(f"Anderson-Darling test for '{column}':")

            if result.statistic <= result.critical_values[2]:
                logger.info(f"For column: {column} A-statistic: {result.statistic:.4f}")
                distribution[column] = {"is_normal": True, "critical_values": result.critical_values.tolist(), "statistic": result.statistic}
            else:
                logger.info(f"For column: {column} A-statistic: {result.statistic:.4f}")
                distribution[column] = {"is_normal": False, "critical_values": result.critical_values.tolist(), "statistic": result.statistic}

        return distribution

    def transform_to_categorical(self, columns: list[str]) -> None:
        """
        Transform the data to categorical
        """
        for column in columns:
            self.df[column] = self.df[column].astype("category")

    def drop_columns(self, columns: list[str]) -> None:
        """
        Drop the columns (inplace)
        """
        self.df.drop(columns=columns, axis=1, inplace=True)

    def remove_outliers_all_numeric_with_condition(self, df: DataFrame | None = None, condition_value: str = "success", multiplier: float = 1.5, is_new: bool = True):
        """
        Remove outliers from all numeric columns in a DataFrame using the IQR method, based on a specific condition.

        Parameters:
        - condition_value: Value in the condition_column for which outliers will be removed (default is 'success')
        - multiplier: Multiplier to control the range of the IQR (default is 1.5)

        Returns:
        - DataFrame with outliers removed based on the specified condition
        """
        # Select numeric columns
        if df is None:
            df = self.df
        numeric_columns = df.select_dtypes(include="number").columns

        # Create a copy of the original DataFrame
        df_no_outliers = df.copy()

        # Iterate through numeric columns and remove outliers using IQR method with the specified condition
        for column in numeric_columns:
            # Calculate IQR only for rows where the condition is met
            condition_mask = df_no_outliers[self.target_bank_col] == condition_value
            q1 = df_no_outliers.loc[condition_mask, column].quantile(0.25)
            q3 = df_no_outliers.loc[condition_mask, column].quantile(0.75)
            iqr = q3 - q1
            # Define outlier bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            # Remove outliers only for rows where the condition is met and the condition value is in outliers
            outliers_mask = condition_mask & (df_no_outliers[column] < lower_bound) | (df_no_outliers[column] > upper_bound)
            df_no_outliers = df_no_outliers[~outliers_mask]
        if is_new:
            self.df_no_outliers = df_no_outliers
        else:
            self.df = df_no_outliers

    def add_time_features(self, is_drop: bool = True, df: DataFrame | None = None):
        """
        is_drop: if True, then the time columns will be dropped
        Add age, job experience, job start year, and birth year features
        """
        if df is None:
            df = self.df
        self.columns_to_datetime(["job_start_date", "birth_date"])
        df["birth_year"] = df["birth_date"].dt.year
        df["age"] = pd.Timestamp.now().year - df["birth_year"]
        df["job_start_year"] = df["job_start_date"].dt.year
        df["job_experience"] = pd.Timestamp.now().year - df["job_start_year"]
        if is_drop:
            df.drop("job_start_date", inplace=True, axis=1)
            df.drop("birth_date", inplace=True, axis=1)

    def fit_one_hot_encoder(self, df: DataFrame = None, columns: list[str] = None, is_save_model: bool = True, ohe_filename: str = "ohe_model.pkl"):
        """
        Fit the one-hot encoder on the specified columns and save the model.
        """
        if df is None:
            df = self.df
        columns = columns or self.to_ohe_columns

        # Fit the one-hot encoder on the specified columns
        self.ohe_model.fit(df[columns])

        # Save the one-hot encoding model
        if is_save_model:
            with open(Path(MODELS_FOLDER, ohe_filename), "wb") as model_file:
                pickle.dump(self.ohe_model, model_file)

    def transform_one_hot_encoder(self, df: DataFrame = None, columns: list[str] = None, is_new: bool = True):
        """
        Transform the specified DataFrame using the pre-fitted one-hot encoder.
        """
        if df is None:
            df = self.df
        columns = columns or self.to_ohe_columns

        # Transform the specified columns
        ohe_result = self.ohe_model.transform(df[columns])

        # Create a DataFrame with the one-hot-encoded columns
        ohe_df = pd.DataFrame(ohe_result, columns=self.ohe_model.get_feature_names_out(columns))

        # Drop the original categorical columns
        df = df.drop(columns, axis=1, errors="ignore")

        # Join the new DataFrame with the original DataFrame based on the index
        df = df.join(ohe_df)

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # If is_new is True, assign the result to self.ohe_df
        if is_new:
            self.ohe_df = df
        else:
            self.df = df

    def ohe_categorical_columns(self, df: DataFrame = None, columns: list[str] = None, is_new: bool = True, is_save_model: bool = True, ohe_filename: str = "ohe_model.pkl"):
        """
        One hot encode categorical columns
        if columns is None, then the default columns will be used
        the default columns are: [
                'education',
                'employment_status',
                'gender',
                'family_status',
                'child_count',
                'loan_term',
                'goods_category',
                'value',
                'snils',
                'merch_code'
            ]
        """
        if df is None:
            df = self.df
        columns = columns or self.to_ohe_columns

        # Fit the one-hot encoder and save the model
        self.fit_one_hot_encoder(df, columns, is_save_model, ohe_filename)

        # Transform the DataFrame using the pre-fitted one-hot encoder
        self.transform_one_hot_encoder(df, columns, is_new)

    def fit_scaler(
        self, df: DataFrame | None = None, columns: list[str] | None = None, scaler_save_name: str | None = "scaler.pkl", scaler_path: str | None = None, is_save: bool = True
    ):
        """
        Fit the scaler on the specified columns.
        Save the scaler for later use.
        """
        scaler_path = scaler_path or Path(MODELS_FOLDER, scaler_save_name)
        if df is None:
            df = self.df
        if columns is None:
            columns = ["month_profit", "month_expense", "loan_amount"]
        self.scaler.fit(df[columns])
        if is_save:
            pickle.dump(self.scaler, open(scaler_path, "wb"))

    def normalize_numeric_features(self, df: DataFrame | None = None, columns: list[str] | None = None, is_new: bool = True, is_save: bool = True):
        """
        Transform the specified columns using the previously fitted scaler.
        """
        if df is None:
            df = self.df
        columns = columns or ["month_profit", "month_expense", "loan_amount"]
        self.fit_scaler(columns=columns, is_save=is_save)
        df[columns] = self.scaler.transform(df[columns])
        if is_new:
            self.normalized_df = df
        else:
            self.df = df

    def save_df(self, path: str, index: bool = False, compression="gzip"):
        """
        Save the data frame to parquet
        """
        self.df.to_parquet(path, index=index, compression=compression)
