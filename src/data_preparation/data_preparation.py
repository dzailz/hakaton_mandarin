import pandas as pd
import re
from scipy.stats import anderson
from sklearn.preprocessing import StandardScaler

from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataPreparation:
    def __init__(self, df: DataFrame, target_bank_col: str, to_drop_columns: list[str] | None = None, ):
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
        column = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column)
        column = re.sub('__([A-Z])', r'_\1', column)
        column = re.sub('([a-z0-9])([A-Z])', r'\1_\2', column)
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
        self.df.columns = self.df.columns.str.replace(' ', '_')

    def columns_to_datetime(self, columns: list[str]):
        """
        Set columns to datetime
        """
        for column in columns:
            self.df[column] = pd.to_datetime(self.df[column])

    def drop_values(self, column: str, value: str = 'error'):
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
        numeric_columns = self.df.select_dtypes(include='number').columns
        distribution = dict.fromkeys(numeric_columns, None)

        for column in numeric_columns:
            # Perform Anderson-Darling test for normality
            logger.info(f"Performing Anderson-Darling test for '{column}'")
            result = anderson(self.df[column])

            logger.info(f"Anderson-Darling test for '{column}':")

            if result.statistic <= result.critical_values[2]:
                logger.info(f"For column: {column} A-statistic: {result.statistic:.4f}")
                distribution[column] = {'is_normal': True, 'critical_values': result.critical_values.tolist(),
                                        'statistic': result.statistic}
            else:
                logger.info(f"For column: {column} A-statistic: {result.statistic:.4f}")
                distribution[column] = {'is_normal': False, 'critical_values': result.critical_values.tolist(),
                                        'statistic': result.statistic}

        return distribution

    def transform_to_categorical(self, columns: list[str]) -> None:
        """
        Transform the data to categorical
        """
        for column in columns:
            self.df[column] = self.df[column].astype('category')

    def drop_columns(self, columns: list[str]) -> None:
        """
        Drop the columns
        """
        self.df = self.df.drop(columns=columns)

    def remove_outliers_all_numeric_with_condition(self, df: DataFrame | None = None, condition_value: str = 'success',
                                                   multiplier: float = 1.5, is_new: bool = True):
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
        numeric_columns = df.select_dtypes(include='number').columns

        # Create a copy of the original DataFrame
        df_no_outliers = df.copy()

        # Iterate through numeric columns and remove outliers using IQR method with the specified condition
        for column in numeric_columns:
            # Calculate IQR only for rows where the condition is met
            condition_mask = (df_no_outliers[self.target_bank_col] == condition_value)
            q1 = df_no_outliers.loc[condition_mask, column].quantile(0.25)
            q3 = df_no_outliers.loc[condition_mask, column].quantile(0.75)
            iqr = q3 - q1

            # Define outlier bounds
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            # Remove outliers only for rows where the condition is met and the condition value is in outliers
            outliers_mask = condition_mask & (df_no_outliers[column] < lower_bound) | (
                    df_no_outliers[column] > upper_bound)
            df_no_outliers = df_no_outliers[~outliers_mask]
        if is_new:
            self.df_no_outliers = df_no_outliers
        else:
            self.df = df_no_outliers

    def add_time_features(self, is_drop: bool = True):
        """
        is_drop: if True, then the time columns will be dropped
        Add age, job experience, job start year, and birth year features
        """
        self.df['birth_year'] = self.df['birth_date'].dt.year
        self.df['age'] = pd.Timestamp.now().year - self.df['birth_year']
        self.df['job_start_year'] = self.df['job_start_date'].dt.year
        self.df['job_experience'] = pd.Timestamp.now().year - self.df['job_start_year']
        if is_drop:
            self.df.drop('job_start_date', inplace=True, axis=1)
            self.df.drop('birth_date', inplace=True, axis=1)

    def ohe_categorical_columns(self, df: DataFrame | None = None, columns: list[str] | None = None,
                                is_new: bool = True):

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
        if not df:
            df = self.df
        if not columns:
            columns = [
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
        if is_new:
            self.ohe_df = pd.get_dummies(df, columns=columns)
        else:
            self.df = pd.get_dummies(df, columns=columns)

    def normalize_numeric_features(self, df: DataFrame | None = None, columns: list[str] | None = None,
                                   is_new: bool = True):
        """
        Normalize numeric columns
        """
        if df is None:
            df = self.df
        if columns is None:
            columns = ['month_profit', 'month_expense', 'loan_amount']
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        if is_new:
            self.normalized_df = df
        else:
            self.df = df

    def save_df(self, path: str, index: bool = False, compression='gzip'):
        """
        Save the data frame to parquet
        """
        self.df.to_parquet(path, index=index, compression=compression)


if __name__ == '__main__':
    from os import path
    banks = [f'bank_{bank}_decision' for bank in ['a', 'b', 'c', 'd', 'e']]

    for i in ['a', 'b', 'c', 'd', 'e']:
        banks_to_drop = banks.copy()
        banks_to_drop.remove(f'bank_{i}_decision')

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        df_path = path.join(path.dirname(__file__), '../../data/datasets/SF_Mandarin_dataset_ver3.csv')

        df = pd.read_csv(df_path, sep=';', index_col=0)

        bank_a = DataPreparation(
            df=df,
            to_drop_columns=banks_to_drop,
            target_bank_col=[i for i in banks if i not in banks_to_drop].pop()
        )
        bank_a.drop_na()
        bank_a.index_as_int()
        bank_a.drop_duplicates()
        bank_a.convert_from_camel_case_to_snake_case()
        bank_a.columns_to_datetime(columns=['job_start_date', 'birth_date'])

        bank_a.columns_to_type(columns=['month_profit', 'month_expense', 'loan_amount'], dtype='UInt64')

        bank_a.columns_to_type(columns=['snils', 'gender', 'merch_code', 'child_count', 'loan_term'], dtype='UInt8')
        bank_a.columns_to_type(
            columns=['family_status', 'goods_category', 'position', 'employment_status', 'education', 'snils',
                     'gender'], dtype='category')

        bank_a.remove_outliers_all_numeric_with_condition(is_new=False)
        bank_a.add_time_features()
        bank_a.ohe_categorical_columns(is_new=False)
        bank_a.normalize_numeric_features(is_new=False)
        save_path = path.join(path.dirname(__file__), f'../../data/datasets/bank_{i}_ohe_norm.parquet')
        bank_a.save_df(save_path)
