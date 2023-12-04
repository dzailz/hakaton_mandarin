import pandas as pd
from scipy.stats import anderson
from sklearn.preprocessing import StandardScaler

from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataPreparation:
    def __init__(self, df: DataFrame, to_drop_columns: list[str], bank: str):
        """
        :param df: dataframe to be cleaned for one bank
        :param to_drop_columns: columns to be dropped
        :param bank: expected values: 'BankA', 'BankB' etc.
        """
        self.df = df
        self.to_drop_columns = to_drop_columns
        self.bank = bank
        self.cleared_df = None
        self.condition_column = f'{self.bank}_decision'
        self.ohe_df = None
        self.df_no_outliers = None
        self.normalized_df = None

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

    def drop_columns(self) -> None:
        """
        Drop the columns
        """
        self.df = self.df.drop(columns=self.to_drop_columns)

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
            condition_mask = (df_no_outliers[self.condition_column] == condition_value)
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
        self.df['BirthYear'] = self.df['BirthDate'].dt.year
        self.df['Age'] = pd.Timestamp.now().year - self.df['BirthYear']
        self.df['JobStartYear'] = self.df['JobStartDate'].dt.year
        self.df['JobExperience'] = pd.Timestamp.now().year - self.df['JobStartYear']
        if is_drop:
            self.df.drop('JobStartDate', inplace=True, axis=1)
            self.df.drop('BirthDate', inplace=True, axis=1)

    def ohe_categorical_columns(self, df: DataFrame | None = None, columns: list[str] | None = None,
                                is_new: bool = True):

        """
        One hot encode categorical columns
        if columns is None, then the default columns will be used
        the default columns are: [
                'education',
                'employment status',
                'Gender',
                'Family status',
                'ChildCount',
                'Loan_term',
                'Goods_category',
                'Value',
                'SNILS',
                'Merch_code'
            ]
        """
        if not df:
            df = self.df
        if not columns:
            columns = [
                'education',
                'employment status',
                'Gender',
                'Family status',
                'ChildCount',
                'Loan_term',
                'Goods_category',
                'Value',
                'SNILS',
                'Merch_code'
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
            columns = ['MonthProfit', 'MonthExpense', 'Loan_amount']
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        if is_new:
            self.normalized_df = df
        else:
            self.df = df

    def save_df(self, path: str, index: bool = False):
        """
        Save the dataframe to csv
        """
        self.df.to_csv(path, index=index, )


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    df = pd.read_csv('../../datasets/bankAds.csv', parse_dates=['BirthDate', 'JobStartDate'], index_col=0, dtype={
        'education': 'category',
        'employment status': 'category',
        'Value': 'category',
        'MonthProfit': 'UInt64',
        'MonthExpense': 'UInt64',
        'Gender': 'category',
        'Family status': 'category',
        'ChildCount': 'category',
        'SNILS': 'category',
        'BankB_decision': 'category',
        'Merch_code': 'category',
        'Loan_amount': 'UInt64',
        'Loan_term': 'category',
        'Goods_category': 'category'
    })

    bankA = DataPreparation(df=df,
                            to_drop_columns=['BankA_decision', 'BankC_decision', 'BankD_decision', 'BankE_decision'],
                            bank='BankA')
    bankA.remove_outliers_all_numeric_with_condition(is_new=False)
    bankA.add_time_features()
    bankA.ohe_categorical_columns(is_new=False)
    bankA.normalize_numeric_features(is_new=False)

    bankA.save_df('../datasets/bankA_ohe_norm.csv')
