from pandas import DataFrame

from src.data_preparation.data_preparation import DataPreparation


class DataPreprocessing(DataPreparation):
    def __init__(
            self,
            df: DataFrame,
            target_bank_col: str,
            banks_cols: list[str],
            banks_cols_to_drop: list[str],
            categorical_columns: list[str],
            numeric_categorical_columns: list[str],
            money_columns: list[str],
            to_drop_columns: list[str] | None = None,
    ):
        super().__init__(df=df, to_drop_columns=to_drop_columns, target_bank_col=target_bank_col)
        self.banks_cols = banks_cols
        self.banks_cols_to_drop = banks_cols_to_drop
        self.categorical_columns = categorical_columns
        self.numeric_cartegorical_columns = numeric_categorical_columns
        self.money_columns = money_columns
        self.all_categorical_columns = self.categorical_columns.copy()
        self.all_categorical_columns.extend(self.numeric_cartegorical_columns)

    def drop_errors(self, columns=list[str], value='error'):
        for column in columns:
            self.drop_values(column=column, value=value)

    def preparation_full_df(self, save_path: str | None = None):
        """
        This method is responsible for preparing the full dataframe for further processing.
        It performs several operations such as dropping NA values, converting indices to integers,
        dropping duplicates, converting column names from camel case to snake case, and converting
        certain columns to specific data types.

        Parameters:
        save_path (str, optional): The path where the prepared dataframe should be saved. If not provided,
                                   the method will return the prepared dataframe.

        Returns:
        DataFrame: The prepared dataframe if save_path is not provided.
        """

        # Drop NA values from the dataframe
        self.drop_na()

        # Convert the index of the dataframe to integer
        self.index_as_int()

        # Drop duplicate rows from the dataframe
        self.drop_duplicates()

        self.drop_errors()
        # Convert column names from camel case to snake case
        self.convert_from_camel_case_to_snake_case()

        # Convert 'job_start_date' and 'birth_date' columns to datetime
        self.columns_to_datetime(columns=['job_start_date', 'birth_date'])

        # Convert 'money_columns' to 'UInt64' data type
        self.columns_to_type(columns=self.money_columns, dtype='UInt64')

        # Convert 'numeric_cartegorical_columns' to 'UInt8' data type
        self.columns_to_type(columns=self.numeric_cartegorical_columns, dtype='UInt8')

        # Convert 'all_categorical_columns' to 'category' data type
        self.columns_to_type(columns=self.all_categorical_columns, dtype='category')

        # If save_path is provided, save the dataframe to the specified path
        if save_path:
            self.save_df(path=save_path)
        else:
            # If save_path is not provided, return the prepared dataframe
            return self.df

    def final_preparation(
            self,
            df: DataFrame | None = None,
            save_path: str | None = None,
            banks_cols_to_drop: list[str] | None = None,
            condition_value: str = 'success',
            multiplier: float = 1.5,
            is_origin_time_column_drop=True,
            ohe_columns: list[str] | None = None
    ):
        """
        This method is responsible for the final preparation of the dataframe. It performs several operations such as
        converting indices to integers, dropping specified columns, removing outliers, adding time features, one-hot encoding
        categorical columns, and normalizing numeric features.

        Parameters:
        df (DataFrame, optional): The dataframe to be prepared. If not provided, the method will use the dataframe already
                                  stored in the instance.
        save_path (str, optional): The path where the prepared dataframe should be saved. If not provided, the method will
                                   return the prepared dataframe.
        banks_cols_to_drop (list[str], optional): The list of bank columns to be dropped from the dataframe. If not provided,
                                                  the method will use the list already stored in the instance.
        condition_value (str, default='success'): The condition value to be used when removing outliers.
        multiplier (float, default=1.5): The multiplier to be used when removing outliers.
        is_origin_time_column_drop (bool, default=True): Whether to drop the original time column when adding time features.
        ohe_columns (list[str], optional): The list of columns to be one-hot encoded. If not provided, the method will one-hot
                                           encode all categorical columns.

        Returns:
        DataFrame: The prepared dataframe if save_path is not provided.
        """

        if df:
            self.df = df
        if banks_cols_to_drop:
            self.banks_cols_to_drop = banks_cols_to_drop

        # Convert the index of the dataframe to integer
        self.index_as_int()

        # Drop specified columns from the dataframe
        self.drop_columns(columns=self.banks_cols_to_drop)

        # Remove outliers from all numeric columns based on the provided condition value and multiplier
        self.remove_outliers_all_numeric_with_condition(
            is_new=False,
            condition_value=condition_value,
            multiplier=multiplier
        )

        # Add time features to the dataframe and optionally drop the original time column
        self.add_time_features(is_drop=is_origin_time_column_drop)

        # One-hot encode specified columns or all categorical columns if no columns are specified
        self.ohe_categorical_columns(is_new=False, columns=ohe_columns)

        # Normalize all numeric features in the dataframe
        self.normalize_numeric_features(is_new=False, columns=self.money_columns)

        # If save_path is provided, save the dataframe to the specified path
        if save_path:
            self.save_df(path=save_path)
        else:
            # If save_path is not provided, return the prepared dataframe
            return self.df


if __name__ == '__main__':
    from os import path
    from pandas import read_csv

    load_path = path.abspath(path.join(__file__, '../../../data/datasets/SF_Mandarin_dataset_ver3.csv'))
    full_prepared_save_path = path.abspath(path.join(__file__, '../../../data/datasets/prepared_full.parquet'))
    one_bank_prepared_save_path = path.abspath(path.join(__file__, '../../../data/datasets/prepared_one_bank.parquet'))

    target_bank_col = 'bank_a_decision'
    banks_cols = [f'bank_{i}_decision' for i in ['a', 'b', 'c', 'd', 'e']]
    banks_to_drop = banks_cols.copy()
    banks_to_drop.remove(target_bank_col)
    categorical_columns = ['family_status', 'goods_category', 'position', 'employment_status', 'education']
    numeric_categorical_columns = ['snils', 'gender', 'merch_code', 'child_count', 'loan_term']
    money_columns = ['month_profit', 'month_expense', 'loan_amount']

    df = read_csv(load_path, index_col=0, sep=';')
    dpp = DataPreprocessing(
        df=df,
        banks_cols_to_drop=banks_to_drop,
        target_bank_col=target_bank_col,
        banks_cols=banks_cols,
        categorical_columns=categorical_columns,
        numeric_categorical_columns=numeric_categorical_columns,
        money_columns=money_columns
    )

    dpp.preparation_full_df(save_path=full_prepared_save_path)
    dpp.final_preparation(save_path=one_bank_prepared_save_path)
