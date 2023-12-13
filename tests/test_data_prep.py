import numpy as np
import pandas as pd
from src.data_preparation.data_preparation import DataPreparation


class TestDataPreparation:
    def test_is_normality_distributed(self, normal_data_preparation):
        # Call the is_normality_distributed method
        distribution = normal_data_preparation.is_normality_distributed()

        # Check the output
        assert 'age' in distribution, "Age is a numeric column, but was not included in the output"
        assert distribution['age'][
                   'is_normal'] == True, f"Age is normally distributed, but the output is {distribution['age']['is_normal']}"
        assert 'income' in distribution, "Income is a numeric column, but was not included in the output"
        assert distribution['income'][
                   'is_normal'] == True, f"Income is normally distributed, but the output is {distribution['income']['is_normal']}"

    def test_is_not_normality_distributed(self, lognormal_data_preparation):
        # Call the is_normality_distributed method
        distribution = lognormal_data_preparation.is_normality_distributed()

        # Check the output
        assert 'age' in distribution, "age is a numeric column, but was not included in the output"
        assert distribution['age'][
                   'is_normal'] == False, f"Age is normally distributed, but the output is {distribution['age'][0]}"
        assert 'income' in distribution, "income is a numeric column, but was not included in the output"
        assert distribution['income'][
                   'is_normal'] == False, f"Income is normally distributed, but the output is {distribution['income'][0]}"

    def test_no_numeric_columns_is_not_in_output(self, normal_data_preparation):
        # Call the is_normality_distributed method
        distribution = normal_data_preparation.is_normality_distributed()
        assert 'Education' not in distribution, "Education is not a numeric column, but was included in the output"
        assert 'Decision' not in distribution, "Decision is not a numeric column, but was included in the output"

    def test_transform_to_categorical(self, normal_data_preparation):
        # Call the transform_to_categorical method
        normal_data_preparation.transform_to_categorical(['education'])
        # Check if the column has been transformed to categorical
        categorical_columns = normal_data_preparation.df.select_dtypes(include='category').columns
        assert 'education' in categorical_columns, "Education is a categorical column, but was not included in the output"

    def test_drop_columns(self, normal_data_preparation):
        # Call the drop_columns method
        normal_data_preparation.drop_columns(columns=['education', 'bank_decision'])

        # Check if the 'Decision' column has been dropped
        assert 'decision' not in normal_data_preparation.df.columns

    def test_add_time_features(self):
        # Create a sample DataFrame
        data = {
            'birth_date': pd.date_range(start='1990-01-01', periods=3),
            'job_start_date': pd.date_range(start='2010-01-01', periods=3)
        }
        df = pd.DataFrame(data)

        # Create an instance of DataPreparation
        data_prep = DataPreparation(df=df, to_drop_columns=[], target_bank_col='Bank')

        # Call the add_time_features method
        data_prep.add_time_features()

        # Check if the time features are added correctly
        assert 'birth_year' in data_prep.df.columns
        assert 'age' in data_prep.df.columns
        assert 'job_start_year' in data_prep.df.columns
        assert 'job_experience' in data_prep.df.columns

        # Check if the original time columns are dropped
        assert 'birth_date' not in data_prep.df.columns
        assert 'job_start_date' not in data_prep.df.columns

        # Check the values of the time features
        assert data_prep.df['birth_year'].tolist() == [1990, 1990, 1990]
        assert data_prep.df['age'].tolist() == [33, 33, 33]
        assert data_prep.df['job_start_year'].tolist() == [2010, 2010, 2010]
        assert data_prep.df['job_experience'].tolist() == [13, 13, 13]

    def test_normalize_numeric_features(self, lognormal_data_preparation):
        # Call the normalize_numeric_features method
        lognormal_data_preparation.normalize_numeric_features(columns=['age', 'income'], is_save=False)

        # Check if the numeric columns are normalized
        assert 'age' in lognormal_data_preparation.df.columns
        assert 'income' in lognormal_data_preparation.df.columns

        # Check if the values in the numeric columns are normalized
        assert np.isclose(lognormal_data_preparation.df['age'].mean(), 0, atol=1e-2)
        assert np.isclose(lognormal_data_preparation.df['age'].std(), 1, atol=1e-2)
        assert np.isclose(lognormal_data_preparation.df['income'].mean(), 0, atol=1e-2)
        assert np.isclose(lognormal_data_preparation.df['income'].std(), 1, atol=1e-2)

    def test_ohe_categorical_columns_default_columns(self):
        # Create a sample DataFrame
        data = {'education': ['Bachelor', 'Master', 'PhD', 'Bachelor'],
                'employment_status': ['Employed', 'Unemployed', 'Employed', 'Employed'],
                'gender': ['Man', 'Woman', 'Man', 'Woman'],
                'family_status': ['Married', 'Single', 'Married', 'Single'],
                'child_count': [0, 1, 0, 3],
                'loan_term': [6, 12, 18, 24],
                'goods_category': ['Electronics', 'Furniture', 'Electronics', 'Education'],
                'value': [1, 3, 2, 4],
                'snils': [1, 0, 1, 0],
                'merch_code': [111, 222, 333, 444],
                'bank': ['Bank1', 'Bank2', 'Bank3', 'Bank4']
                }
        data_prep = DataPreparation(df=pd.DataFrame(data), to_drop_columns=[], target_bank_col='bank')
        data_prep.ohe_categorical_columns()
        # Check if the categorical columns are one-hot encoded
        assert 'education' not in data_prep.ohe_df.columns
        assert 'employment status' not in data_prep.ohe_df.columns
        assert 'Gender' not in data_prep.ohe_df.columns
        assert 'Family status' not in data_prep.ohe_df.columns
        assert 'ChildCount' not in data_prep.ohe_df.columns
        assert 'Loan_term' not in data_prep.ohe_df.columns
        assert 'Goods_category' not in data_prep.ohe_df.columns
        assert 'Value' not in data_prep.ohe_df.columns
        assert 'SNILS' not in data_prep.ohe_df.columns
        assert 'Merch_code' not in data_prep.ohe_df.columns

        # Check if the new DataFrame with one-hot encoded columns is stored in ohe_df attribute
        assert hasattr(data_prep, 'ohe_df')
        assert isinstance(data_prep.ohe_df, pd.DataFrame)

    def test_ohe_categorical_columns_custom_columns(self, normal_data_preparation):
        normal_data_preparation.ohe_categorical_columns(columns=['education', 'bank_decision'], is_new=True)

        # Check if the custom categorical columns are one-hot encoded
        assert 'education' not in normal_data_preparation.ohe_df.columns
        assert 'bank_decision' not in normal_data_preparation.ohe_df.columns

        # Check if the new DataFrame with one-hot encoded columns is stored in ohe_df attribute
        assert hasattr(normal_data_preparation, 'ohe_df')
        assert isinstance(normal_data_preparation.ohe_df, pd.DataFrame)

    def test_ohe_categorical_columns_existing_df(self, lognormal_data_preparation):
        lognormal_data_preparation.ohe_categorical_columns(columns=['education', 'bank_decision'], is_new=False)

        # Check if the categorical columns are one-hot encoded
        assert 'Education' not in lognormal_data_preparation.df.columns
        assert 'Bank_decision' not in lognormal_data_preparation.df.columns

        # Check if the original DataFrame is updated with one-hot encoded columns
        assert isinstance(lognormal_data_preparation.df, pd.DataFrame)
