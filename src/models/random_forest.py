from typing import Any

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dvclive import Live
from imblearn.over_sampling import SMOTE


class RandomForest(RandomForestClassifier):
    def __init__(self, df: DataFrame, n_estimators: int = 100,
                 X_resampled: Any = None, y_resampled: Any = None,
                 X_train: Any = None, X_test: Any = None, y_train: Any = None, y_test: Any = None):
        self.df = df
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        super().__init__(n_estimators=n_estimators)

    @staticmethod
    def numerics_and_non_numerics(df: DataFrame) -> tuple[list[str], list[str]]:
        """
        Extract numeric and non-numeric columns from a DataFrame
        """
        # Extract numerical columns
        numeric_columns = df.select_dtypes(include=['number']).columns

        # Extract non-numeric columns, including datetime columns
        non_numeric_columns = df.select_dtypes(exclude=['number']).columns

        return numeric_columns, non_numeric_columns

    @staticmethod
    def convert_datetime_to_numeric(non_numeric_columns: list[str], df: DataFrame) -> None:
        """
        Convert datetime columns to numeric (year only)
        """
        for column in non_numeric_columns:
            if df[column].dtype == 'datetime64[ns]':
                df[column] = df[column].dt.year

    def add_smote(self, df: DataFrame = None, numeric_columns: list[str] | None = None,
                  non_numeric_columns: list[str] | None = None):
        """
        Add SMOTE to the data
        """
        # Extract numerical columns
        if not df:
            df = self.df

        if not numeric_columns or not non_numeric_columns:
            numeric_columns, non_numeric_columns = self.numerics_and_non_numerics(df=df)

        self.convert_datetime_to_numeric(df=df, non_numeric_columns=non_numeric_columns)

        # Separate features and target variable
        X = df.drop('bank_a_decision', axis=1)
        y = df['bank_a_decision']

        # Apply SMOTE after scaling numerical features
        scaler = StandardScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled

    def split_data(self, X_resampled: Any = None, y_resampled: Any = None, test_size: float = 0.2,
                   random_state: int = 42):
        if not X_resampled or y_resampled:
            X_resampled = self.X_resampled
            y_resampled = self.y_resampled

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size,
                                                            random_state=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def save_to_pickle(self, path: str):
        pass


if __name__ == '__main__':
    import pandas as pd
    import pickle
    from os import path

    df_path = path.join(path.dirname(__file__), '../../data/datasets/prepared_one_bank.parquet')

    df = pd.read_parquet(df_path)
    df.drop('position', axis=1, inplace=True)

    for n_estimators in (50, 100, 150):
        with Live() as live:
            live.log_param("n_estimators", n_estimators)
            rf = RandomForest(df=df, n_estimators=n_estimators)
            rf.add_smote()
            rf.split_data()

            rf.fit(rf.X_train, rf.y_train)

            y_train_pred = rf.predict(rf.X_train)

            live.log_metric("train/classification_report", classification_report(rf.y_train, y_train_pred), plot=False)
            live.log_metric("train/f1", f1_score(rf.y_train, y_train_pred, average="weighted"), plot=False)
            live.log_metric("ROC_AUC", roc_auc_score(rf.y_train, rf.predict_proba(rf.X_train)[:, 1]), plot=False)

            live.log_sklearn_plot(
                "confusion_matrix", rf.y_train, y_train_pred, name="train/confusion_matrix",
                title="Train Confusion Matrix")

            y_test_pred = rf.predict(rf.X_test)

            live.log_metric("test/classification_report", classification_report(rf.y_test, y_test_pred), plot=False)
            live.log_metric("test/f1", f1_score(rf.y_test, y_test_pred, average="weighted"), plot=False)
            live.log_metric("ROC_AUC", roc_auc_score(rf.y_test, rf.predict_proba(rf.X_test)[:, 1]), plot=False)

            live.log_sklearn_plot(
                "confusion_matrix", rf.y_test, y_test_pred, name="test/confusion_matrix",
                title="Test Confusion Matrix")

            model_path = path.join(path.dirname(__file__),
                                   f'../../data/trained_models/random_forest_{n_estimators}.pkl')
            pickle.dump(rf, open(model_path, 'wb'))
