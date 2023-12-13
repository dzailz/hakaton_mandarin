from typing import Any, Literal, Mapping, Sequence

from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE


class RandomForest(RandomForestClassifier):
    def __init__(
            self,
            df: DataFrame,
            bank: str | None = None,
            n_estimators: int = 100,
            random_state: int = 42,
            criterion: Literal['gini', 'entropy', 'log_loss'] = "gini",
            max_features: float | int | Literal['sqrt', 'log2'] = "sqrt",
            class_weight: Mapping | Sequence[Mapping] | Literal['balanced', 'balanced_subsample'] | None = None,
            X_resampled: Any | None = None,
            y_resampled: Any | None = None,
            X_train: Any | None = None,
            X_test: Any | None = None,
            y_train: Any | None = None,
            y_test: Any | None = None,
    ):
        self.df = df
        self.bank = bank
        self.bank_decision = f"{self.bank}_decision"
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        super().__init__(
            n_estimators=n_estimators,
            random_state=random_state,
            criterion=criterion,
            max_features=max_features,
            class_weight=class_weight
        )

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
        X = df.drop(self.bank_decision, axis=1)
        y = df[self.bank_decision]

        # Apply SMOTE after scaling numerical features
        # (commented because data is already scaled)
        # scaler = StandardScaler()
        # X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

        X_resampled, y_resampled = SMOTE().fit_resample(X, y)
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled

    def save_to_pickle(self, path: str):
        pass

# if __name__ == '__main__':
#     import pandas as pd
#     import pickle
#     from os import path
#     from dvclive import Live
#     from src.models.common.split import data_split


#     df_path = path.join(path.dirname(__file__), '../../data/datasets/prepared_one_bank.parquet')

#     df = pd.read_parquet(df_path)
#     df.drop('position', axis=1, inplace=True)

#     for n_estimators in (50, 100, 150):
#         with Live() as live:
#             live.log_param("n_estimators", n_estimators)
#             rf = RandomForest(df=df, n_estimators=n_estimators)
#             rf.add_smote()
#             X_train, X_test, y_train, y_test = data_split(rf.X_resampled, rf.y_resampled)

#             rf.fit(X_train, y_train)

#             y_train_pred = rf.predict(X_train)

#             live.log_metric("train/classification_report", classification_report(y_train, y_train_pred), plot=True)
#             live.log_metric("train/f1", f1_score(y_train, y_train_pred, average="weighted"), plot=True)
#             live.log_metric("ROC_AUC", roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]), plot=True)

#             live.log_sklearn_plot(
#                 "confusion_matrix", y_train, y_train_pred, name="train/confusion_matrix",
#                 title="Train Confusion Matrix")

#             y_test_pred = rf.predict(X_test)

#             live.log_metric("test/classification_report", classification_report(y_test, y_test_pred), plot=False)
#             live.log_metric("test/f1", f1_score(y_test, y_test_pred, average="weighted"), plot=False)
#             live.log_metric("ROC_AUC", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), plot=False)

#             live.log_sklearn_plot(
#                 "confusion_matrix", y_test, y_test_pred, name="test/confusion_matrix",
#                 title="Test Confusion Matrix")

#             model_path = path.join(path.dirname(__file__),
#                                    f'../../data/trained_models/random_forest_{n_estimators}.pkl')
#             pickle.dump(rf, open(model_path, 'wb'))
#             live.log_artifact(model_path)
