from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from settings import DATASETS_FOLDER


class IsolationForestMinorityClassPredictor(IsolationForest):
    def __init__(
        self,
        df: DataFrame,
        bank: str | None = None,
        random_state: int = 42,
        contamination: float = 0.05,  # Adjust the contamination parameter as needed
        # max_features: float | int | str = "auto",
    ):
        self.df = df
        self.bank = bank
        self.bank_decision = f"{self.bank}_decision"

        super().__init__(
            random_state=random_state,
            contamination=contamination,
            # max_features=max_features
        )

    @staticmethod
    def numerics_and_non_numerics(df: DataFrame) -> tuple[list[str], list[str]]:
        numeric_columns = df.select_dtypes(include=["number"]).columns
        non_numeric_columns = df.select_dtypes(exclude=["number"]).columns
        return numeric_columns, non_numeric_columns

    @staticmethod
    def convert_datetime_to_numeric(non_numeric_columns: list[str], df: DataFrame) -> None:
        for column in non_numeric_columns:
            if df[column].dtype == "datetime64[ns]":
                df[column] = df[column].dt.year

    def prepare_data(self):
        numeric_columns, non_numeric_columns = self.numerics_and_non_numerics(df=self.df)
        self.convert_datetime_to_numeric(df=self.df, non_numeric_columns=non_numeric_columns)

        X = self.df.drop(self.bank_decision, axis=1)

        # Assign labels based on the minority class
        y = self.df[self.bank_decision].apply(lambda x: 1 if x == "denied" else 0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.random_state)

        return X_train, X_test, y_train, y_test

    def train_and_evaluate(self):
        X_train, X_test, y_train, y_test = self.prepare_data()

        self.fit(X_train)

        # Make predictions
        y_pred = self.predict(X_test)

        # Evaluate performance
        print("Classification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    import re

    for df_path in Path(DATASETS_FOLDER).glob("*.parquet"):
        if re.search(r"bank_(\w+)", df_path.name):
            bank_name = re.search(r"bank_(\w+)", df_path.name).group(0)
            df = pd.read_parquet(df_path)
            df.drop("position", axis=1, inplace=True)
            # Assuming df is your DataFrame and "bank_decision" is the target variable
            predictor = IsolationForestMinorityClassPredictor(df=df, bank=bank_name, random_state=42)
            predictor.train_and_evaluate()
