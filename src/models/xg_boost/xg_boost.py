from typing import Any, Literal

from imblearn.over_sampling import ADASYN
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


class XGBoost(XGBClassifier):
    def __init__(
        self,
        df: DataFrame,
        bank: str | None = None,
        n_estimators: int = 100,
        random_state: int = 42,
        objective: Literal["binary:logistic", "multi:softmax"] = "binary:logistic",
        booster: Literal["gbtree", "gblinear", "dart"] = "gbtree",
        early_stopping_rounds: int | None = None,
        eval_metric: str | list[str] | None = None,
        callbacks: list[Any] | None = None,
        max_depth: int = 3,
        enable_categorical: bool = False,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        scale_pos_weight: float = 1.0,
        base_score: float = 0.5,
        reg_alpha: float = 0,
        reg_lambda: float = 1,
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
            early_stopping_rounds=early_stopping_rounds,
            eval_metric=eval_metric,
            enable_categorical=enable_categorical,
            callbacks=callbacks,
            n_estimators=n_estimators,
            random_state=random_state,
            objective=objective,
            booster=booster,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )

    @staticmethod
    def numerics_and_non_numerics(df: DataFrame) -> tuple[list[str], list[str]]:
        """
        Extract numeric and non-numeric columns from a DataFrame
        """
        # Extract numerical columns
        numeric_columns = df.select_dtypes(include=["number"]).columns

        # Extract non-numeric columns, including datetime columns
        non_numeric_columns = df.select_dtypes(exclude=["number"]).columns

        return numeric_columns, non_numeric_columns

    @staticmethod
    def convert_datetime_to_numeric(non_numeric_columns: list[str], df: DataFrame) -> None:
        """
        Convert datetime columns to numeric (year only)
        """
        for column in non_numeric_columns:
            if df[column].dtype == "datetime64[ns]":
                df[column] = df[column].dt.year

    def add_smote(self, df: DataFrame = None, numeric_columns: list[str] | None = None, non_numeric_columns: list[str] | None = None):
        """
        Apply Synthetic Minority Over-sampling Technique (SMOTE) to the data.

        This method applies SMOTE to the data to balance the classes in the target variable.
        It first separates the features and the target variable, then applies SMOTE to the features and target variable.
        The resampled features and target variable are then stored in the instance variables self.X_resampled and self.y_resampled.

        Args:
            df (DataFrame, optional): The DataFrame to apply SMOTE to. If not provided, the instance variable self.df is used.
            numeric_columns (list[str] | None, optional): The list of numeric column names. If not provided, it is extracted from the DataFrame.
            non_numeric_columns (list[str] | None, optional): The list of non-numeric column names. If not provided, it is extracted from the DataFrame.

        Returns:
            None

        Raises:
            ValueError: If the DataFrame is empty or if the target variable column does not exist in the DataFrame.
            TypeError: If the input data is not of the correct type.
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

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        # Apply SMOTE
        X_resampled, y_resampled = ADASYN(n_neighbors=7, sampling_strategy="minority").fit_resample(X, y)
        self.X_resampled = X_resampled
        self.y_resampled = y_resampled
