import logging
from pathlib import Path

import pandas as pd
from box import ConfigBox
from dvclive import Live
from dvclive.xgb import DVCLiveCallback
from ruamel.yaml import YAML
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from settings import (
    DATASETS_FOLDER,
    DVC_PARAMS_FILE,
    DVC_YAML_FILE,
    MODELS_FOLDER,
    RESULTS_FOLDER,
)
from src.models.common.split import data_split
from src.models.xg_boost.xg_boost import XGBoost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

RANDOM_SEED = params.base.random_seed
N_ESTIMATORS = params.train.random_forest.n_estimators


def train_predict_xg(df: pd.DataFrame, bank: str):
    """
    This function trains and predicts a Random Forest model for a given bank.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be used for training and prediction.
    bank (str): The name of the bank for which the model is being trained.

    Returns:
    None
    """

    # Initialize a Live instance for logging metrics and parameters
    with Live(dir=RESULTS_FOLDER, dvcyaml=DVC_YAML_FILE) as live:
        # Log the number of estimators as a parameter
        live.log_param("n_estimators", N_ESTIMATORS)
        # Initialize a RandomForest instance
        xgb = XGBoost(
            df=df,
            bank=bank,
            n_estimators=N_ESTIMATORS,
            early_stopping_rounds=5,
            eval_metric=["error", "logloss"],  # Update metrics if needed
            callbacks=[DVCLiveCallback(dir=RESULTS_FOLDER, dvcyaml=DVC_YAML_FILE)],
            scale_pos_weight=0.5,
            objective="binary:logistic",  # Set the objective for binary classification
        )
        # Apply SMOTE to the data
        xgb.add_smote()
        # Split the data into training, validation, and testing sets
        X_train, X_temp, y_train, y_temp = data_split(xgb.X_resampled, xgb.y_resampled, test_size=0.2, random_state=RANDOM_SEED)
        X_val, X_test, y_val, y_test = data_split(X_temp, y_temp, test_size=0.5, random_state=RANDOM_SEED)

        # Fit the model to the training data with early stopping on the validation set
        model = xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

        # Predict the testing data
        y_test_pred = model.predict(X_test)
        # Generate a classification report for the testing data
        classification_report_test = classification_report(y_test, y_test_pred, output_dict=False)
        # Calculate the F1 score for the testing data
        f1_score_test = f1_score(y_test, y_test_pred, average="weighted")
        proba = model.predict_proba(X_test)[:, 1]
        # Calculate the ROC AUC score for the testing data
        roc_auc_score_test = roc_auc_score(y_test, proba)
        # Log the classification report, F1 score, and ROC AUC score for the testing data
        live.log_metric(f"test/{bank}_classification_report", classification_report_test, plot=False)
        logger.info(f"Test {bank} classification report: {classification_report_test}")

        live.log_metric(f"test/{bank}_f1", f1_score_test, plot=False)
        logger.info(f"{bank}_f1 score: {f1_score_test}")

        live.log_metric(f"test/{bank}ROC_AUC", roc_auc_score_test, plot=False)
        logger.info(f"{bank}_ROC AUC score: {roc_auc_score_test}")
        # Log the confusion matrix, ROC curve, and precision-recall curve for the testing data
        live.log_sklearn_plot("confusion_matrix", y_test, y_test_pred, name=f"test/{bank}_confusion_matrix", title=f"Test {bank} Confusion Matrix")
        live.log_metric("summary_metric", 1.0, plot=False)

        model_name = f"XGBoost{N_ESTIMATORS}{bank.capitalize()}.pkl"
        # Save the trained model to a file
        model_path = Path(MODELS_FOLDER, model_name)
        model.save_model(model_path)

        # Log the model as an artifact
        live.log_artifact(path=MODELS_FOLDER, type="model", name=model_name, labels=[N_ESTIMATORS, "random_forest", bank])


if __name__ == "__main__":
    import re

    for df_path in Path(DATASETS_FOLDER).glob("*.parquet"):
        if re.search(r"bank_(\w+)", df_path.name):
            bank_name = re.search(r"bank_(\w+)", df_path.name).group(0)
            df = pd.read_parquet(df_path)
            df.drop("position", axis=1, inplace=True)
            train_predict_xg(df=df.copy(), bank=bank_name)
