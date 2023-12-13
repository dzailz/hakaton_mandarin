import logging
import pickle
from pathlib import Path

import pandas as pd
from box import ConfigBox
from dvclive import Live
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
from src.models.random_forest.random_forest import RandomForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

RANDOM_SEED = params.base.random_seed
N_ESTIMATORS = list(params.train.random_forest.n_estimators)


def train_predict_rf(df: pd.DataFrame, bank: str):
    """
    This function trains and predicts a Random Forest model for a given bank.

    Parameters:
    df (pd.DataFrame): The input dataframe containing the data to be used for training and prediction.
    bank (str): The name of the bank for which the model is being trained.

    Returns:
    None
    """

    # Iterate over the list of estimators
    for n_estimators in N_ESTIMATORS:
        # Log the number of estimators being used for the current iteration
        logger.info(f"Training Random Forest with {n_estimators} estimators")

        # Initialize a Live instance for logging metrics and parameters
        with Live(dir=RESULTS_FOLDER, dvcyaml=DVC_YAML_FILE) as live:
            # Log the number of estimators as a parameter
            live.log_param("n_estimators", n_estimators)
            # Initialize a RandomForest instance
            rf = RandomForest(df=df, bank=bank, n_estimators=n_estimators)
            # Apply SMOTE to the data
            rf.add_smote()
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = data_split(rf.X_resampled, rf.y_resampled)
            # Fit the model to the training data
            model = rf.fit(X_train, y_train)
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
            y_test = y_test.replace({"denied": 0, "success": 1})
            live.log_sklearn_plot(
                "roc",
                y_test,
                predictions=proba,
                name=f"test/{bank}_roc_curve",
            )
            live.log_sklearn_plot(
                "precision_recall",
                y_test,
                proba,
                name=f"test/{bank}_precision_recall_curve",
            )
            model_name = f"random_forest_{n_estimators}_{bank}.pkl"
            # Save the trained model to a file
            model_path = Path(MODELS_FOLDER, model_name)
            pickle.dump(model, open(model_path, "wb"))

            # Log the model as an artifact
            live.log_artifact(path=MODELS_FOLDER, type="model", name=model_name, labels=[n_estimators, "random_forest", bank])


if __name__ == "__main__":
    import re

    for df_path in Path(DATASETS_FOLDER).glob("*.parquet"):
        if re.search(r"bank_(\w+)", df_path.name):
            bank_name = re.search(r"bank_(\w+)", df_path.name).group(0)
            df = pd.read_parquet(df_path)
            df.drop("position", axis=1, inplace=True)
            train_predict_rf(df=df.copy(), bank=bank_name)
