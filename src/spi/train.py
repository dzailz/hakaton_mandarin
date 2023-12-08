import pandas as pd
import pickle
import logging

from pathlib import Path
from dvclive import Live
from box import ConfigBox
from ruamel.yaml import YAML
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from settings import DVC_PARAMS_FILE, MODELS_FOLDER, DATASETS_FOLDER, RESULTS_FOLDER, DVC_YAML_FILE
from src.models.common.split import data_split
from src.models.random_forest.random_forest import RandomForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ='safe')

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

RANDOM_SEED = params.base.random_seed
N_ESTIMATORS = list(params.train.random_forest.n_estimators)


def train_predict_rf(df: pd.DataFrame):
    """
    This function trains and predicts a Random Forest model on the given DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to be used for training and prediction.

    The function iterates over a list of n_estimators, and for each n_estimator:
    - Logs the current n_estimator being used.
    - Initializes a Random Forest model with the current n_estimator.
    - Applies SMOTE to the data.
    - Splits the data into training and testing sets.
    - Fits the model on the training data.
    - Makes predictions on the training data.
    - Calculates and logs the classification report, f1 score, and ROC AUC score for the training data.
    - Logs a confusion matrix plot for the training data.
    - Makes predictions on the testing data.
    - Calculates and logs the classification report, f1 score, and ROC AUC score for the testing data.
    - Logs a confusion matrix plot for the testing data.
    - Saves the trained model as a pickle file.
    - Logs the model as an artifact.

    """
    for n_estimators in N_ESTIMATORS:
        logger.info(f"Training Random Forest with {n_estimators} estimators")
        with Live(dir=RESULTS_FOLDER, dvcyaml=DVC_YAML_FILE) as live:
            live.log_param("n_estimators", n_estimators)

            rf = RandomForest(df=df, n_estimators=n_estimators)

            rf.add_smote()
            X_train, X_test, y_train, y_test = data_split(rf.X_resampled, rf.y_resampled)

            rf.fit(X_train, y_train)

            y_train_pred = rf.predict(X_train)
            classification_report_train = classification_report(y_train, y_train_pred, output_dict=True)
            f1_score_train = f1_score(y_train, y_train_pred, average="weighted")
            roc_auc_score_train = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])

            live.log_metric("train/classification_report", classification_report_train, plot=True)
            logger.info(f"Train classification report: {classification_report_train}")
            live.log_metric("train/f1", f1_score_train, plot=True)
            logger.info(f"Train f1 score: {f1_score_train}")
            live.log_metric("ROC_AUC", roc_auc_score_train, plot=True)
            logger.info(f"Train ROC AUC score: {roc_auc_score_train}")

            live.log_sklearn_plot(
                "confusion_matrix", y_train, y_train_pred, name="train/confusion_matrix",
                title="Train Confusion Matrix")

            y_test_pred = rf.predict(X_test)

            classification_report_test = classification_report(y_test, y_test_pred, output_dict=True)
            f1_score_test = f1_score(y_test, y_test_pred, average="weighted")
            roc_auc_score_test = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

            live.log_metric("test/classification_report", classification_report_test, plot=True)
            logger.info(f"Test classification report: {classification_report_test}")
            live.log_metric("test/f1", f1_score_test, plot=True)
            logger.info(f"Test f1 score: {f1_score_test}")
            live.log_metric("ROC_AUC", roc_auc_score_test, plot=True)
            logger.info(f"Test ROC AUC score: {roc_auc_score_test}")

            live.log_sklearn_plot(
                "confusion_matrix", y_test, y_test_pred, name="test/confusion_matrix",
                title="Test Confusion Matrix")

            model_path = Path(MODELS_FOLDER, f'random_forest_{n_estimators}.pkl')
            pickle.dump(rf, open(model_path, 'wb'))
            live.log_artifact(
                path=model_path,
                type="model",
                name=f"random_forest_{n_estimators}.pkl",
                labels=[n_estimators, "random_forest"]
            )


if __name__ == '__main__':
    df_path = Path(DATASETS_FOLDER, 'prepared_one_bank.parquet')

    df = pd.read_parquet(df_path)
    df.drop('position', axis=1, inplace=True)

    train_predict_rf(df=df.copy())
