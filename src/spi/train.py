import pandas as pd
import pickle
import logging

from pathlib import Path
from dvclive import Live
from box import ConfigBox
from ruamel.yaml import YAML
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve, auc

from settings import DVC_PARAMS_FILE, MODELS_FOLDER, DATASETS_FOLDER, RESULTS_FOLDER, DVC_YAML_FILE
from src.models.common.split import data_split
from src.models.random_forest.random_forest import RandomForest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

yaml = YAML(typ='safe')

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
            rf.fit(X_train, y_train)

            # Predict the training data
            y_train_pred = rf.predict(X_train)

            # Generate a classification report for the training data
            classification_report_train = classification_report(y_train, y_train_pred, output_dict=False)

            # Calculate the F1 score for the training data
            f1_score_train = f1_score(y_train, y_train_pred, average="weighted")

            # Calculate the ROC AUC score for the training data
            roc_auc_score_train = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])

            # Log the classification report, F1 score, and ROC AUC score for the training data
            live.log_metric("train/classification_report", classification_report_train, plot=False)
            logger.info(f"Train classification report: {classification_report_train}")
            live.log_metric("train/f1", f1_score_train, plot=False)
            logger.info(f"Train f1 score: {f1_score_train}")
            live.log_metric("ROC_AUC", roc_auc_score_train, plot=False)
            logger.info(f"Train ROC AUC score: {roc_auc_score_train}")
            # Log the confusion matrix, ROC curve, and precision-recall curve for the training data
            live.log_sklearn_plot("confusion_matrix",
                                  y_train, y_train_pred, name="train/confusion_matrix", title="Train Confusion Matrix")
            y_train = y_train.replace({'denied': 0, 'success': 1})
            live.log_sklearn_plot("roc", y_train,
                                  rf.predict_proba(X_train)[:, 1], name="train/roc_curve", )
            # Predict the testing data
            y_test_pred = rf.predict(X_test)
            # Generate a classification report for the testing data
            classification_report_test = classification_report(y_test, y_test_pred, output_dict=False)
            # Calculate the F1 score for the testing data
            f1_score_test = f1_score(y_test, y_test_pred, average="weighted")
            # Calculate the ROC AUC score for the testing data
            roc_auc_score_test = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
            # Log the classification report, F1 score, and ROC AUC score for the testing data
            live.log_metric("test/classification_report", classification_report_test, plot=False)
            logger.info(f"Test classification report: {classification_report_test}")
            live.log_metric("test/f1", f1_score_test, plot=False)
            logger.info(f"Test f1 score: {f1_score_test}")
            live.log_metric("test/ROC_AUC", roc_auc_score_test, plot=False)
            logger.info(f"Test ROC AUC score: {roc_auc_score_test}")
            # Log the confusion matrix, ROC curve, and precision-recall curve for the testing data
            live.log_sklearn_plot("confusion_matrix",
                                  y_test, y_test_pred, name="test/confusion_matrix", title="Test Confusion Matrix")
            y_test = y_test.replace({'denied': 0, 'success': 1})
            live.log_sklearn_plot("roc", y_test,
                                  rf.predict_proba(X_test)[:, 1], name="test/roc_curve",)
            live.log_sklearn_plot(
                "precision_recall",
                y_test,
                rf.predict_proba(X_test)[:, 1],
                name="test/precision_recall_curve",
            )
            model_name = f"random_forest_{n_estimators}{'_' + bank}.pkl"
            # Save the trained model to a file
            model_path = Path(MODELS_FOLDER, model_name)
            pickle.dump(rf, open(model_path, 'wb'))

            # Log the model as an artifact
            # live.log_artifact(
            #     path=MODELS_FOLDER,
            #     type="model",
            #     name=model_name,
            #     labels=[n_estimators, "random_forest", bank]
            # )


if __name__ == '__main__':
    import re

    for df_path in Path(DATASETS_FOLDER).glob('*.parquet'):
        if re.search(r'bank_(\w+)', df_path.name):
            bank_name = re.search(r'bank_(\w+)', df_path.name).group(0)
            df = pd.read_parquet(df_path)
            df.drop('position', axis=1, inplace=True)
            train_predict_rf(df=df.copy(), bank=bank_name)

    # df_path = Path(DATASETS_FOLDER, 'prepared_one_bank.parquet')
    # df = pd.read_parquet(df_path)
    # df.drop('position', axis=1, inplace=True)
    #
    # train_predict_rf(df=df.copy())
