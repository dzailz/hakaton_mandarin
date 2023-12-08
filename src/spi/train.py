import pandas as pd
import pickle

from pathlib import Path
from dvclive import Live
from box import ConfigBox
from ruamel.yaml import YAML
from sklearn.metrics import classification_report, f1_score, roc_auc_score

from src.models.common.split import data_split
from src.models.random_forest.random_forest import RandomForest

yaml = YAML(typ='safe')
params = ConfigBox(yaml.load(open('params.yaml')))

RANDOM_SEED = params.base.random_seed
N_ESTIMATORS = params.train.random_forest.n_estimators


def train_predict_rf(df: pd.DataFrame):
    for n_estimators in N_ESTIMATORS:
        with Live() as live:
            live.log_param("n_estimators", n_estimators)
            rf = RandomForest(df=df, n_estimators=n_estimators)
            rf.add_smote()
            X_train, X_test, y_train, y_test = data_split(rf.X_resampled, rf.y_resampled)

            rf.fit(X_train, y_train)

            y_train_pred = rf.predict(X_train)

            live.log_metric("train/classification_report", classification_report(y_train, y_train_pred), plot=True)
            live.log_metric("train/f1", f1_score(y_train, y_train_pred, average="weighted"), plot=True)
            live.log_metric("ROC_AUC", roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1]), plot=True)

            live.log_sklearn_plot(
                "confusion_matrix", y_train, y_train_pred, name="train/confusion_matrix",
                title="Train Confusion Matrix")

            y_test_pred = rf.predict(X_test)

            live.log_metric("test/classification_report", classification_report(y_test, y_test_pred), plot=False)
            live.log_metric("test/f1", f1_score(y_test, y_test_pred, average="weighted"), plot=False)
            live.log_metric("ROC_AUC", roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]), plot=False)

            live.log_sklearn_plot(
                "confusion_matrix", y_test, y_test_pred, name="test/confusion_matrix",
                title="Test Confusion Matrix")

            model_path = Path(f'data/trained_models/random_forest_{n_estimators}.pkl')
            pickle.dump(rf, open(model_path, 'wb'))
            live.log_artifact(model_path)


if __name__ == '__main__':
    df_path = Path('data/datasets/prepared_one_bank.parquet')
    df = pd.read_parquet(df_path)
    df.drop('position', axis=1, inplace=True)

    train_predict_rf(df=df.copy())
