from pathlib import Path
from typing import Any

from box import ConfigBox
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split

from settings import DVC_PARAMS_FILE

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open(Path(DVC_PARAMS_FILE))))

RANDOM_SEED = params.base.random_seed
TEST_SIZE = params.base.test_size


def data_split(X: Any, y: Any, test_size: float = TEST_SIZE, random_state: int = RANDOM_SEED):
    """
    Split the data into training and testing sets.

    Parameters:
    X (Any): The input features.
    y (Any): The target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed used by the random number generator.

    Returns:
    X_train (Any): The training set of input features.
    X_test (Any): The testing set of input features.
    y_train (Any): The training set of target variable.
    y_test (Any): The testing set of target variable.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
