from decouple import config
from os import path


dvc_params_abs_path = path.abspath(path.join(__file__, '../params.yaml'))

DVC_PARAMS_FILE = config("DVC_PARAMS_FILE", default=dvc_params_abs_path, cast=str)
RESULTS_FOLDER = config("RESULTS_FOLDER", default=path.abspath(path.join(__file__, '../results')), cast=str)
DATA_FOLDER = config("DATA_FOLDER", default=path.abspath(path.join(__file__, '../data')), cast=str)
MODELS_FOLDER = config("MODELS_FOLDER", default=path.abspath(path.join(__file__, '../data/trained_models')), cast=str)
DATASETS_FOLDER = config("DATASETS_FOLDER", default=path.abspath(path.join(__file__, '../data/datasets')), cast=str)
