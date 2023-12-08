from decouple import config
from os import path


dvc_params_abs_path = path.abspath(path.join(__file__, '../params.yaml'))
dvc_yaml_abs_path = path.abspath(path.join(__file__, '../dvc.yaml'))
data_folder_abs_path = path.abspath(path.join(__file__, '../data'))
models_folder_abs_path = path.abspath(path.join(data_folder_abs_path, 'trained_models'))
datasets_folder_abs_path = path.abspath(path.join(data_folder_abs_path, 'datasets'))

DVC_PARAMS_FILE = config("DVC_PARAMS_FILE", default=dvc_params_abs_path, cast=str)
DVC_YAML_FILE = config("DVC_YAML_FILE", default=dvc_yaml_abs_path, cast=str)
RESULTS_FOLDER = config("RESULTS_FOLDER", default=path.abspath(path.join(__file__, '../results')), cast=str)
DATA_FOLDER = config("DATA_FOLDER", default=data_folder_abs_path, cast=str)
MODELS_FOLDER = config("MODELS_FOLDER", default=models_folder_abs_path, cast=str)
DATASETS_FOLDER = config("DATASETS_FOLDER", default=datasets_folder_abs_path, cast=str)
