from decouple import config
from os import path


dvc_params_abs_path = path.abspath(path.join(__file__, '../params.yaml'))

DVC_PARAMS_FILE = config("DVC_PARAMS_FILE", default=dvc_params_abs_path, cast=str)
