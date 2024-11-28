import os
from enum import Enum


DEBUG = os.environ.get("DEBUG")

LOGGER_NAME_PREFIX = "simpleml"

CUSTOM_FILE_NAME = "custom"


class CustomHooks:
    INIT = "init"
    READ_INPUT_DATA = "read_input_data"
    LOAD_MODEL = "load_model"
    TRANSFORM = "transform"
    FIT = "fit"
    SCORE = "score"

    ALL_PREDICT = [
        INIT,
        READ_INPUT_DATA,
        LOAD_MODEL,
        TRANSFORM,
        SCORE,
    ]


class TargetType(str, Enum):
    BINARY = "binary"
    REGRESSION = "regression"
    ANOMALY = "anomaly"
    UNSTRUCTURED = "unstructured"
    MULTICLASS = "multiclass"
    TRANSFORM = "transform"
