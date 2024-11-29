import os
from enum import Enum


DEBUG = os.environ.get("DEBUG")

LOGGER_NAME_PREFIX = "simpleml"

CUSTOM_FILE_NAME = "custom"

TARGET_TYPE_ARG_NAME = "target_type"

CLASS_LABELS_ARG_NAME = "class_labels"

POS_CLASS_LABEL_ARG_NAME = "positive_class_label"

NEG_CLASS_LABEL_ARG_NAME = "negative_class_label"


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

    def is_classification(self) -> bool:
        return self in [self.BINARY, self.MULTICLASS]

    def is_regression_or_anomaly(self) -> bool:
        return self in [self.REGRESSION, self.ANOMALY]


class SupportedFrameworks:
    SKLEARN = "scikit-learn"


class SupportedArtifacts:
    PKL_EXTENSION = ".pkl"


framework_deps = {
    SupportedFrameworks.SKLEARN: ["scikit-learn", "scipy", "numpy"]
}
