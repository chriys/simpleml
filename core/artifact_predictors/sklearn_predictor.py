from typing import Any

import pickle
import pandas as pd

from core.enums import framework_deps, SupportedFrameworks, SupportedArtifacts
from core.artifact_predictors.abstract_predictor import AbstractPredictor


class SKLearnPredictor(AbstractPredictor):
    def __init__(self):
        super(SKLearnPredictor, self).__init__(
            SupportedFrameworks.SKLEARN, SupportedArtifacts.PKL_EXTENSION
        )


    def is_framework_present(self) -> bool:
        try:
            from sklearn.base import BaseEstimator
            return True
        except ModuleNotFoundError:
            return False


    def can_load_artifact(self, artifact_path: str) -> bool:
        return self.is_artifact_supported(artifact_path)


    def can_use_model(self, model: Any) -> bool:
        if not self.is_framework_present():
            return False

        from sklearn.base import BaseEstimator

        if isinstance(model, BaseEstimator):
            return True

        return False


    def load_model_from_artifact(self, artifact_path: str) -> Any:
        with open(artifact_path, "rb") as pickle_file:
            try:
                model = pickle.load(pickle_file)
            except TypeError as exc:
                raise ImportError(f"Unable to unpickle the provided artifact {artifact_path}")

            return model


    def framework_requirements(self) -> list:
        return framework_deps[SupportedFrameworks.SKLEARN]


    def predict(self, data: pd.DataFrame, model: Any, **kwargs):
        # run predict from parent to make certain predicate are met.
        super(SKLearnPredictor, self).predict(data, model, **kwargs)

        labels_to_use = None
        if self.target_type.is_classification():
            if hasattr(model, "classes_"):
                labels_to_use = list(model.classes_)
            preds = model.predict_proba(data)
            preds = pd.DataFrame(preds)
        elif self.target_type.is_regression_or_anomaly():
            preds = model.predict(data)
        else:
            raise ValueError(
                f"Target type {self.target_type.value} is not supported by {self.__class__.__name__} predictor"
            )

        return preds, labels_to_use
