import logging
from typing import Any, Optional
from pathlib import Path
from abc import ABC, abstractmethod

from core.utils import get_fullpath
from core.enums import (
    TargetType,
    CLASS_LABELS_ARG_NAME,
    TARGET_TYPE_ARG_NAME,
    POS_CLASS_LABEL_ARG_NAME,
    NEG_CLASS_LABEL_ARG_NAME,
    LOGGER_NAME_PREFIX,
)


class AbstractPredictor(ABC):
    def __init__(
        self,
        name: str,
        extension: str
    ):
        self._name = name
        self._artifact_extension = extension
        self.pos_class_label = None
        self.neg_class_label = None
        self.class_labels = None
        self.target_type: Optional[TargetType] = None
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)


    @property
    def name(self):
        return self._name


    @property
    def artifact_extension(self):
        return self._artifact_extension


    def is_artifact_supported(self, artifact_path) -> bool:
        artifact_path = get_fullpath(artifact_path)
        return artifact_path.suffix.lower() == self._artifact_extension.lower()


    @abstractmethod
    def is_framework_present(self) -> bool:
        """Check if the framework can be loaded"""
        pass


    @abstractmethod
    def can_load_artifact(self, artifact_path) -> bool:
        """Check if the model artifact can be loaded"""
        pass


    @abstractmethod
    def can_use_model(self, model) -> bool:
        """Given a model object, can this predictor use the given model"""
        pass


    @abstractmethod
    def load_model_from_artifact(self, artifact_path) -> Any:
        pass


    @abstractmethod
    def framework_requirements(self) -> list:
        """Return a list of the framework python requirements"""
        pass


    @abstractmethod
    def predict(self, data, model, **kwargs):
        """
        This method will be called to do parameter validation before predict() is called in the subclass.
        """
        self.target_type = kwargs.get(TARGET_TYPE_ARG_NAME)
        self.class_labels = kwargs.get(CLASS_LABELS_ARG_NAME)

        if self.target_type == TargetType.MULTICLASS and not self.class_labels:
            raise ValueError(
                f"For `{self.target_type.value}` target type, class labels must be provied. Found: {self.class_labels}"
            )

        if self.target_type == TargetType.BINARY:
            # set the class labels for binary classification
            self.class_labels = [
                kwargs.get(NEG_CLASS_LABEL_ARG_NAME),
                kwargs.get(POS_CLASS_LABEL_ARG_NAME),
            ]
            if self.class_labels:
                raise ValueError(
                    f"For `{self.target_type.value}` target type the positive and negative class labels must be provied. Found: {self.class_labels}"
                )
