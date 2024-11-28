import sys
import logging
from pathlib import Path
from typing import Any, NoReturn

import pandas as pd

from core.enums import (
    LOGGER_NAME_PREFIX,
    CUSTOM_FILE_NAME,
    CustomHooks,
    TargetType,
)
from core.utils import get_fullpath


class ModelAdapterError(Exception):
    """
    Raised when error occurs in ModelAdapter
    """


class ModelAdapter():
    def __init__(
        self,
        code_dir: str,
        target_type: str = None
    ):
        self.code_dir = code_dir
        self._target_type = target_type
        self._predictor = None
        self._hooks = {hook: None for hook in CustomHooks.ALL_PREDICT}
        self._logger = logging.getLogger(LOGGER_NAME_PREFIX + "." + self.__class__.__name__)


    def predict(self, model: Any = None, **kwargs):
        data = self.load_data(
            binary_data=kwargs.get("binary_data"),
            mimetype=kwargs.get("mimetype")
        )
        data = self.preprocess(data, model)
        return self._predict(data, model, **kwargs)


    def load_data(self, binary_data, mimetype):
        if self.has_custom_hook(CustomHooks.READ_INPUT_DATA):
            try:
                data = self._hooks[CustomHooks.READ_INPUT_DATA](binary_data)
            except Exception as exc:
                self._log_and_raise_error(exc, "Failed to read input data using 'read_input_data' hook.")
        else:
            data = self._read_structured_input_data_df(binary_data, mimetype)

        return data


    def _read_structured_input_data_df(self, binary_data, mimetype) -> pd.DataFrame:
        # convert json data into df
        return


    def preprocess(self, data, model: Any = None) -> pd.DataFrame:
        if self.has_custom_hook(CustomHooks.TRANSFORM):
            try:
                output = self._hooks[CustomHooks.TRANSFORM](data, model)
            except Exception as exc:
                self._log_and_raise_error(exc, "The 'transform hook' has failed to transform the dataset")

            self._validate_data(output, CustomHooks.TRANSFORM)
        else:
            output = data

        return output


    def _validate_data(self, to_validate, hook) -> NoReturn:
        if hook in {CustomHooks.SCORE, CustomHooks.TRANSFORM}:
            if not isinstance(to_validate, pd.DataFrame):
                raise ValueError(f"{hook} must return a pandas dataframe, but received {type(to_validate)}")

        if len(to_validate.shape) != 2:
            raise ValueError(f"{hook} must return dataframe with 2 dimensions, but received one with dims {to_validate.shape}")


    def _predict(self, data, model, **kwargs):
        if self.has_custom_hook(CustomHooks.SCORE):
            try:
                preds_df = self._hooks.get(CustomHooks.SCORE)(data, model, **kwargs)
            except Exception as exc:
                self._log_and_raise_error(exc, "Model 'score' hook failed to make predictions.")
        else:
            try:
                preds_df = self._predictor_to_use.predict(data, model, **kwargs)
            except Exception as exc:
                self._log_and_raise_error(exc, "Failed to make predictions.")

        return preds_df


    def load_custom_hooks(self):
        code_dir = get_fullpath(self.code_dir)
        custom_files = list(Path(code_dir).rglob(f"{CUSTOM_FILE_NAME}.py"))
        if len(custom_files) > 1:
            raise RuntimeError(f"Found more than 1 custom hook files: {custom_files}")

        if len(custom_files) == 0:
            self._logger.info(f"No {CUSTOM_FILE_NAME}.py detected in {self.code_dir}")
            return

        custom_file_path = custom_files[0].parent
        self._logger.info(f"Detected {custom_file_path}... loading hooks")
        sys.path.insert(0, str(custom_file_path))

        try:
            custom_module = __import__(CUSTOM_FILE_NAME)
            self._load_custom_hooks(custom_module)
        except ImportError as e:
            self._log_and_raise_error(e, f"Failed to load hooks from [{custom_file_path}]")


    def _load_custom_hooks(self, custom_module):
        for hook in CustomHooks.ALL_PREDICT:
            self._hooks[hook] = getattr(custom_module, hook, None)

        # Run init hook if found
        if self.has_custom_hook(CustomHooks.INIT):
            self._hooks[CustomHooks.INIT](code_dir=self.code_dir)

        self._logger.debug(f"Hooks loaded: {self._hooks}")


    def load_model_from_artifact(self):
        if self.has_custom_hook(CustomHooks.INIT):
            self._model = self._load_model_via_hook()
        else:
            model_artifact_file = self._detect_model_artifact_file()
            self._model = self._load_model_via_predictors(model_artifact_file)

        # If no score hook is given, find a predictor that can handle the mdel
        if (
            self._target_type not in [TargetType.TRANSFORM, TargetType.UNSTRUCTURED]
            and not self.has_custom_hook(CustomHooks.SCORE)
        ):
            self._find_predictor_to_use()

        if (
            self._target_type == TargetType.TRANSFORM
            and not self.has_custom_hook(CustomHooks.SCORE)
        ):
            raise ModelAdapter("A transform task requires a user-defined hook to run transformations.")

        return self._model


    def _load_model_via_hook(self):
        pass


    def _detect_model_artifact_file(self):
        pass


    def _load_model_via_predictors(self):
        pass


    def _find_predictor_use(self):
        pass


    def has_custom_hook(self, hook_type: CustomHooks) -> bool:
        return self._hooks.get(hook_type, None) is not None


    def _log_and_raise_error(self, exc: Exception, msg: str) -> NoReturn:
        self._logger.exception(f"{msg} Exception: {exc!r}")
        raise ModelAdapterError(f"{msg} Exception: {exc!r}")
