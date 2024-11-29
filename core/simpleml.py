import io
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
from core.artifact_predictors.sklearn_predictor import SKLearnPredictor


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
        self._predictor_to_use = None
        self._hooks = {hook: None for hook in CustomHooks.ALL_PREDICT}

        self._artifact_predictors = [
            SKLearnPredictor(),
        ]

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
        # FIXME: handle mimetype
        try:
            df = pd.read_csv(io.BytesIO(binary_data))
        except UnicodeDecodeError:
            self._logger.error("A non UTF-8 encoding was encountered when opening the data.")
            raise ModelAdapterError("Supplied CSV input file encoding must be UTF-8.")

        return df


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
                preds_df, model_labels = self._predictor_to_use.predict(data, model, **kwargs)
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

        if (self._no_hook_to_run_score()):
            self._find_predictor_to_use()

        if (self._no_hook_to_run_transform()):
            raise ModelAdapter("A transform task requires a user-defined hook to run transformations.")

        return self._model


    def _no_hook_to_run_score(self):
        return self._target_type not in [TargetType.TRANSFORM, TargetType.UNSTRUCTURED] and not self.has_custom_hook(CustomHooks.SCORE)


    def _no_hook_to_run_transform(self) -> bool:
        return self._target_type == TargetType.TRANSFORM and not self.has_custom_hook(CustomHooks.TRANSFORM)


    def _load_model_via_hook(self):
        pass


    def _detect_model_artifact_file(self):
        supported_extensions = [p.artifact_extension.lower() for p in self._artifact_predictors]
        artifact_file = None
        files = get_fullpath(self.code_dir).rglob("*")

        for file in files:
            # check if file has the supported extensions
            if file.suffix.lower() in supported_extensions:
                if artifact_file:
                    raise ModelAdapterError(
                        "Multiple serialized model files have been found. Remove additional artifacts"
                        "or define custom.load_model()\n"
                        f"Retrieved artifacts are {files}"
                    )
                artifact_file = str(file)

        if not artifact_file:
            raise ModelAdapterError(
                f"Couldn't find any model artifact file in: {self.code_dir} supported by default predictors.\n"
                f"Only the following files extensions are supported {supported_extensions}"
            )

        self._logger.debug(f"model_artifact_file: {artifact_file}")
        return artifact_file


    def _load_model_via_predictors(self, model_artifact_file):
        model = None
        predictors_for_artifact = []

        for pred in self._artifact_predictors:
            if pred.is_artifact_supported(model_artifact_file):
                predictors_for_artifact.append(pred)

            if pred.can_load_artifact(model_artifact_file):
                try:
                    model = pred.load_model_from_artifact(model_artifact_file)
                except Exception as exc:
                    self._log_and_raise_error(exc, "Could not load model from artifact file.")
                # stop the loop because model was loaded
                break

        if model is None:
            if len(predictors_for_artifact) > 0:
                framework_err = f"""
                The following frameworks support the loaded model artifact: {model_artifact_file}"
                but no model could be loaded. Check if requirements are missing."
                """
                for pred in predictors_for_artifact:
                    framework_err += f"Framework: {pred.name}, requirements: {pred.framework_requirements()}"

                raise ModelAdapterError(framework_err)
            else:
                raise ModelAdapterError(
                    f"Could not load model from artifact file {model_artifact_file}"
                )

        self._model = model
        return model


    def _find_predictor_to_use(self) -> bool:
        self._predictor_to_use = None

        for pred in self._artifact_predictors:
            if pred.can_use_model(self._model):
                self._predictor_to_use = pred
                break

        if self._no_predictor_to_use_found():
            raise ModelAdapter(f"Could not find a framework to handle the loaded model and no **{CustomHooks.SCORE}** hook is provided in custom.py")

        self._logger.debug(f"Predictor to use: {self._predictor_to_use.name}")


    def _no_predictor_to_use_found(self) -> bool:
        return not self._predictor_to_use and not self._hooks[CustomHooks.SCORE]


    def has_custom_hook(self, hook_type: CustomHooks) -> bool:
        return self._hooks.get(hook_type, None) is not None


    def _log_and_raise_error(self, exc: Exception, msg: str) -> NoReturn:
        self._logger.exception(f"{msg} Exception: {exc!r}")
        raise ModelAdapterError(f"{msg} Exception: {exc!r}")
