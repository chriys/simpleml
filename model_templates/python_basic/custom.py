from typing import Any, Dict

import pandas as pd


def init(**kwargs):
    return


def read_input_data(binary_data: bytes) -> Any:
    return [1, 2]


def load_model(code_dir: str) -> Any:
    return "fake_model"


def transform(data, model: Any = None) -> pd.DataFrame:
    return pd.DataFrame([42 for _ in data])


def score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame:
    random_n = 42
    preds = pd.DataFrame([random_n for _ in range(data.shape[0])], columns=["predictions"])
    return preds
