import os

from typing import Annotated

from fastapi import FastAPI, Body, HTTPException
from dotenv import load_dotenv
from pydantic import BaseModel

from core.utils import check_folder_exists
from core.enums import TargetType
from core.python_predictor import PredictResponse, PythonPredictor


load_dotenv()

if "CODE_DIR" not in os.environ:
  raise RuntimeError("CODE_DIR not defined in environment variables.")

app = FastAPI()


class Input(BaseModel):
    """
    json schema of the input data to be scored by the /predict endpoint.

    Schema
    ------
    index: the list of indexes
    columns: the list of column (feature) names
    data: the list of data to be scored
    """
    index: list
    columns: list
    data: list


@app.get("/")
def main():
    return {"Hello": "World"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    input: Annotated[Input, Body(embed=True)],
    target_type: Annotated[TargetType, Body(embed=True)] = TargetType.BINARY
):
    code_dir = os.environ['CODE_DIR']

    if not check_folder_exists(code_dir):
        raise HTTPException(status_code=404, detail=f"The following code_dir {code_dir} cannot be found")

    predictor = PythonPredictor()
    predictor.configure({"code_dir": code_dir, "target_type": target_type})

    return predictor.predict(
        binary_data=input,
        mimetype="application/json"
    )


@app.post("/batch-predict")
def batch_predict():
    batch_preds = {}
    return batch_preds


@app.post("/transform")
def transform():
    pass
