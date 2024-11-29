import os

from typing import Annotated, Optional, List

from fastapi import (
    FastAPI,
    Depends,
    Body,
    File,
    UploadFile,
    Form,
    HTTPException,
)
from dotenv import load_dotenv
from pydantic import BaseModel

from core.utils import check_folder_exists
from core.enums import (
    TargetType,
    TARGET_TYPE_ARG_NAME,
    POS_CLASS_LABEL_ARG_NAME,
    NEG_CLASS_LABEL_ARG_NAME,
    CLASS_LABELS_ARG_NAME,
)
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


async def common_predict_params(
    target_type: Annotated[TargetType, Form()],
    input: Annotated[Input | None, Body(embed=True)] = None,
    input_file: Annotated[UploadFile | None, File()] = None,
    positive_class_label: Annotated[Optional[str], Form()] = None,
    negative_class_label: Annotated[Optional[str], Form()] = None,
    class_labels: Annotated[List[str] | None, Form()] = None,
    output_destination: Annotated[Optional[str], Form()] = None
):
    return {
        TARGET_TYPE_ARG_NAME: target_type,
        POS_CLASS_LABEL_ARG_NAME: positive_class_label,
        NEG_CLASS_LABEL_ARG_NAME: negative_class_label,
        CLASS_LABELS_ARG_NAME: class_labels,
        "input": input,
        "input_file": input_file,
        "output_destination": output_destination,
    }


commons_predict_dep = Annotated[dict, Depends(common_predict_params)]


def init_predictor(params):
    code_dir = os.environ['CODE_DIR']

    if not check_folder_exists(code_dir):
        raise HTTPException(status_code=404, detail=f"The following code_dir {code_dir} cannot be found")

    predictor = PythonPredictor()
    params["code_dir"] = code_dir
    predictor.configure(params)

    return predictor


@app.get("/")
def main():
    return {"Hello": "World"}


@app.post("/predict", response_model=PredictResponse)
def predict(commons: commons_predict_dep):
    predictor = init_predictor(commons)

    return predictor.predict(
        target_type=commons.get("target_type"),
        binary_data=commons.get("input"),
        mimetype="application/json",
    )


@app.post("/predict-file", response_model=PredictResponse)
async def predict_file(commons: commons_predict_dep):
    predictor = init_predictor(commons)

    return predictor.predict(
        target_type=commons.get("target_type"),
        binary_data=await commons.get("input_file").read(),
        mimetype=commons.get("input_file").content_type,
    )


@app.post("/batch-predict")
async def batch_predict(commons: commons_predict_dep):
    # TODO: save predictions to file
    predictor = init_predictor(commons)

    return predictor.predict(
        target_type=commons.get("target_type"),
        binary_data=await commons.get("input_file").read(),
        mimetype=commons.get("input_file").content_type,
        output_destination=commons.get("output_destination"),
    )


@app.post("/transform")
def transform():
    pass
