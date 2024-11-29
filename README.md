# Simpleml
---
## Installation
1. Git clone the current repo. The below commands will be executed from the repo's folder.
2. Create and active your virtual environement
    ```
    python3 -m venv env
    source venv/bin/activate
    ```
3. Install pip dependencies
    ```
    pip install -r requirements.txt
    ```
3. Create and `.env` file. You can copy `.example-env` and rename it to `.env`


## Guardrails for Data Scientists
There are guardrails put in place to make the work of data scientist more easier
and minimize errors during training and testing phase.

1. Dedicated folder for your model and data.

    For every model you're working on, you have the freedom to create it's folder structure inside of **/model_templates**. 
    In your model's folder every is possible. Save your artifacts, data and custom code in the same place.

1. Model hooks.

    Create a `custom.py` file and define a set of hooks that the library will know how to call to interact with your model and your data.

1. Automatic Artifact discovery.

1. Already defined API endpoints to interact with your model and artifacts.
1. CLI tool. (WIP)
1. Environment file to define your custom environment variables.
1. Available Docker image. (WIP)
1. Library as a package. (WIP)


## Make Predictions

### Create Your Model Template

### Data format
When working with structured models, the supported data as files is `csv`.
We do not perform any sanitation and fixing missing or malformed column names.

### Available Model Hooks
Custom hooks are methods you can define inside a file called `custom.py` to interact with your data and/or your model.

Custom hooks only supports Python, although R models can be added in the future.
See the accepted method below to have an idea of how to use them.

- `init(**kwargs) -> None`
    - Executed once at the beginning of the run. Can be used to initiliaze the environment before the model is called.
    - `kwargs` - additional arguments passed to the method.
        - i.e `code_dir` - code folder passed in by `CODE_DIR` environment var.
- `read_input_data(binary_data: bytes) -> Any`
    - This hook can be used to define how you read data, e.g: encoding, handle missing values.
    - If you return a non `pd.DataFrame`, make sure to implement your own score method.
- `load_model(code_dir: str) -> Any`
- `transform(data: pd.DataFrame, model: Any) -> pd.DataFrame`
- `score(data: pd.DataFrame, model: Any, **kwargs: Dict[str, Any]) -> pd.DataFrame`


## API Endpoints


## CLI Tool


## Supported models

## Supported Artifacts

Even if you do not define a `custom.py` file, the library will discover artifacts it knows how to use to score your data.

You only need to set `CODE_DIR` to be your model's directory and make sure your artifact is in there.

    i.e.: You're working on a model and have create a model template containing your artifact.
    simpleml
    |   app.py
    |   README.md
    |---core
    |   ..
    |---model_templates
    |------my_super_model
           |   my_keras_model.h5

    Your `.env` file should have this entry `CODE_DIR=model_templates/my_super_model/`.


If needed, you can override how a `.pkl` model scores data by provided a custom hook like `score()` or `transform()`.

Below is a list of supported model artifacts:

- `.pkl`

