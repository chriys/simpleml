# Simpleml
> :warning: **If you are cloning this repo**: Keep in mind this library is a Work-In-Progress. Some functionalities are yet to be fully implemented.

## Content
1. [Installation](#installation)
1. [Guardrails for Data Scientists](#guardrails-for-data-scientists)
1. [Make Predictions](#make-predictions)
1. [Create a custom model template](#create-a-custom-model-template)
    1. [Data format](#data-format)
    1. [Available Model Hooks](#available-model-hooks)
1. [API Endpoints](#api-endpoints)
1. [CLI Tool](#cli-tool)
1. [Supported models](#supported-models)
1. [Supported Artifacts](#supported-artifacts)


## Installation
1. Git clone the current repo `git clone https://github.com/chriys/simpleml.git`.
1. Inside the cloned repo, create and active your virtual environement
    ```
    python3 -m venv env
    source venv/bin/activate
    ```
1. Install pip dependencies
    ```
    pip install -r requirements.txt
    ```
1. Create and `.env` file. You can copy `.example-env` and rename it to `.env`
1. Run `fastapi dev app.py` to start the server locally


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
There are essentially to way to make predictions:
- Using the API endpoints
- Using the CLI tool (Work In Progress).
For each one of the above steps, you need to create a **model template**.

## Create a custom model template
1. Inside `/model_templates`, create a folder where you will place your custom code and files.
1. Inside your model's folder, create a `custom.py` file and add the hook you'll need the library to call.
    - You can ommit this step if your serialized model extension is supported and if you don't need additionnal hooks.
1. Set the `CODE_DIR` vars inside `.env`.
    - If your model has a similar structure as below, set the variable as follow: `CODE_DIR=model_templates/my_super_model`
    - `CODE_DIR` also supports absolute paths. (i.e: `CODE_DIR=/Users/me/simpleml/model_templates/my_super_model`)

```
    Below is what the folder structure can look like.
    ...
    |
    |-- model_templates
    |   |-- README.md
    |   |-- ..
    `----- my_super_model
           |-- keras_model.h5
           |-- custom.py
           `---data
               |-- iris.csv
               `-- abc.csv
```

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
- **SCIKIT-LEARN**


## Supported Artifacts

Even if you do not define a `custom.py` file, the library will discover artifacts it knows how to use to score your data.

You only need to set `CODE_DIR` found in your `.env` to be your model's directory and make sure your artifact is in there.

If needed, you can override how a serialized model scores data by providing a custom hook like `score()` or `transform()`.

Below is a list of supported model artifacts:

- `.pkl`

