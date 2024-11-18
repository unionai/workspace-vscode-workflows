import flytekit as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression



image = fl.ImageSpec(
    name="union-workspace",
    packages=[
        "pandas",
        "pyarrow",
        "plotly",
        "scikit-learn",
        "matplotlib",
    ],
)


# 🧱 @task decorators define the building blocks of your pipeline
@fl.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).frame


@fl.task(container_image=image)
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Simplify the task from a 3-class to a binary classification problem."""
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@fl.task(container_image=image)
def train_model(data: pd.DataFrame) -> LogisticRegression:
    """Train a model on the wine dataset."""
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(max_iter=1000).fit(features, target)


# 🔀 @workflows decorators define the flow of data through the tasks
@fl.workflow
def training_workflow() -> LogisticRegression:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    processed_data = process_data(data)
    return train_model(processed_data)


if __name__ == "__main__":
    # You can run this script with pre-defined arguments with `python union_workflow.py`
    # but we recommend running it with the `union run` CLI command, as you'll see in
    # the next step of this walkthrough.
    print(f"Running training_workflow() {training_workflow()}")
