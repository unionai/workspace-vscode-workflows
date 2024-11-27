import flytekit as fl
import pandas as pd
from sklearn.linear_model import LogisticRegression


image = fl.ImageSpec(packages=["pandas", "pyarrow", "scikit-learn"])


@fl.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    print("Getting data")
    data = load_wine(as_frame=True).frame
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@fl.task(container_image=image)
def train_model(data: pd.DataFrame) -> LogisticRegression:
    """Train a model on the wine dataset."""
    print("Training model")
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression().fit(features, target)


@fl.workflow
def training_workflow() -> LogisticRegression:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    return train_model(data)


if __name__ == "__main__":
    # You can run this script with pre-defined arguments with `python union_workflow.py`
    # but we recommend running it with the `union run` CLI command, as you'll see in
    # the next step of this walkthrough.
    print(f"Running training_workflow() {training_workflow()}")
