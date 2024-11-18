from dataclasses import dataclass, asdict
from typing import Annotated

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


# 🍷 Create a type for the wine dataset
WineDataset = Annotated[
    fl.StructuredDataset,
    fl.kwtypes(
        alcohol=float,
        malic_acid=float,
        ash=float,
        alcalinity_of_ash=float,
        magnesium=float,
        total_phenols=float,
        flavanoids=float,
        nonflavanoid_phenols=float,
        proanthocyanins=float,
        color_intensity=float,
        hue=float,
        od280_od315_of_diluted_wines=float,
        proline=float,
        target=int,
    )
]

# 📦 Define Hyperparameters dataclass to contain hyperparameters and the dataset
@dataclass
class Hyperparameters:
    C: float
    max_iter: int


@fl.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).frame.rename(columns=lambda x: x.replace("/", "_"))


@fl.task(container_image=image)
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Simplify the task from a 3-class to a binary classification problem."""
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


# Update the train_model task:
# 1. 🎒 Set `cache=True` with a `cache_version` string to prevent re-running the function
#    given the same inputs.
# 2. 🔄 Add `retries=3` to retry the task in the event of system-level failures.
# 3. 🍷 Annotate the `data` argument with the `WineDataset` type that we defined above.
# 4. 📄 Add a `hyperparameters: dict` input to experiment with different model settings.
@fl.task(cache=True, cache_version="1", retries=3, container_image=image)
def train_model(hyperparameters: Hyperparameters, data: WineDataset) -> LogisticRegression:
    """Train a model on the wine dataset."""

    # open StructuredDataset as a pandas DataFrame 🐼
    data: pd.DataFrame = data.open(pd.DataFrame).all()
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(**asdict(hyperparameters)).fit(features, target)


# 📄 Add a `hyperparameters: dict` input to the workflow to feed into `train_model`
@fl.workflow
def training_workflow(hyperparameters: Hyperparameters) -> LogisticRegression:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    processed_data = process_data(data)
    return train_model(hyperparameters, processed_data)


if __name__ == "__main__":
    print(
        "Running training_workflow() "
        f"{training_workflow(hyperparameters={'max_iter': 1000})}"
    )
