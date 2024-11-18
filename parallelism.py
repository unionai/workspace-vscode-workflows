from dataclasses import dataclass, asdict
from functools import partial
from typing import List

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


# 📦 Define Hyperparameters dataclass to contain hyperparameters and the dataset
@dataclass
class Hyperparameters:
    C: float
    max_iter: int = 1000


@fl.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    return load_wine(as_frame=True).frame


@fl.task(container_image=image)
def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """Simplify the task from a 3-class to a binary classification problem."""
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


# Update the train_model task to:
# 1. ✨ Accept a single argument of type TrainArgs
# 2. 💪 Use additional CPUs and memory
@fl.task(
    container_image=image,
    cache=True,
    cache_version="1",
    retries=3,
    requests=fl.Resources(cpu="2", mem="1Gi"),
)
def train_model(hyperparameters: Hyperparameters, data: pd.DataFrame) -> LogisticRegression:
    """Train a model on the wine dataset."""

    # open StructuredDataset as a pandas DataFrame 🐼
    features = data.drop("target", axis="columns")
    target = data["target"]
    return LogisticRegression(**asdict(hyperparameters)).fit(features, target)


@fl.workflow
def training_workflow(hp_grid: List[Hyperparameters]) -> List[LogisticRegression]:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    processed_data = process_data(data=data)
    # 🎁 wrap the train_model task in map_task, with a concurrency of
    # 5 executions at any given time.
    partial_train_model = partial(train_model, data=processed_data)
    return fl.map_task(partial_train_model, concurrency=5)(hp_grid)


if __name__ == "__main__":
    hp_grid = [{"C": x, "max_iter": 1000} for x in (0.1, 0.01, 0.001)]
    print(
        "Running training_workflow() "
        f"{training_workflow(hp_grid=hp_grid)}"
    )
