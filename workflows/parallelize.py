from dataclasses import dataclass, asdict
from functools import partial

import union
import pandas as pd
from sklearn.neural_network import MLPClassifier


image = union.ImageSpec(packages=["pandas", "pyarrow", "scikit-learn"])


@dataclass
class Hyperparameters:
    hidden_layer_sizes: list[int]


@union.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    print("Getting data")
    data = load_wine(as_frame=True).frame
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@union.task(container_image=image)
def train_model(hyperparameters: Hyperparameters, data: pd.DataFrame) -> MLPClassifier:
    """Train a model on the wine dataset."""
    print("Training model")
    features = data.drop("target", axis="columns")
    target = data["target"]
    return MLPClassifier(**asdict(hyperparameters)).fit(features, target)


@union.workflow
def training_workflow(hp_grid: list[Hyperparameters]) -> list[MLPClassifier]:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    # 🎁 wrap the train_model task in map_task, with a concurrency of
    # 5 executions at any given time.
    partial_train_model = partial(train_model, data=data)
    return union.map_task(partial_train_model, concurrency=5)(hp_grid)
