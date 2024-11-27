import union
import pandas as pd
from sklearn.neural_network import MLPClassifier


image = union.ImageSpec(packages=["pandas", "pyarrow", "scikit-learn"])


@union.task(container_image=image)
def get_data() -> pd.DataFrame:
    """Get the wine dataset."""
    from sklearn.datasets import load_wine

    print("Getting data")
    data = load_wine(as_frame=True).frame
    return data.assign(target=lambda x: x["target"].where(x["target"] == 0, 1))


@union.task(container_image=image)
def train_model(data: pd.DataFrame) -> MLPClassifier:
    """Train a model on the wine dataset."""
    print("Training model")
    features = data.drop("target", axis="columns")
    target = data["target"]
    return MLPClassifier().fit(features, target)


@union.workflow
def training_workflow() -> MLPClassifier:
    """Put all of the steps together into a single workflow."""
    data = get_data()
    return train_model(data)
