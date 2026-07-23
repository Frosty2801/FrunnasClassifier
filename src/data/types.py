
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torch.utils.data import Dataset


@dataclass(slots=True)
class Datasets:
    train: Dataset
    val: Dataset
    test: Dataset

    classes: list[str]


@dataclass(slots=True)
class Dataloaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader

    classes: list[str]

