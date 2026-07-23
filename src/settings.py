"""
Project configuration
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass(frozen=True)
class DatasetConfig:
    data_dir: Path = Path("data")

    image_size: int = 100

    validation_split: float = 0.2

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32

    learning_rate: float = 1e-3

    epochs: int = 20

    random_seed: int = 42
    

@dataclass(frozen=True)
class DataloaderConfig:
    num_workers: int = 2

    pin_memory: bool = False


@dataclass(frozen=True)
class PathsConfig:
    weights_dir: Path = Path("weights")
    outputs_dir: Path = Path("outputs")


@dataclass(frozen=True)
class Settings:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

settings = Settings()