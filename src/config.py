"""
Project configuration
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """
    Configuration class
    """

    data_dir: Path = DATA_DIR
    train_dir: Path = TRAIN_DIR
    test_dir: Path = TEST_DIR
    weights_dir: Path = WEIGHTS_DIR
    outputs_dir: Path = OUTPUTS_DIR

    image_size: int = IMAGE_SIZE

    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    epochs: int = EPOCHS
    validation_split: float = VALIDATION_SPLIT

    random_seed: int = RANDOM_SEED

    num_workers: int = NUM_WORKERS
    pin_memory: bool = PIN_MEMORY

    """
# Paths
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

WEIGHTS_DIR = Path("weights")
OUTPUTS_DIR = Path("outputs")

# Dataset
IMAGE_SIZE = 100

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
VALIDATION_SPLIT = 0.2

# Reproducibility
RANDOM_SEED = (42)

# Dataloader
NUM_WORKERS = 2
PIN_MEMORY = False

"""
