"""
Dataset and Dataloader
"""

import torch
from pathlib import Path

from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.settings import settings

from src.data.transforms import (
    get_train_transforms,
    get_test_transforms,
)

from src.data.types import Datasets

def _load_image_folder(
    root: Path,
    transform: transforms.Compose,
) -> ImageFolder:
    """
    Load image folder dataset from directory

    Args:
        root (Path): Root directory of the dataset.
        transform (transforms.Compose): Transformations to apply to the images.
    Returns:
        ImageFolder: The loaded dataset.
    """

    return ImageFolder(
        root=root,
        transform=transform,
)

def _split_indices(
        dataset: Dataset,
) -> tuple[list[int], list[int]]:
    """
    Split dataset into training and validation subsets

    Args:
        dataset (Dataset): The dataset to split.
    Returns:
        tuple[list[int], list[int]]: The indices for the training and validation subsets.
    """

    train_size = int(
        (1 - settings.dataset.validation_split)
        * len(dataset)
    )

    val_size = len(dataset) - train_size

    generator_size = torch.Generator().manual_seed(
        settings.training.random_seed
    )

    train_subset, val_subset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=generator_size,
    )

    return (
        train_subset.indices,
        val_subset.indices
    )


def _create_subset(
    train_dataset: Dataset,
    val_dataset: Dataset,
    train_indices: list[int],
    val_indices: list[int],
) -> tuple[Subset, Subset]:
    """
    Create training and validation subsets from the original datasets

    Args:
        train_dataset (Dataset): The original training dataset.
        val_dataset (Dataset): The original validation dataset.
        train_indices (list[int]): The indices for the training subset.
        val_indices (list[int]): The indices for the validation subset.
    Returns:
        tuple[Subset, Subset]: The training and validation subsets.
    """

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    val_subset = Subset(
        val_dataset,
        val_indices,
    )

    return (
        train_subset,
        val_subset
    )


def create_datasets() -> Datasets:
    """
    Create the datasets used by the project.

    Returns:
        A Datasets object containing the training,
        validation and test datasets.
    """

    train_dataset = _load_image_folder(
        root=settings.dataset.train_dir,
        transform=get_train_transforms(),
    )

    val_dataset = _load_image_folder(
        root=settings.dataset.train_dir,
        transform=get_test_transforms(),
    )

    test_dataset = _load_image_folder(
        root=settings.dataset.test_dir,
        transform=get_test_transforms(),
    )

    train_indices, val_indices = _split_indices(
        train_dataset,
    )

    train_subset, val_subset = _create_subset(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_indices=train_indices,
        val_indices=val_indices,
    )

    return Datasets(
        train=train_subset,
        val=val_subset,
        test=test_dataset,
        classes=train_dataset.classes,
    )



if __name__ == "__main__":
    datasets = create_datasets()

    print(f"Train: {len(datasets.train)}")
    print(f"Validation: {len(datasets.val)}")
    print(f"Test: {len(datasets.test)}")

    print(f"Classes: {len(datasets.classes)}")
    print(datasets.classes[:10])