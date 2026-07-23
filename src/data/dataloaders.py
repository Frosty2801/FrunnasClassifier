
"""
Dataloader creation utilities
"""

from torch.utils.data import DataLoader, Dataset

from src.settings import settings
from src.data.datasets import create_datasets
from src.data.types import Dataloaders



def _create_dataloader(
    dataset: Dataset,
    shuffle: bool = False,
    ) -> DataLoader:
    """
    Create a DataLoader for a dataset.

    Args:
        dataset: Dataset to iterate over.
        shuffle: Whether to shuffle the dataset.

    Returns:
        Configured DataLoader.
    """

    return DataLoader(
        dataset=dataset,
        batch_size=settings.training.batch_size,
        shuffle=shuffle,
        num_workers=settings.dataloader.num_workers,
        pin_memory=settings.dataloader.pin_memory,
    )

def create_dataloaders() -> Dataloaders:
    """
    Create DataLoaders for training and testing datasets.

    Returns:
        Dataloaders for training and testing.
    """

    datasets = create_datasets()

    train_loader = _create_dataloader(
        dataset=datasets.train,
        shuffle=True,
    )

    val_loader = _create_dataloader(
        dataset=datasets.val,
        shuffle=False,
    )

    test_loader = _create_dataloader(
        dataset=datasets.test,
        shuffle=False,
    )

    return Dataloaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        classes=datasets.classes,
    )


if __name__ == "__main__":
    dataloaders = create_dataloaders()

    print(f"Classes: {len(dataloaders.classes)}")

    images, labels = next(iter(dataloaders.train))

    print(f"Images: {images.shape}")
    print(f"Labels: {labels.shape}")