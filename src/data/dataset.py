"""
Dataset and Dataloader
"""

from torch.utils.data import (
    DataLoader,
    random_split,
    )
from torchvision.datasets import ImageFolder

from src.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    TRAIN_DIR,
    TEST_DIR,
    VALIDATION_SPLIT
)

from src.transforms import (
    get_train_transforms,
    get_test_transforms,
)

def create_datasets() -> tuple[ImageFolder, ImageFolder, ImageFolder]:
    """
    training and test datasets
    """

    train_dataset = ImageFolder(
        root=TRAIN_DIR,
        transform=get_train_transforms(),
    )

    train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(
        train_dataset, 
        [train_size, val_size]
    )

    test_dataset = ImageFolder(
        root=TEST_DIR,
        transform=get_test_transforms(),
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders() -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Dataloaders
    """

    train_dataset, val_dataset, test_dataset = create_datasets()

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return train_loader, test_loader, train_dataset.classes
if __name__ == "__main__":
    train_loader, test_loader, train_classes = create_dataloaders()

    print(f"Train classes: {len(train_classes)}")

    images, labels = next(iter(train_loader))

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
