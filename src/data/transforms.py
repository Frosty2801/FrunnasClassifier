"""
Image transformation to tensors
"""

from torchvision import transforms

from src.settings import settings


def get_train_transforms() -> transforms.Compose:
    """
    Transform training images
    """

    return transforms.Compose(
        [
            transforms.Resize((settings.dataset.image_size, settings.dataset.image_size)),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """
    Transform validation images
    """

    return transforms.Compose(
        [
            transforms.Resize((settings.dataset.image_size, settings.dataset.image_size)),
            transforms.ToTensor(),
        ]
    )


def get_test_transforms() -> transforms.Compose:
    """
    Transform test images
    """

    return transforms.Compose(
        [
            transforms.Resize((settings.dataset.image_size, settings.dataset.image_size)),
            transforms.ToTensor(),
        ]
    )