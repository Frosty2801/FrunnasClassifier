"""
Image transformation to tensors
"""

from torchvision import transforms

from src.config import IMAGE_SIZE


def get_train_transforms() -> transforms.Compose:
    """
    Transform training images
    """

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """
    Transform validation images
    """

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )


def get_test_transforms() -> transforms.Compose:
    """
    Transform test images
    """

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )