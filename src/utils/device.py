"""
Device utilities.
"""

import torch

def get_device() -> torch.device:
    """
    Get the device to use for computations.

    Priority:
        1. CUDA (GPU)
        2. MPS (Apple Silicon GPU)
        3. CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def print_device_info() -> None:
    """
    Prints detailed information about the available computation device.
    """
    device = get_device()
    print("=" * 40)
    print(f"Active device: {device.type.upper()}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
        cuda_version = torch.version.cuda

        print(f"  • GPU: {gpu_name}")
        print(f"  • Total VRAM: {gpu_memory:.2f} GB")
        print(f"  • CUDA Version (PyTorch): {cuda_version}")

    elif device.type == "mps":
        print("  • Apple Silicon acceleration (MPS) enabled")

    else:
        print("  • Using CPU (No CUDA or MPS GPU detected)")

    print("=" * 40)


if __name__ == "__main__":
    print_device_info()