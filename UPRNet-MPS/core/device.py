import torch


def resolve_device(device_arg="auto"):
    if isinstance(device_arg, torch.device):
        return device_arg

    if device_arg in (None, "", "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Warning: MPS was requested but is not available in this Python environment. Falling back to CPU.")
        return torch.device("cpu")

    if isinstance(device_arg, str) and device_arg.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_arg}")
        print(f"Warning: CUDA device {device_arg} was requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    if isinstance(device_arg, str) and device_arg.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_arg)
        print(f"Warning: {device_arg} was requested but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")

    return torch.device(device_arg)
