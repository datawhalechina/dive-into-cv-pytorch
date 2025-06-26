import os
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """
    Create a temporary directory for test files.
    
    Yields:
        Path: Path to the temporary directory
    """
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """
    Provide a sample configuration dictionary for testing.
    
    Returns:
        Dict[str, Any]: Sample configuration
    """
    return {
        "model": {
            "name": "test_model",
            "num_classes": 10,
            "input_size": 224,
            "channels": 3,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "device": "cpu",
        },
        "data": {
            "train_path": "/path/to/train",
            "val_path": "/path/to/val",
            "test_path": "/path/to/test",
        }
    }


@pytest.fixture
def mock_dataset_path(temp_dir: Path) -> Path:
    """
    Create a mock dataset directory structure.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Path: Path to the mock dataset
    """
    dataset_path = temp_dir / "mock_dataset"
    
    # Create subdirectories
    for split in ["train", "val", "test"]:
        split_dir = dataset_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some mock class directories
        for class_idx in range(3):
            class_dir = split_dir / f"class_{class_idx}"
            class_dir.mkdir(exist_ok=True)
            
            # Create mock image files
            for img_idx in range(2):
                img_file = class_dir / f"image_{img_idx}.jpg"
                img_file.touch()
    
    return dataset_path


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """
    Create a sample tensor for testing.
    
    Returns:
        torch.Tensor: A sample 4D tensor (batch, channels, height, width)
    """
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_labels() -> torch.Tensor:
    """
    Create sample labels for testing.
    
    Returns:
        torch.Tensor: A sample label tensor
    """
    return torch.randint(0, 10, (4,))


@pytest.fixture
def device() -> torch.device:
    """
    Get the appropriate device for testing.
    
    Returns:
        torch.device: CPU device for consistent testing
    """
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """
    Reset random seeds before each test for reproducibility.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Dict[str, Any]:
    """
    Create a mock model checkpoint.
    
    Args:
        temp_dir: Temporary directory fixture
        
    Returns:
        Dict[str, Any]: Mock checkpoint data
    """
    checkpoint = {
        "epoch": 5,
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        "loss": 0.1234,
        "accuracy": 0.95,
    }
    
    checkpoint_path = temp_dir / "checkpoint.pth"
    torch.save(checkpoint, checkpoint_path)
    
    return {
        "path": checkpoint_path,
        "data": checkpoint
    }


@pytest.fixture
def capture_stdout(monkeypatch):
    """
    Capture stdout for testing print statements.
    
    Args:
        monkeypatch: pytest monkeypatch fixture
        
    Returns:
        list: List to collect stdout outputs
    """
    outputs = []
    
    def mock_print(*args, **kwargs):
        outputs.append(" ".join(map(str, args)))
    
    monkeypatch.setattr("builtins.print", mock_print)
    return outputs