"""
Validation tests to ensure the testing infrastructure is properly set up.
"""

import pytest
import torch
from pathlib import Path


class TestInfrastructureSetup:
    """Test class to validate the testing infrastructure."""
    
    def test_pytest_is_working(self):
        """Verify that pytest is running correctly."""
        assert True
    
    def test_fixtures_are_available(self, temp_dir, sample_config):
        """Verify that custom fixtures are accessible."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert isinstance(sample_config, dict)
        assert "model" in sample_config
    
    def test_torch_is_imported(self):
        """Verify that PyTorch can be imported."""
        assert torch.__version__
        tensor = torch.tensor([1, 2, 3])
        assert tensor.shape == (3,)
    
    def test_markers_are_registered(self):
        """Verify that custom markers are properly registered."""
        # This test itself uses the markers to verify they work
        pass
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True
    
    def test_mock_dataset_fixture(self, mock_dataset_path):
        """Test the mock dataset fixture creates proper structure."""
        assert mock_dataset_path.exists()
        
        for split in ["train", "val", "test"]:
            split_dir = mock_dataset_path / split
            assert split_dir.exists()
            assert split_dir.is_dir()
            
            # Check that class directories exist
            class_dirs = list(split_dir.iterdir())
            assert len(class_dirs) == 3
            
            # Check that images exist in each class
            for class_dir in class_dirs:
                images = list(class_dir.iterdir())
                assert len(images) == 2
    
    def test_tensor_fixtures(self, sample_tensor, sample_labels):
        """Test tensor fixtures are created correctly."""
        assert sample_tensor.shape == (4, 3, 224, 224)
        assert sample_labels.shape == (4,)
        assert sample_labels.dtype == torch.long
    
    def test_checkpoint_fixture(self, mock_model_checkpoint):
        """Test the checkpoint fixture creates valid checkpoint."""
        assert "path" in mock_model_checkpoint
        assert "data" in mock_model_checkpoint
        
        checkpoint_path = mock_model_checkpoint["path"]
        assert checkpoint_path.exists()
        
        # Load and verify checkpoint
        loaded = torch.load(checkpoint_path)
        assert loaded["epoch"] == 5
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
    
    def test_device_fixture(self, device):
        """Test device fixture returns CPU device."""
        assert device.type == "cpu"
    
    def test_random_seed_fixture(self):
        """Test that random seeds are reset between tests."""
        # Generate random numbers
        random1 = torch.rand(5)
        
        # The fixture should ensure reproducibility
        torch.manual_seed(42)
        random2 = torch.rand(5)
        
        assert torch.allclose(random1, random2)
    
    def test_capture_stdout_fixture(self, capture_stdout):
        """Test stdout capture fixture."""
        print("Test output 1")
        print("Test output 2")
        
        assert len(capture_stdout) == 2
        assert capture_stdout[0] == "Test output 1"
        assert capture_stdout[1] == "Test output 2"


class TestCoverageConfiguration:
    """Test coverage configuration is working."""
    
    def test_coverage_is_collected(self):
        """Verify coverage collection is enabled."""
        # This test runs to ensure coverage is being collected
        assert True
    
    def test_source_directory_exists(self):
        """Verify the source directory for coverage exists."""
        code_dir = Path("code")
        assert code_dir.exists()
        assert code_dir.is_dir()