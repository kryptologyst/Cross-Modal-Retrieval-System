"""Unit tests for cross-modal retrieval system."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import json

from src.models.retrieval import CrossModalRetrievalModel, ContrastiveLoss
from src.eval.metrics import compute_recall_at_k, compute_map_score, compute_median_rank
from src.data.dataset import CrossModalDataset, create_toy_dataset
from src.utils.device import setup_device, setup_seed


class TestCrossModalRetrievalModel:
    """Test cases for CrossModalRetrievalModel."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = CrossModalRetrievalModel()
        assert isinstance(model, CrossModalRetrievalModel)
        assert hasattr(model, 'clip_model')
        assert hasattr(model, 'image_projection')
        assert hasattr(model, 'text_projection')
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = CrossModalRetrievalModel()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        outputs = model(images, input_ids, attention_mask)
        
        assert "logits_per_image" in outputs
        assert "logits_per_text" in outputs
        assert "image_embeddings" in outputs
        assert "text_embeddings" in outputs
        
        assert outputs["logits_per_image"].shape == (batch_size, batch_size)
        assert outputs["logits_per_text"].shape == (batch_size, batch_size)
    
    def test_encode_image(self):
        """Test image encoding."""
        model = CrossModalRetrievalModel()
        
        images = torch.randn(2, 3, 224, 224)
        embeddings = model.encode_image(images)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.projection_dim
        
        # Check normalization
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    
    def test_encode_text(self):
        """Test text encoding."""
        model = CrossModalRetrievalModel()
        
        input_ids = torch.randint(0, 1000, (2, 77))
        attention_mask = torch.ones(2, 77)
        
        embeddings = model.encode_text(input_ids, attention_mask)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.projection_dim
        
        # Check normalization
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


class TestContrastiveLoss:
    """Test cases for ContrastiveLoss."""
    
    def test_loss_computation(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss()
        
        batch_size = 4
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = torch.randn(batch_size, batch_size)
        
        loss = loss_fn(logits_per_image, logits_per_text)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_loss_shape(self):
        """Test loss output shape."""
        loss_fn = ContrastiveLoss()
        
        batch_size = 3
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = torch.randn(batch_size, batch_size)
        
        loss = loss_fn(logits_per_image, logits_per_text)
        
        assert loss.shape == ()


class TestEvaluationMetrics:
    """Test cases for evaluation metrics."""
    
    def test_recall_at_k(self):
        """Test Recall@K computation."""
        batch_size = 5
        similarities = torch.randn(batch_size, batch_size)
        
        # Make diagonal elements highest (perfect retrieval)
        for i in range(batch_size):
            similarities[i, i] = similarities[i].max() + 1
        
        results = compute_recall_at_k(similarities, k_values=[1, 2, 5])
        
        assert "recall@1" in results
        assert "recall@2" in results
        assert "recall@5" in results
        
        # Perfect retrieval should give recall@1 = 1.0
        assert results["recall@1"] == 1.0
        assert results["recall@2"] == 1.0
        assert results["recall@5"] == 1.0
    
    def test_map_score(self):
        """Test mAP computation."""
        batch_size = 3
        similarities = torch.randn(batch_size, batch_size)
        
        # Make diagonal elements highest
        for i in range(batch_size):
            similarities[i, i] = similarities[i].max() + 1
        
        map_score = compute_map_score(similarities)
        
        assert isinstance(map_score, float)
        assert 0 <= map_score <= 1
        assert map_score > 0.9  # Should be high for perfect retrieval
    
    def test_median_rank(self):
        """Test median rank computation."""
        batch_size = 4
        similarities = torch.randn(batch_size, batch_size)
        
        # Make diagonal elements highest
        for i in range(batch_size):
            similarities[i, i] = similarities[i].max() + 1
        
        median_rank = compute_median_rank(similarities)
        
        assert isinstance(median_rank, float)
        assert median_rank == 1.0  # Perfect retrieval should give rank 1


class TestDataset:
    """Test cases for dataset."""
    
    def test_toy_dataset_creation(self):
        """Test toy dataset creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            create_toy_dataset(temp_dir, num_samples=10)
            
            # Check if files were created
            data_path = Path(temp_dir)
            assert (data_path / "annotations.json").exists()
            assert (data_path / "images").exists()
            
            # Check annotations
            with open(data_path / "annotations.json") as f:
                annotations = json.load(f)
            
            assert "train" in annotations
            assert "val" in annotations
            assert "test" in annotations
            
            total_samples = sum(len(split) for split in annotations.values())
            assert total_samples == 10
    
    def test_dataset_loading(self):
        """Test dataset loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create toy dataset
            create_toy_dataset(temp_dir, num_samples=5)
            
            # Mock processor
            class MockProcessor:
                def __call__(self, text, images, return_tensors, padding, truncation):
                    return {
                        "pixel_values": torch.randn(1, 3, 224, 224),
                        "input_ids": torch.randint(0, 1000, (1, 77)),
                        "attention_mask": torch.ones(1, 77),
                    }
            
            processor = MockProcessor()
            
            dataset = CrossModalDataset(
                data_dir=temp_dir,
                annotations_file=str(Path(temp_dir) / "annotations.json"),
                processor=processor,
                split="train",
            )
            
            assert len(dataset) > 0
            
            # Test getting an item
            item = dataset[0]
            assert "image" in item
            assert "input_ids" in item
            assert "attention_mask" in item
            assert "text" in item


class TestDeviceUtils:
    """Test cases for device utilities."""
    
    def test_setup_device(self):
        """Test device setup."""
        device = setup_device("cpu")
        assert device.type == "cpu"
        
        # Test auto device selection
        device = setup_device("auto")
        assert device.type in ["cpu", "cuda", "mps"]
    
    def test_setup_seed(self):
        """Test seed setup."""
        setup_seed(42)
        
        # Check if random numbers are deterministic
        torch.manual_seed(42)
        first_random = torch.randn(1)
        
        setup_seed(42)
        torch.manual_seed(42)
        second_random = torch.randn(1)
        
        assert torch.allclose(first_random, second_random)


if __name__ == "__main__":
    pytest.main([__file__])
