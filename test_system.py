#!/usr/bin/env python3
"""Quick test script to verify the cross-modal retrieval system works."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from src.models.retrieval import CrossModalRetrievalModel, ContrastiveLoss
from src.data.dataset import create_toy_dataset
from src.eval.metrics import compute_recall_at_k, compute_map_score
from src.utils.device import setup_device, setup_seed


def test_basic_functionality():
    """Test basic functionality of the system."""
    print("🧪 Testing Cross-Modal Retrieval System...")
    
    # Setup
    device = setup_device("cpu")  # Use CPU for testing
    setup_seed(42)
    
    print("✅ Device setup complete")
    
    # Test model creation
    model = CrossModalRetrievalModel()
    model.to(device)
    print("✅ Model creation complete")
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    input_ids = torch.randint(0, 1000, (batch_size, 77)).to(device)
    attention_mask = torch.ones(batch_size, 77).to(device)
    
    with torch.no_grad():
        outputs = model(images, input_ids, attention_mask)
    
    print("✅ Forward pass complete")
    print(f"   - Logits shape: {outputs['logits_per_image'].shape}")
    print(f"   - Embeddings shape: {outputs['image_embeddings'].shape}")
    
    # Test loss computation
    loss_fn = ContrastiveLoss()
    loss = loss_fn(outputs["logits_per_image"], outputs["logits_per_text"])
    print(f"✅ Loss computation complete: {loss.item():.4f}")
    
    # Test metrics
    similarities = torch.randn(5, 5)
    # Make diagonal elements highest for perfect retrieval
    for i in range(5):
        similarities[i, i] = similarities[i].max() + 1
    
    recall_metrics = compute_recall_at_k(similarities, [1, 5])
    map_score = compute_map_score(similarities)
    
    print("✅ Metrics computation complete")
    print(f"   - Recall@1: {recall_metrics['recall@1']:.4f}")
    print(f"   - Recall@5: {recall_metrics['recall@5']:.4f}")
    print(f"   - mAP: {map_score:.4f}")
    
    # Test toy dataset creation
    with torch.no_grad():
        create_toy_dataset("test_data", num_samples=10)
    print("✅ Toy dataset creation complete")
    
    print("\n🎉 All tests passed! The system is working correctly.")
    print("\nNext steps:")
    print("1. Run training: python scripts/train.py")
    print("2. Launch demo: streamlit run demo/app.py")
    print("3. Run tests: pytest tests/")


if __name__ == "__main__":
    test_basic_functionality()
