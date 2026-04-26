"""Data loading and preprocessing utilities for cross-modal retrieval."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class CrossModalDataset(Dataset):
    """Dataset for cross-modal retrieval tasks.
    
    Supports image-text pairs with flexible data loading and preprocessing.
    """
    
    def __init__(
        self,
        data_dir: str,
        annotations_file: str,
        processor: CLIPProcessor,
        split: str = "train",
        max_samples: Optional[int] = None,
        image_size: int = 224,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            annotations_file: Path to JSON file with annotations
            processor: CLIP processor for text and image processing
            split: Data split ("train", "val", "test")
            max_samples: Maximum number of samples to load
            image_size: Target image size for resizing
        """
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.split = split
        self.image_size = image_size
        
        # Load annotations
        with open(annotations_file, "r") as f:
            self.annotations = json.load(f)
        
        # Filter by split if specified
        if split in self.annotations:
            self.data = self.annotations[split]
        else:
            self.data = self.annotations
        
        # Limit samples if specified
        if max_samples is not None:
            self.data = self.data[:max_samples]
        
        self.image_dir = self.data_dir / "images"
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dict containing processed image and text data
        """
        item = self.data[idx]
        
        # Load and process image
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        
        # Process with CLIP processor
        processed = self.processor(
            text=item["text"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return {
            "image": processed["pixel_values"].squeeze(0),
            "input_ids": processed["input_ids"].squeeze(0),
            "attention_mask": processed["attention_mask"].squeeze(0),
            "text": item["text"],
            "image_path": str(image_path),
            "idx": idx,
        }


def create_toy_dataset(
    output_dir: str,
    num_samples: int = 100,
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    """Create a toy dataset for testing purposes.
    
    Args:
        output_dir: Directory to save the dataset
        num_samples: Number of samples to generate
        image_size: Size of generated images
    """
    import random
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create images directory
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Sample categories and descriptions
    categories = [
        "cat", "dog", "bird", "car", "tree", "house", "mountain", "ocean",
        "flower", "book", "phone", "laptop", "chair", "table", "food"
    ]
    
    descriptions = [
        "A beautiful {category} in the garden",
        "A cute {category} playing outside",
        "A majestic {category} in nature",
        "A colorful {category} in the sunlight",
        "A peaceful {category} in the forest",
        "A happy {category} enjoying life",
        "A lovely {category} with flowers",
        "A stunning {category} at sunset",
    ]
    
    annotations = {"train": [], "val": [], "test": []}
    
    for i in range(num_samples):
        # Generate random image
        image_array = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
        image = Image.fromarray(image_array)
        
        # Save image
        image_filename = f"sample_{i:04d}.jpg"
        image.save(images_dir / image_filename)
        
        # Generate description
        category = random.choice(categories)
        description_template = random.choice(descriptions)
        description = description_template.format(category=category)
        
        # Assign to split
        split = "train" if i < int(0.8 * num_samples) else "val" if i < int(0.9 * num_samples) else "test"
        
        annotations[split].append({
            "image": image_filename,
            "text": description,
            "category": category,
        })
    
    # Save annotations
    with open(output_path / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created toy dataset with {num_samples} samples in {output_dir}")


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.
    
    Args:
        batch: List of batch items
        
    Returns:
        Dict containing batched tensors
    """
    images = torch.stack([item["image"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    
    texts = [item["text"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    indices = torch.tensor([item["idx"] for item in batch])
    
    return {
        "images": images,
        "input_ids": input_ids,
        "attention_masks": attention_masks,
        "texts": texts,
        "image_paths": image_paths,
        "indices": indices,
    }
