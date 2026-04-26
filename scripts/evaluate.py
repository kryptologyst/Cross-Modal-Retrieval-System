"""Evaluation script for cross-modal retrieval model."""

import argparse
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import DataLoader

from src.data.dataset import CrossModalDataset, collate_fn
from src.models.retrieval import CrossModalRetrievalModel
from src.eval.metrics import evaluate_retrieval, create_leaderboard
from src.utils.device import setup_device, setup_seed, setup_logging, load_config


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate cross-modal retrieval model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup device and seeding
    device = setup_device(config.device)
    setup_seed(config.seed, config.deterministic)
    
    # Setup logging
    logger = setup_logging(config)
    
    # Create model
    model = CrossModalRetrievalModel(
        model_name=config.model.name,
        temperature=config.model.temperature,
        projection_dim=config.model.projection_dim,
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create data loader
    from transformers import CLIPProcessor
    
    processor = CLIPProcessor.from_pretrained(config.model.pretrained)
    
    dataset = CrossModalDataset(
        data_dir=str(Path(config.data.image_dir).parent),
        annotations_file=config.data.text_file,
        processor=processor,
        split="test",
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    metrics = evaluate_retrieval(
        model,
        dataloader,
        device,
        k_values=config.evaluation.k_values,
    )
    
    # Print results
    logger.info("Evaluation Results:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Create leaderboard
    results = {"best_model": metrics}
    leaderboard = create_leaderboard(results)
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "evaluation_results.txt", "w") as f:
        f.write(leaderboard)
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
