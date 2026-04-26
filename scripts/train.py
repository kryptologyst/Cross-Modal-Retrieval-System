"""Training script for cross-modal retrieval model."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import CrossModalDataset, collate_fn, create_toy_dataset
from src.models.retrieval import CrossModalRetrievalModel, ContrastiveLoss
from src.eval.metrics import evaluate_retrieval
from src.utils.device import setup_device, setup_seed, setup_logging, load_config


class Trainer:
    """Trainer class for cross-modal retrieval model."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Setup device and seeding
        self.device = setup_device(config.device)
        setup_seed(config.seed, config.deterministic)
        
        # Setup logging
        self.logger = setup_logging(config)
        
        # Create model and processor
        self.model, self.processor = self._create_model()
        self.model.to(self.device)
        
        # Create loss function
        self.criterion = ContrastiveLoss(temperature=config.model.temperature)
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()
        
        # Training state
        self.current_epoch = 0
        self.best_score = 0.0
        
    def _create_model(self) -> tuple:
        """Create model and processor."""
        model = CrossModalRetrievalModel(
            model_name=self.config.model.name,
            temperature=self.config.model.temperature,
            projection_dim=self.config.model.projection_dim,
        )
        
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(self.config.model.pretrained)
        
        return model, processor
    
    def _create_data_loaders(self) -> tuple:
        """Create data loaders."""
        # Create toy dataset if it doesn't exist
        data_dir = Path(self.config.data.image_dir).parent
        annotations_file = self.config.data.text_file
        
        if not os.path.exists(annotations_file):
            self.logger.info("Creating toy dataset...")
            create_toy_dataset(
                str(data_dir),
                num_samples=self.config.data.max_samples,
            )
        
        # Create datasets
        train_dataset = CrossModalDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            processor=self.processor,
            split="train",
            max_samples=self.config.data.max_samples,
        )
        
        val_dataset = CrossModalDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            processor=self.processor,
            split="val",
            max_samples=self.config.data.max_samples,
        )
        
        test_dataset = CrossModalDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            processor=self.processor,
            split="test",
            max_samples=self.config.data.max_samples,
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.data.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.evaluation.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            collate_fn=collate_fn,
        )
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch in pbar:
            images = batch["images"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_masks = batch["attention_masks"].to(self.device)
            
            # Forward pass
            outputs = self.model(images, input_ids, attention_masks)
            
            # Compute loss
            loss = self.criterion(
                outputs["logits_per_image"],
                outputs["logits_per_text"],
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip_norm,
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.logger.info("Running validation...")
        
        metrics = evaluate_retrieval(
            self.model,
            self.val_loader,
            self.device,
            k_values=self.config.evaluation.k_values,
        )
        
        return metrics
    
    def train(self) -> None:
        """Train the model."""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}:")
            self.logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            self.logger.info(f"  Val Recall@1: {val_metrics['avg_recall@1']:.4f}")
            self.logger.info(f"  Val mAP: {val_metrics['avg_map']:.4f}")
            
            # Save checkpoint
            if epoch % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics)
            
            # Update best score
            current_score = val_metrics["avg_recall@1"]
            if current_score > self.best_score:
                self.best_score = current_score
                self.save_checkpoint(epoch, val_metrics, is_best=True)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }
        
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        if is_best:
            checkpoint_path = checkpoint_dir / "best_model.pth"
        else:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train cross-modal retrieval model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Train
    trainer.train()
    
    # Final evaluation on test set
    trainer.logger.info("Running final evaluation on test set...")
    test_metrics = evaluate_retrieval(
        trainer.model,
        trainer.test_loader,
        trainer.device,
        k_values=config.evaluation.k_values,
    )
    
    trainer.logger.info("Final Test Results:")
    for metric, value in test_metrics.items():
        trainer.logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
