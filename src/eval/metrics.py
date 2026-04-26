"""Evaluation metrics and utilities for cross-modal retrieval."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score


def compute_recall_at_k(
    similarities: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Compute Recall@K metrics.
    
    Args:
        similarities: Similarity matrix (n_queries, n_gallery)
        k_values: List of K values to compute
        
    Returns:
        Dict containing Recall@K scores
    """
    # Get top-k indices for each query
    _, top_k_indices = torch.topk(similarities, max(k_values), dim=1)
    
    results = {}
    batch_size = similarities.shape[0]
    
    for k in k_values:
        # Check if correct item is in top-k
        correct_mask = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(batch_size):
            # For retrieval, the correct match is at index i (diagonal)
            correct_mask[i] = i in top_k_indices[i, :k]
        
        recall_k = correct_mask.float().mean().item()
        results[f"recall@{k}"] = recall_k
    
    return results


def compute_map_score(similarities: torch.Tensor) -> float:
    """Compute Mean Average Precision (mAP).
    
    Args:
        similarities: Similarity matrix (n_queries, n_gallery)
        
    Returns:
        float: mAP score
    """
    batch_size = similarities.shape[0]
    ap_scores = []
    
    for i in range(batch_size):
        # Create binary relevance labels (1 for correct match, 0 for others)
        y_true = torch.zeros(batch_size)
        y_true[i] = 1  # Correct match is at diagonal
        
        # Get similarity scores for this query
        y_scores = similarities[i].cpu().numpy()
        
        # Compute average precision
        ap = average_precision_score(y_true.numpy(), y_scores)
        ap_scores.append(ap)
    
    return np.mean(ap_scores)


def compute_median_rank(similarities: torch.Tensor) -> float:
    """Compute median rank of correct matches.
    
    Args:
        similarities: Similarity matrix (n_queries, n_gallery)
        
    Returns:
        float: Median rank
    """
    batch_size = similarities.shape[0]
    ranks = []
    
    for i in range(batch_size):
        # Get sorted indices for this query
        _, sorted_indices = torch.sort(similarities[i], descending=True)
        
        # Find rank of correct match
        rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    
    return np.median(ranks)


def evaluate_retrieval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    k_values: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    """Evaluate retrieval performance.
    
    Args:
        model: Trained model
        dataloader: Data loader for evaluation
        device: Device to run evaluation on
        k_values: List of K values for Recall@K
        
    Returns:
        Dict containing evaluation metrics
    """
    model.eval()
    
    all_image_embeddings = []
    all_text_embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            
            # Get embeddings
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(input_ids, attention_masks)
            
            all_image_embeddings.append(image_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())
    
    # Concatenate all embeddings
    image_embeddings = torch.cat(all_image_embeddings, dim=0)
    text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    # Compute similarities
    similarities = torch.matmul(image_embeddings, text_embeddings.T)
    
    # Compute metrics
    metrics = {}
    
    # Image-to-Text retrieval
    i2t_recall = compute_recall_at_k(similarities, k_values)
    for key, value in i2t_recall.items():
        metrics[f"i2t_{key}"] = value
    
    metrics["i2t_map"] = compute_map_score(similarities)
    metrics["i2t_median_rank"] = compute_median_rank(similarities)
    
    # Text-to-Image retrieval
    t2i_recall = compute_recall_at_k(similarities.T, k_values)
    for key, value in t2i_recall.items():
        metrics[f"t2i_{key}"] = value
    
    metrics["t2i_map"] = compute_map_score(similarities.T)
    metrics["t2i_median_rank"] = compute_median_rank(similarities.T)
    
    # Average metrics
    metrics["avg_recall@1"] = (metrics["i2t_recall@1"] + metrics["t2i_recall@1"]) / 2
    metrics["avg_recall@5"] = (metrics["i2t_recall@5"] + metrics["t2i_recall@5"]) / 2
    metrics["avg_recall@10"] = (metrics["i2t_recall@10"] + metrics["t2i_recall@10"]) / 2
    metrics["avg_map"] = (metrics["i2t_map"] + metrics["t2i_map"]) / 2
    metrics["avg_median_rank"] = (metrics["i2t_median_rank"] + metrics["t2i_median_rank"]) / 2
    
    return metrics


def create_leaderboard(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> str:
    """Create a leaderboard from evaluation results.
    
    Args:
        results: Dictionary of results from different models/runs
        save_path: Optional path to save leaderboard
        
    Returns:
        str: Formatted leaderboard
    """
    # Define metrics to display
    metrics = [
        "avg_recall@1", "avg_recall@5", "avg_recall@10",
        "avg_map", "avg_median_rank"
    ]
    
    # Create header
    header = "Model".ljust(20)
    for metric in metrics:
        header += f"{metric}".ljust(15)
    
    lines = [header, "=" * len(header)]
    
    # Add results
    for model_name, metrics_dict in results.items():
        line = model_name.ljust(20)
        for metric in metrics:
            value = metrics_dict.get(metric, 0.0)
            line += f"{value:.4f}".ljust(15)
        lines.append(line)
    
    leaderboard = "\n".join(lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(leaderboard)
    
    return leaderboard
