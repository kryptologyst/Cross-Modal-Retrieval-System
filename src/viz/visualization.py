"""Visualization utilities for cross-modal retrieval."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Tuple, Optional
from PIL import Image


def plot_similarity_matrix(
    similarities: torch.Tensor,
    labels: Optional[List[str]] = None,
    title: str = "Similarity Matrix",
    save_path: Optional[str] = None,
) -> None:
    """Plot similarity matrix heatmap.
    
    Args:
        similarities: Similarity matrix tensor
        labels: Optional labels for axes
        title: Plot title
        save_path: Optional path to save plot
    """
    plt.figure(figsize=(10, 8))
    
    # Convert to numpy and create heatmap
    sim_matrix = similarities.cpu().numpy()
    
    im = plt.imshow(sim_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Similarity Score')
    
    plt.title(title)
    plt.xlabel('Gallery Index')
    plt.ylabel('Query Index')
    
    if labels:
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.yticks(range(len(labels)), labels)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_retrieval_results(
    query_image: Image.Image,
    results: List[Tuple[str, float, str]],
    top_k: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Plot retrieval results with query image and top results.
    
    Args:
        query_image: Query image
        results: List of (text, score, image_path) tuples
        top_k: Number of results to display
        save_path: Optional path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image", fontsize=12)
    axes[0].axis('off')
    
    # Plot top results
    for i, (text, score, image_path) in enumerate(results[:top_k]):
        if i + 1 < len(axes):
            try:
                result_image = Image.open(image_path)
                axes[i + 1].imshow(result_image)
                axes[i + 1].set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=10)
                axes[i + 1].axis('off')
            except Exception as e:
                axes[i + 1].text(0.5, 0.5, f"Error loading image:\n{str(e)}", 
                               ha='center', va='center', transform=axes[i + 1].transAxes)
                axes[i + 1].set_title(f"Rank {i+1}\nScore: {score:.3f}", fontsize=10)
                axes[i + 1].axis('off')
    
    # Hide unused subplots
    for i in range(top_k + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_metrics_comparison(
    metrics_dict: dict,
    metric_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Plot comparison of metrics across different models/runs.
    
    Args:
        metrics_dict: Dictionary of {model_name: {metric: value}}
        metric_names: List of metrics to plot
        save_path: Optional path to save plot
    """
    models = list(metrics_dict.keys())
    x = np.arange(len(models))
    width = 0.8 / len(metric_names)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, metric in enumerate(metric_names):
        values = [metrics_dict[model].get(metric, 0) for model in models]
        ax.bar(x + i * width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Metric Value')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_embedding_space(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = "tsne",
    save_path: Optional[str] = None,
) -> None:
    """Plot 2D visualization of embedding space.
    
    Args:
        embeddings: Embedding tensor
        labels: Optional labels for points
        method: Dimensionality reduction method ("tsne" or "pca")
        save_path: Optional path to save plot
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    # Reduce dimensionality
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    elif method == "pca":
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
    
    embeddings_2d = reducer.fit_transform(embeddings.cpu().numpy())
    
    plt.figure(figsize=(10, 8))
    
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = [l == label for l in labels]
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=label, alpha=0.7)
        
        plt.legend()
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    plt.title(f'Embedding Space Visualization ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
