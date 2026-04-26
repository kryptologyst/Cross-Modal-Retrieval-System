"""Cross-modal retrieval models and architectures."""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor


class CrossModalRetrievalModel(nn.Module):
    """Enhanced cross-modal retrieval model with contrastive learning.
    
    Extends CLIP with additional features for better retrieval performance.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        temperature: float = 0.07,
        projection_dim: int = 512,
        freeze_backbone: bool = False,
    ):
        """Initialize the model.
        
        Args:
            model_name: Name of the CLIP model to use
            temperature: Temperature for contrastive learning
            projection_dim: Dimension of projection layers
            freeze_backbone: Whether to freeze CLIP backbone
        """
        super().__init__()
        
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Load CLIP model
        self.clip_model = CLIPModel.from_pretrained(model_name)
        
        if freeze_backbone:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Additional projection layers for enhanced representations
        self.image_projection = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        
        # Initialize projections
        self._init_projections()
    
    def _init_projections(self) -> None:
        """Initialize projection layers."""
        for module in [self.image_projection, self.text_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to embeddings.
        
        Args:
            images: Input images tensor
            
        Returns:
            torch.Tensor: Image embeddings
        """
        image_features = self.clip_model.get_image_features(images)
        image_features = F.normalize(image_features, p=2, dim=-1)
        
        # Apply additional projection
        enhanced_features = self.image_projection(image_features)
        enhanced_features = F.normalize(enhanced_features, p=2, dim=-1)
        
        return enhanced_features
    
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text to embeddings.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            torch.Tensor: Text embeddings
        """
        text_features = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # Apply additional projection
        enhanced_features = self.text_projection(text_features)
        enhanced_features = F.normalize(enhanced_features, p=2, dim=-1)
        
        return enhanced_features
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            images: Input images
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Dict containing logits and embeddings
        """
        # Get embeddings
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(input_ids, attention_mask)
        
        # Compute similarity logits
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        logits_per_text = logits_per_image.T
        
        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
        }
    
    def compute_similarity(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between image and text embeddings.
        
        Args:
            image_embeddings: Image embeddings
            text_embeddings: Text embeddings
            
        Returns:
            torch.Tensor: Similarity scores
        """
        return torch.matmul(image_embeddings, text_embeddings.T) / self.temperature


class ContrastiveLoss(nn.Module):
    """Contrastive loss for cross-modal retrieval."""
    
    def __init__(self, temperature: float = 0.07):
        """Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for scaling
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits_per_image: torch.Tensor,
        logits_per_text: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive loss.
        
        Args:
            logits_per_image: Image-to-text logits
            logits_per_text: Text-to-image logits
            
        Returns:
            torch.Tensor: Contrastive loss
        """
        batch_size = logits_per_image.shape[0]
        
        # Create labels for positive pairs (diagonal)
        labels = torch.arange(batch_size, device=logits_per_image.device)
        
        # Compute cross-entropy losses
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        
        # Total loss
        total_loss = (loss_i2t + loss_t2i) / 2
        
        return total_loss


def create_model(config: Dict[str, Any]) -> Tuple[CrossModalRetrievalModel, CLIPProcessor]:
    """Create model and processor from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Tuple of (model, processor)
    """
    model = CrossModalRetrievalModel(
        model_name=config["name"],
        temperature=config["temperature"],
        projection_dim=config["projection_dim"],
    )
    
    processor = CLIPProcessor.from_pretrained(config["pretrained"])
    
    return model, processor
