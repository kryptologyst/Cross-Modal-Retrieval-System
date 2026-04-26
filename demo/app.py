"""Streamlit demo for cross-modal retrieval system."""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.retrieval import CrossModalRetrievalModel
from data.dataset import CrossModalDataset, create_toy_dataset
from utils.device import setup_device, load_config


class RetrievalDemo:
    """Demo class for cross-modal retrieval."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize demo.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.device = setup_device(self.config.device)
        
        # Load model
        self.model = self._load_model()
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Create gallery embeddings
        self.gallery_embeddings = self._create_gallery_embeddings()
    
    def _load_model(self) -> CrossModalRetrievalModel:
        """Load trained model."""
        model = CrossModalRetrievalModel(
            model_name=self.config.model.name,
            temperature=self.config.model.temperature,
            projection_dim=self.config.model.projection_dim,
        )
        
        # Try to load checkpoint
        checkpoint_path = Path("checkpoints/best_model.pth")
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            st.success("Loaded trained model checkpoint")
        else:
            st.warning("No trained model found, using pre-trained CLIP")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def _load_dataset(self) -> CrossModalDataset:
        """Load dataset."""
        from transformers import CLIPProcessor
        
        processor = CLIPProcessor.from_pretrained(self.config.model.pretrained)
        
        # Create toy dataset if it doesn't exist
        data_dir = Path(self.config.data.image_dir).parent
        annotations_file = self.config.data.text_file
        
        if not os.path.exists(annotations_file):
            create_toy_dataset(
                str(data_dir),
                num_samples=self.config.data.max_samples,
            )
        
        dataset = CrossModalDataset(
            data_dir=str(data_dir),
            annotations_file=annotations_file,
            processor=processor,
            split="test",  # Use test split for demo
        )
        
        return dataset
    
    def _create_gallery_embeddings(self) -> Tuple[torch.Tensor, List[str]]:
        """Create embeddings for gallery images."""
        embeddings = []
        texts = []
        
        with torch.no_grad():
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                
                image = item["image"].unsqueeze(0).to(self.device)
                input_ids = item["input_ids"].unsqueeze(0).to(self.device)
                attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
                
                # Get embeddings
                image_emb = self.model.encode_image(image)
                text_emb = self.model.encode_text(input_ids, attention_mask)
                
                embeddings.append(image_emb.cpu())
                texts.append(item["text"])
        
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings, texts
    
    def search_by_image(self, query_image: Image.Image, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Search gallery by image query.
        
        Args:
            query_image: Query image
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, image_path) tuples
        """
        from transformers import CLIPProcessor
        
        processor = CLIPProcessor.from_pretrained(self.config.model.pretrained)
        
        # Process query image
        inputs = processor(images=query_image, return_tensors="pt")
        query_image_tensor = inputs["pixel_values"].to(self.device)
        
        with torch.no_grad():
            # Get query embedding
            query_embedding = self.model.encode_image(query_image_tensor)
            
            # Compute similarities
            similarities = torch.matmul(query_embedding, self.gallery_embeddings.T)
            similarities = similarities.squeeze(0)
            
            # Get top-k results
            _, top_indices = torch.topk(similarities, top_k)
            
            results = []
            for idx in top_indices:
                score = similarities[idx].item()
                text = self.dataset[idx]["text"]
                image_path = self.dataset[idx]["image_path"]
                results.append((text, score, image_path))
        
        return results
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """Search gallery by text query.
        
        Args:
            query_text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (text, score, image_path) tuples
        """
        from transformers import CLIPProcessor
        
        processor = CLIPProcessor.from_pretrained(self.config.model.pretrained)
        
        # Process query text
        inputs = processor(text=query_text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            # Get query embedding
            query_embedding = self.model.encode_text(input_ids, attention_mask)
            
            # Compute similarities
            similarities = torch.matmul(query_embedding, self.gallery_embeddings.T)
            similarities = similarities.squeeze(0)
            
            # Get top-k results
            _, top_indices = torch.topk(similarities, top_k)
            
            results = []
            for idx in top_indices:
                score = similarities[idx].item()
                text = self.dataset[idx]["text"]
                image_path = self.dataset[idx]["image_path"]
                results.append((text, score, image_path))
        
        return results


def main():
    """Main demo function."""
    st.set_page_config(
        page_title="Cross-Modal Retrieval Demo",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Cross-Modal Retrieval System")
    st.markdown("Search images with text or find text descriptions for images using CLIP-based cross-modal retrieval.")
    
    # Initialize demo
    if "demo" not in st.session_state:
        with st.spinner("Loading model and dataset..."):
            st.session_state.demo = RetrievalDemo()
    
    demo = st.session_state.demo
    
    # Sidebar controls
    st.sidebar.header("Search Options")
    search_type = st.sidebar.selectbox(
        "Search Type",
        ["Image → Text", "Text → Image"],
        help="Choose whether to search with an image or text query"
    )
    
    top_k = st.sidebar.slider(
        "Number of Results",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of top results to display"
    )
    
    # Main content
    if search_type == "Image → Text":
        st.header("Image → Text Retrieval")
        st.markdown("Upload an image to find the most relevant text descriptions.")
        
        # Image upload
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image to search for relevant text descriptions"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Query Image", use_column_width=True)
            
            # Search
            if st.button("Search", type="primary"):
                with st.spinner("Searching..."):
                    results = demo.search_by_image(image, top_k)
                
                # Display results
                st.subheader("Search Results")
                
                for i, (text, score, image_path) in enumerate(results):
                    with st.expander(f"Result {i+1} (Score: {score:.3f})"):
                        st.write(f"**Text:** {text}")
                        st.write(f"**Similarity Score:** {score:.3f}")
                        
                        # Show corresponding image
                        if os.path.exists(image_path):
                            result_image = Image.open(image_path)
                            st.image(result_image, caption=f"Gallery Image {i+1}")
    
    else:  # Text → Image
        st.header("Text → Image Retrieval")
        st.markdown("Enter text to find the most relevant images.")
        
        # Text input
        query_text = st.text_area(
            "Enter your search query",
            placeholder="e.g., 'A beautiful cat in the garden'",
            help="Describe what you're looking for in the images"
        )
        
        if query_text.strip():
            if st.button("Search", type="primary"):
                with st.spinner("Searching..."):
                    results = demo.search_by_text(query_text, top_k)
                
                # Display results
                st.subheader("Search Results")
                
                # Create columns for results
                cols = st.columns(min(top_k, 3))
                
                for i, (text, score, image_path) in enumerate(results):
                    col_idx = i % 3
                    
                    with cols[col_idx]:
                        st.write(f"**Score:** {score:.3f}")
                        st.write(f"**Description:** {text}")
                        
                        # Show corresponding image
                        if os.path.exists(image_path):
                            result_image = Image.open(image_path)
                            st.image(result_image, caption=f"Result {i+1}")
    
    # Dataset statistics
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"**Gallery Size:** {len(demo.dataset)}")
    st.sidebar.write(f"**Model:** {demo.config.model.name}")
    st.sidebar.write(f"**Device:** {demo.device}")
    
    # Safety disclaimer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚠️ Disclaimer**")
    st.sidebar.markdown(
        "This is a research/educational demo. Results may not be accurate for all queries. "
        "Use responsibly and verify important information independently."
    )


if __name__ == "__main__":
    main()
