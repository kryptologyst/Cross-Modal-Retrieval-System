"""Utility functions for device management, seeding, and logging."""

import logging
import os
import random
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig


def setup_device(device: str = "auto") -> torch.device:
    """Setup device with fallback: CUDA -> MPS -> CPU.
    
    Args:
        device: Device preference ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The selected device
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    
    # Set device-specific optimizations
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    elif device_obj.type == "mps":
        # MPS-specific optimizations if needed
        pass
    
    return device_obj


def setup_seed(seed: int, deterministic: bool = True) -> None:
    """Setup random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_logging(config: DictConfig) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        logging.Logger: Configured logger
    """
    log_dir = config.logging.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("cross_modal_retrieval")
    logger.setLevel(getattr(logging, config.logging.level))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "cross_modal_retrieval.log")
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DictConfig: Loaded configuration
    """
    from omegaconf import OmegaConf
    
    config = OmegaConf.load(config_path)
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration to save
        save_path: Path to save configuration
    """
    from omegaconf import OmegaConf
    
    OmegaConf.save(config, save_path)


def get_model_size(model: torch.nn.Module) -> Dict[str, int]:
    """Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dict containing parameter counts and model size
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
    }


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total parameter count
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
