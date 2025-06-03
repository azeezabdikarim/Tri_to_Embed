import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Tuple

def setup_tensorboard(log_dir: str) -> Tuple[SummaryWriter, Path]:
    """
    Setup TensorBoard writer with timestamped directory.
    
    Args:
        log_dir: Base logging directory
        
    Returns:
        Tuple of (SummaryWriter instance, timestamped log directory path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / timestamp
    log_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_path)
    return writer, log_path

def log_metrics(
    writer: SummaryWriter,
    metrics: Dict[str, float],
    step: int,
    prefix: str = ''
):
    """
    Log metrics to TensorBoard.
    
    Args:
        writer: TensorBoard writer
        metrics: Dictionary of metric names and values
        step: Global step
        prefix: Prefix for metric names
    """
    for name, value in metrics.items():
        if prefix:
            name = f"{prefix}/{name}"
        writer.add_scalar(name, value, step)

def save_config(config: Dict[str, Any], save_dir: Path):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

def log_model_info(model: torch.nn.Module, writer: SummaryWriter):
    """
    Log model information to TensorBoard.
    
    Args:
        model: PyTorch model
        writer: TensorBoard writer
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Log as text
    model_info = f"""
    Model Information:
    - Total parameters: {total_params:,}
    - Trainable parameters: {trainable_params:,}
    - Non-trainable parameters: {total_params - trainable_params:,}
    """
    
    writer.add_text('model/info', model_info)