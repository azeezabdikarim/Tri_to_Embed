"""
Experiment Manager for Organized Experiment Tracking

Generates descriptive experiment names and manages experiment organization
without external dependencies.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentManager:
    """
    Manages experiment organization and naming.
    
    Features:
    - Generates descriptive experiment names from config
    - Creates organized directory structure
    - Saves experiment metadata
    - Supports experiment comparison
    """
    
    def __init__(self, base_log_dir: str = "logs"):
        """
        Initialize experiment manager.
        
        Args:
            base_log_dir: Base directory for all experiments
        """
        self.base_log_dir = Path(base_log_dir)
        self.base_log_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_experiment_name(self, config: Dict[str, Any]) -> str:
        """
        Generate a descriptive experiment name from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            experiment_name: Descriptive name for the experiment
        """
        name_parts = []
        
        # Model type
        model_config = config.get('model', {})
        use_planes = model_config.get('use_planes', True)
        use_mlp_features = model_config.get('use_mlp_features', False)
        
        if use_planes and use_mlp_features:
            name_parts.append("hybrid")
        elif use_mlp_features:
            name_parts.append("mlp_only")
        else:
            name_parts.append("planes_only")
        
        # Encoder details for planes
        if use_planes:
            encoder_type = model_config.get('encoder_type', 'resnet18')
            name_parts.append(encoder_type)
            
            plane_fusion = model_config.get('plane_fusion', 'concat')
            name_parts.append(f"pf_{plane_fusion}")
        
        # MLP encoder details
        if use_mlp_features:
            mlp_encoders = model_config.get('mlp_encoders', {})
            
            # Base MLP embedding dim
            base_mlp_dim = mlp_encoders.get('base_mlp', {}).get('embedding_dim', 128)
            # Head MLP embedding dim  
            head_mlp_dim = mlp_encoders.get('head_mlp', {}).get('embedding_dim', 256)
            name_parts.append(f"mlp_{base_mlp_dim}_{head_mlp_dim}")
            
            # Pooling strategy
            pooling = mlp_encoders.get('pooling_strategy', 'max')
            name_parts.append(f"pool_{pooling}")
            
            # Fusion strategies
            fusion = model_config.get('fusion', {})
            layer_fusion = fusion.get('layer_fusion', 'concat')
            stream_fusion = fusion.get('stream_fusion', 'concat')
            name_parts.append(f"lf_{layer_fusion}")
            if use_planes and use_mlp_features:
                name_parts.append(f"sf_{stream_fusion}")
        
        # Pair combination
        pair_combination = model_config.get('pair_combination', 'subtract')
        name_parts.append(f"pc_{pair_combination}")
        
        # Training details
        training_config = config.get('training', {})
        lr = training_config.get('learning_rate', 0.001)
        name_parts.append(f"lr{lr}")
        
        batch_size = config.get('data', {}).get('batch_size', 32)
        name_parts.append(f"bs{batch_size}")
        
        # Join with underscores
        experiment_name = "_".join(name_parts)
        
        return experiment_name
    
    def create_experiment_directory(
        self, 
        config: Dict[str, Any], 
        experiment_name: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> Path:
        """
        Create organized experiment directory.
        
        Args:
            config: Configuration dictionary
            experiment_name: Custom experiment name (auto-generated if None)
            timestamp: Custom timestamp (auto-generated if None)
            
        Returns:
            experiment_dir: Path to the experiment directory
        """
        if experiment_name is None:
            experiment_name = self.generate_experiment_name(config)
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure: logs/experiment_name/timestamp/
        experiment_dir = self.base_log_dir / experiment_name / timestamp
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration and metadata
        self.save_experiment_metadata(experiment_dir, config, experiment_name)
        
        return experiment_dir
    
    def save_experiment_metadata(
        self, 
        experiment_dir: Path, 
        config: Dict[str, Any], 
        experiment_name: str
    ):
        """
        Save experiment metadata and configuration.
        
        Args:
            experiment_dir: Experiment directory
            config: Configuration dictionary
            experiment_name: Experiment name
        """
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "description": self._generate_experiment_description(config)
        }
        
        # Save metadata
        with open(experiment_dir / "experiment_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save config separately for easy access
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save human-readable description
        with open(experiment_dir / "description.txt", 'w') as f:
            f.write(metadata["description"])
    
    def _generate_experiment_description(self, config: Dict[str, Any]) -> str:
        """
        Generate human-readable experiment description.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            description: Human-readable description
        """
        model_config = config.get('model', {})
        use_planes = model_config.get('use_planes', True)
        use_mlp_features = model_config.get('use_mlp_features', False)
        
        description_lines = []
        
        # Experiment type
        if use_planes and use_mlp_features:
            description_lines.append("Hybrid Classification Experiment")
            description_lines.append("- Uses both feature planes and MLP weight embeddings")
        elif use_mlp_features:
            description_lines.append("MLP-Only Classification Experiment")  
            description_lines.append("- Uses only MLP weight embeddings (no feature planes)")
        else:
            description_lines.append("Planes-Only Classification Experiment")
            description_lines.append("- Uses only feature planes (baseline)")
        
        description_lines.append("")
        
        # Model details
        if use_planes:
            encoder_type = model_config.get('encoder_type', 'resnet18')
            embedding_dim = model_config.get('embedding_dim', 256)
            plane_fusion = model_config.get('plane_fusion', 'concat')
            
            description_lines.append("Plane Encoding:")
            description_lines.append(f"- Encoder: {encoder_type}")
            description_lines.append(f"- Embedding dim: {embedding_dim}")
            description_lines.append(f"- Plane fusion: {plane_fusion}")
            description_lines.append("")
        
        if use_mlp_features:
            mlp_encoders = model_config.get('mlp_encoders', {})
            base_mlp = mlp_encoders.get('base_mlp', {})
            head_mlp = mlp_encoders.get('head_mlp', {})
            
            description_lines.append("MLP Encoding:")
            description_lines.append(f"- Base MLP embedding dim: {base_mlp.get('embedding_dim', 128)}")
            description_lines.append(f"- Head MLP embedding dim: {head_mlp.get('embedding_dim', 256)}")
            description_lines.append(f"- Pooling strategy: {mlp_encoders.get('pooling_strategy', 'max')}")
            
            fusion = model_config.get('fusion', {})
            description_lines.append(f"- Layer fusion: {fusion.get('layer_fusion', 'concat')}")
            if use_planes:
                description_lines.append(f"- Stream fusion: {fusion.get('stream_fusion', 'concat')}")
            description_lines.append("")
        
        # Training details
        training_config = config.get('training', {})
        data_config = config.get('data', {})
        
        description_lines.append("Training Configuration:")
        description_lines.append(f"- Learning rate: {training_config.get('learning_rate', 0.001)}")
        description_lines.append(f"- Batch size: {data_config.get('batch_size', 32)}")
        description_lines.append(f"- Epochs: {training_config.get('num_epochs', 50)}")
        description_lines.append(f"- Pair combination: {model_config.get('pair_combination', 'subtract')}")
        
        return "\n".join(description_lines)
    
    def list_experiments(self) -> Dict[str, list]:
        """
        List all experiments organized by type.
        
        Returns:
            experiments: Dictionary mapping experiment types to lists of runs
        """
        experiments = {}
        
        for exp_type_dir in self.base_log_dir.iterdir():
            if not exp_type_dir.is_dir():
                continue
            
            exp_type = exp_type_dir.name
            experiments[exp_type] = []
            
            for run_dir in exp_type_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # Check if it has experiment metadata
                metadata_file = run_dir / "experiment_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    experiments[exp_type].append({
                        "timestamp": run_dir.name,
                        "path": str(run_dir),
                        "description": metadata.get("description", "No description"),
                        "config": metadata.get("config", {})
                    })
        
        return experiments
    
    def find_experiment(self, experiment_name: str, timestamp: Optional[str] = None) -> Optional[Path]:
        """
        Find an experiment directory.
        
        Args:
            experiment_name: Name of the experiment
            timestamp: Specific timestamp (finds latest if None)
            
        Returns:
            experiment_dir: Path to experiment directory if found
        """
        exp_type_dir = self.base_log_dir / experiment_name
        
        if not exp_type_dir.exists():
            return None
        
        if timestamp is not None:
            exp_dir = exp_type_dir / timestamp
            return exp_dir if exp_dir.exists() else None
        else:
            # Find latest timestamp
            timestamps = [d.name for d in exp_type_dir.iterdir() if d.is_dir()]
            if not timestamps:
                return None
            
            latest_timestamp = max(timestamps)
            return exp_type_dir / latest_timestamp