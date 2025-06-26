#!/usr/bin/env python3

import sys
sys.path.append('.')

import json
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from training_utils.experiment_manager import ExperimentManager


def load_tensorboard_metrics(experiment_dir: Path) -> Dict:
    metrics = {}
    
    events_files = list(experiment_dir.glob("**/events.out.tfevents.*"))
    if not events_files:
        return metrics
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        ea = EventAccumulator(str(experiment_dir))
        ea.Reload()
        
        scalar_tags = ea.Tags()['scalars']
        
        for tag in scalar_tags:
            scalar_events = ea.Scalars(tag)
            values = [event.value for event in scalar_events]
            steps = [event.step for event in scalar_events]
            
            if values:
                metrics[tag] = {
                    'final_value': values[-1],
                    'max_value': max(values),
                    'values': values,
                    'steps': steps
                }
    except ImportError:
        print("TensorBoard not available for metric extraction")
    
    return metrics


def extract_best_metrics(experiment_dir: Path) -> Dict:
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return {}
    
    best_model_path = checkpoint_dir / "best_model.pth"
    if not best_model_path.exists():
        return {}
    
    try:
        import torch
        checkpoint = torch.load(best_model_path, map_location='cpu')
        return checkpoint.get('metrics', {})
    except:
        return {}


def compare_experiments(log_dir: str = "./logs", output_dir: str = "./comparison_results"):
    exp_manager = ExperimentManager(log_dir)
    experiments = exp_manager.list_experiments()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for exp_type, runs in experiments.items():
        print(f"\nProcessing experiment type: {exp_type}")
        
        for run in runs:
            run_path = Path(run["path"])
            
            config = run["config"]
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            
            tb_metrics = load_tensorboard_metrics(run_path)
            best_metrics = extract_best_metrics(run_path)
            
            result = {
                "experiment_type": exp_type,
                "timestamp": run["timestamp"],
                "use_planes": model_config.get("use_planes", True),
                "use_mlp_features": model_config.get("use_mlp_features", False),
                "encoder_type": model_config.get("encoder_type", "resnet18"),
                "plane_fusion": model_config.get("plane_fusion", "concat"),
                "pair_combination": model_config.get("pair_combination", "subtract"),
                "learning_rate": training_config.get("learning_rate", 0.001),
                "batch_size": config.get("data", {}).get("batch_size", 16),
                "num_epochs": training_config.get("num_epochs", 50),
            }
            
            if model_config.get("use_mlp_features", False):
                mlp_config = model_config.get("mlp_encoders", {})
                fusion_config = model_config.get("fusion", {})
                
                result.update({
                    "pooling_strategy": mlp_config.get("pooling_strategy", "max"),
                    "layer_fusion": fusion_config.get("layer_fusion", "concat"),
                    "stream_fusion": fusion_config.get("stream_fusion", "concat"),
                    "base_mlp_dim": mlp_config.get("base_mlp", {}).get("embedding_dim", 128),
                    "head_mlp_dim": mlp_config.get("head_mlp", {}).get("embedding_dim", 256),
                })
            
            if 'val/accuracy' in tb_metrics:
                result["final_val_accuracy"] = tb_metrics['val/accuracy']['final_value']
                result["max_val_accuracy"] = tb_metrics['val/accuracy']['max_value']
            
            if 'train_epoch/accuracy' in tb_metrics:
                result["final_train_accuracy"] = tb_metrics['train_epoch/accuracy']['final_value']
            
            if 'val/loss' in tb_metrics:
                result["final_val_loss"] = tb_metrics['val/loss']['final_value']
            
            if best_metrics:
                result["best_val_accuracy"] = best_metrics.get("accuracy", None)
                result["best_val_loss"] = best_metrics.get("loss", None)
            
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    
    if df.empty:
        print("No experiment results found")
        return
    
    df.to_csv(output_path / "experiment_comparison.csv", index=False)
    
    print(f"\nFound {len(df)} experiment runs across {len(experiments)} experiment types")
    
    print("\nSummary by experiment type:")
    summary = df.groupby("experiment_type").agg({
        "max_val_accuracy": ["count", "mean", "std", "max"],
        "final_val_loss": ["mean", "std", "min"]
    }).round(4)
    
    print(summary)
    
    create_comparison_plots(df, output_path)
    
    print(f"\nResults saved to {output_path}")


def create_comparison_plots(df: pd.DataFrame, output_path: Path):
    plt.style.use('default')
    
    if 'max_val_accuracy' in df.columns and df['max_val_accuracy'].notna().any():
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='experiment_type', y='max_val_accuracy')
        plt.title('Validation Accuracy by Experiment Type')
        plt.xticks(rotation=45)
        plt.ylabel('Max Validation Accuracy')
        
        plt.subplot(1, 2, 2)
        if 'final_val_loss' in df.columns:
            sns.boxplot(data=df, x='experiment_type', y='final_val_loss')
            plt.title('Final Validation Loss by Experiment Type')
            plt.xticks(rotation=45)
            plt.ylabel('Final Validation Loss')
        
        plt.tight_layout()
        plt.savefig(output_path / 'experiment_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if len(df['experiment_type'].unique()) > 1:
        comparison_table = df.groupby('experiment_type').agg({
            'max_val_accuracy': ['mean', 'std', 'count'],
            'final_val_loss': ['mean', 'std']
        }).round(4)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = []
        for exp_type in comparison_table.index:
            row = [
                exp_type,
                f"{comparison_table.loc[exp_type, ('max_val_accuracy', 'mean')]:.3f} ± {comparison_table.loc[exp_type, ('max_val_accuracy', 'std')]:.3f}",
                f"{comparison_table.loc[exp_type, ('final_val_loss', 'mean')]:.3f} ± {comparison_table.loc[exp_type, ('final_val_loss', 'std')]:.3f}",
                f"{comparison_table.loc[exp_type, ('max_val_accuracy', 'count')]}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Experiment Type', 'Val Accuracy (mean ± std)', 'Val Loss (mean ± std)', 'Runs'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        plt.title('Experiment Comparison Summary', pad=20)
        plt.savefig(output_path / 'summary_table.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare experiment results')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--output_dir', type=str, default='./comparison_results')
    parser.add_argument('--experiment_type', type=str, default=None)
    args = parser.parse_args()
    
    compare_experiments(args.log_dir, args.output_dir)


if __name__ == '__main__':
    main()