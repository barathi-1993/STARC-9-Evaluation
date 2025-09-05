import os
import subprocess
import argparse
import pandas as pd
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import glob
import time
from tqdm.auto import tqdm
import warnings 
warnings.filterwarnings("error")

from config import SAVE_DIR, METRICS, MODEL_PATHS


# List of all models to benchmark
ALL_MODELS = [
    # Base models
     'resnet50', 'vit_base',
     'swin_base', 'convnext', 'efficientnet', 'deit', 'kimianet',
    # Foundation models
     'hipt', 'transpath', 'pathdino', 'vim4path', 'conch', 'virchow', 
     'uni', 'prov_gigapath',
    # Custom models
     'custom_cnn', 'custom_histovit'
]

def run_benchmark(models=None, data_path=None, batch_size=32, epochs=10, lr=0.0001, seed=42, save_dir=SAVE_DIR, multi_gpu=False):
    """Run benchmark for selected models without interruption on error/warning"""
    if models is None:
        models = ALL_MODELS
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = os.path.join(save_dir, f"benchmark_{timestamp}")
    os.makedirs(benchmark_dir, exist_ok=True)
    
    # Initialize results dataframe
    results = []
    
    # Summary metrics CSV
    summary_path = os.path.join(benchmark_dir, "benchmark_summary.csv")
    summary_columns = [
        'model', 'accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro',
        'total_time', 'avg_epoch_time', 'params_total', 'params_trainable'
    ]
    
    # Write CSV header
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(summary_columns)
    
    # Iterate through models with a master loop
    for model_name in models:
        print(f"\n{'='*80}")
        print(f"Starting benchmark for {model_name}")
        print(f"{'='*80}")
        
        model_start_time = time.time()
        
        # Build command to launch training (assuming main.py handles one model training)
        cmd = [
            "python", "main.py",
            "--model", model_name,
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--seed", str(seed),
            "--save_dir", os.path.join(benchmark_dir, model_name)
        ]
        
        if multi_gpu:
            cmd.append("--multi_gpu")
        if data_path:
            cmd.extend(["--data_path", data_path])
        
        # Wrap the subprocess call in try/except so that errors or warnings (if filtered as errors)
        # do not interrupt the overall benchmark.
        try:
            subprocess.run(cmd, check=True)
            
            # Load results after training completes
            results_path = os.path.join(benchmark_dir, model_name, f"{model_name}_results.json")
            with open(results_path, 'r') as f:
                model_results = json.load(f)
            
            # Load epoch metrics to compute training time
            epoch_metrics_path = glob.glob(os.path.join(benchmark_dir, model_name, f"epoch_metrics_{model_name}*.csv"))
            if epoch_metrics_path:
                epoch_df = pd.read_csv(epoch_metrics_path[0])
                total_time = epoch_df['epoch_time'].sum()
                avg_epoch_time = epoch_df['epoch_time'].mean()
            else:
                total_time = time.time() - model_start_time
                avg_epoch_time = total_time / epochs
            
            try:
                with open(os.path.join(benchmark_dir, model_name, f"{model_name}_params.json"), 'r') as f:
                    params_info = json.load(f)
                params_total = params_info.get('total_params', 0)
                params_trainable = params_info.get('trainable_params', 0)
            except:
                params_total = 0
                params_trainable = 0
            
            # Format total training time
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
            
            # Prepare result entry
            result_entry = {
                'model': model_name,
                'accuracy': model_results['accuracy'],
                'precision': model_results['precision_macro'],
                'recall': model_results['recall_macro'],
                'f1_macro': model_results['f1_macro'],
                'f1_micro': model_results['f1_micro'],
                'total_time': total_time,
                'avg_epoch_time': avg_epoch_time,
                'params_total': params_total,
                'params_trainable': params_trainable
            }
            results.append(result_entry)
            
            # Write summary entry to CSV
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    model_name,
                    model_results['accuracy'],
                    model_results['precision_macro'],
                    model_results['recall_macro'],
                    model_results['f1_macro'],
                    model_results['f1_micro'],
                    total_time,
                    avg_epoch_time,
                    params_total,
                    params_trainable
                ])
            print(f"\nSuccessfully benchmarked {model_name}")
            print(f"Accuracy: {model_results['accuracy']:.4f}")
            print(f"F1 Macro: {model_results['f1_macro']:.4f}")
            print(f"Training time: {time_str}")
        except Exception as e:
            print(f"Error encountered while benchmarking {model_name}: {e}")
            print("Skipping to next model...\n")
            continue
    
    # Save complete results
    results_df = pd.DataFrame(results)
    complete_results_path = os.path.join(benchmark_dir, "benchmark_complete_results.csv")
    results_df.to_csv(complete_results_path, index=False)
    print(f"\nBenchmark results saved to {complete_results_path}")
    
    # Optionally, call your plotting function for summary results here
    plot_benchmark_results(results_df, benchmark_dir)
    
    return results_df, benchmark_dir

def plot_benchmark_results(results_df, output_dir):
    """Plot benchmark results"""
    if len(results_df) == 0:
        print("No results to plot.")
        return
        
    sorted_df = results_df.sort_values('f1_macro', ascending=False)
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='model', y='f1_macro', data=sorted_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Models Comparison by F1 Macro Score')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "models_f1_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    if len(sorted_df) > 1:
        plt.figure(figsize=(12, 8))
        metrics_df = sorted_df.set_index('model')[['accuracy', 'precision', 'recall', 'f1_macro', 'f1_micro']]
        sns.heatmap(metrics_df, annot=True, cmap='YlGnBu', fmt='.4f')
        plt.title('Performance Metrics Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "metrics_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(12, 8))
    time_df = sorted_df.sort_values('total_time')
    sns.barplot(x='model', y='total_time', data=time_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Training Time Comparison (seconds)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "training_time_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(12, 8))
    param_df = sorted_df.sort_values('params_total', ascending=False)
    if 'params_total' in param_df.columns and not param_df['params_total'].isnull().all():
        sns.barplot(x='model', y='params_total', data=param_df)
        plt.xticks(rotation=45, ha='right')
        plt.title('Model Size Comparison (parameters)')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "parameter_count_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    if len(sorted_df) >= 3:
        plt.figure(figsize=(10, 8))
        plt.scatter(sorted_df['total_time'], sorted_df['f1_macro'], s=100, alpha=0.7)
        for i, row in sorted_df.iterrows():
            plt.annotate(row['model'], (row['total_time'], row['f1_macro']), xytext=(7, 0), textcoords='offset points', fontsize=9)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('F1 Macro Score')
        plt.title('Performance vs. Training Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "performance_vs_time.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {plots_dir}")

def main():
    parser = argparse.ArgumentParser(description='Run CRC classification benchmark')
    parser.add_argument('--models', nargs='+', help='Models to benchmark (space separated)')
    parser.add_argument('--data_path', type=str, default=None, help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR, help='Directory to save outputs')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs for training')
    args = parser.parse_args()
    
    # Run benchmark
    if args.models:
        for model in args.models:
            if model not in ALL_MODELS:
                print(f"Warning: {model} is not in the list of available models")
        models = args.models
    else:
        models = ALL_MODELS
    
    results_df, benchmark_dir = run_benchmark(
        models=models,
        data_path=args.data_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        multi_gpu=args.multi_gpu
    )
    
    if len(results_df) >= 3:
        top3 = results_df.sort_values('f1_macro', ascending=False).head(3)
        print("\nTop 3 performing models:")
        for i, (_, row) in enumerate(top3.iterrows()):
            print(f"{i+1}. {row['model']}: F1 Macro = {row['f1_macro']:.4f}, Accuracy = {row['accuracy']:.4f}")

if __name__ == "__main__":
    main()
