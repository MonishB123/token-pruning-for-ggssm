"""
Comprehensive Distance Metric Benchmarking for GraphSSM on ETT Dataset

This script integrates with eval_forecasting.py to benchmark different distance metrics
on actual time series forecasting tasks with full metrics, performance profiling, and visualization.

Usage:
    python benchmark_et_distances.py --data ETTm1 --seq_len 96 --pred_len 24
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for terminal
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Try to import terminal display libraries
try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Install with: pip install Pillow")

try:
    # For displaying images in Jupyter/IPython terminals
    from IPython.display import Image as IPImage, display
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

# Add paths
gg_ssms_path = os.path.expanduser("/workspace")
mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")
sys.path.append(mamba_ts_path)
sys.path.append(os.path.join(gg_ssms_path, "core", "graph_ssm"))

from data_provider.data_factory import data_provider
from utils.tools import set_seed
from main import GraphSSM


class TimeSeriesForecaster(nn.Module):
    """Time series forecaster with configurable distance metric"""
    def __init__(
        self,
        enc_in: int,
        c_out: int,
        seq_len: int,
        pred_len: int,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        distance_metric: str = 'cosine',
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Core GraphSSM with specified distance metric
        self.graph_ssm = GraphSSM(
            d_model=d_model, 
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            distance_metric=distance_metric  # KEY ADDITION
        )
        
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        b, seq_len, enc_in = x_enc.shape
        
        # Normalize
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        
        # Embed and process
        embedded = self.input_embedding(x_enc)
        context_len = min(seq_len, 4)
        processed = self.graph_ssm(embedded, context_len)
        
        # Extract predictions
        if processed.shape[1] >= self.pred_len:
            processed = processed[:, -self.pred_len:, :]
        else:
            last_timestep = processed[:, -1:, :].repeat(1, self.pred_len, 1)
            processed = last_timestep
        
        # Project and denormalize
        output = self.output_projection(processed)
        output = output * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        output = output + (means[:, [0], :].repeat(1, self.pred_len, 1))
        
        return output


def estimate_flops(model: nn.Module, batch_size: int, seq_len: int, d_model: int) -> int:
    """Estimate FLOPs for a forward pass"""
    # Simplified FLOP estimation
    flops = 0
    
    # Input embedding: batch_size * seq_len * enc_in * d_model * 2
    flops += batch_size * seq_len * model.enc_in * d_model * 2
    
    # GraphSSM (approximate): batch_size * seq_len * d_model * d_model * 4
    flops += batch_size * seq_len * d_model * d_model * 4
    
    # Output projection: batch_size * pred_len * d_model * c_out * 2
    flops += batch_size * model.pred_len * d_model * model.c_out * 2
    
    return flops


def evaluate_with_profiling(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
    distance_metric: str
) -> Dict:
    """Comprehensive evaluation with profiling"""
    model.eval()
    
    # Metrics storage
    total_loss = 0.0
    all_preds = []
    all_trues = []
    batch_times = []
    
    # FLOP estimation
    first_batch = next(iter(loader))
    batch_size = first_batch[0].shape[0]
    seq_len = first_batch[0].shape[1]
    d_model = model.d_model
    flops_per_batch = estimate_flops(model, batch_size, seq_len, d_model)
    total_flops = 0
    
    # Timing
    start_time = time.time()
    
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
            batch_start = time.time()
            
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()
            
            dec_inp = torch.zeros_like(batch_y).float().to(device)
            
            # Synchronize for accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            batch_end = time.time()
            batch_times.append(batch_end - batch_start)
            
            # Trim batch_y if needed
            if batch_y.shape[1] > outputs.shape[1]:
                batch_y = batch_y[:, -outputs.shape[1]:, :]
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            
            total_flops += flops_per_batch
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Compute metrics
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-8))) * 100
    
    # Performance metrics
    avg_batch_time = np.mean(batch_times)
    tflops = (total_flops / total_time) / 1e12
    
    return {
        'distance_metric': distance_metric,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
        'total_loss': total_loss / len(loader.dataset),
        'total_time': total_time,
        'avg_batch_time': avg_batch_time,
        'tflops': tflops,
        'num_batches': len(loader),
        'batch_times': batch_times,
    }


def run_benchmark_for_metric(
    distance_metric: str,
    args: argparse.Namespace,
    test_loader: DataLoader,
    device: torch.device
) -> Dict:
    """Run complete benchmark for a single distance metric"""
    
    print(f"\n{'='*70}")
    print(f"Benchmarking Distance Metric: {distance_metric.upper()}")
    print(f"{'='*70}")
    
    # Create model with this distance metric
    model = TimeSeriesForecaster(
        enc_in=args.enc_in,
        c_out=args.c_out,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        distance_metric=distance_metric
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    criterion = nn.MSELoss()
    
    # Run evaluation
    results = evaluate_with_profiling(model, test_loader, device, criterion, distance_metric)
    
    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS - {distance_metric.upper()}")
    print(f"{'='*70}")
    print(f"Test Loss (MSE): {results['total_loss']:.6f}")
    print(f"MAE:  {results['mae']:.6f}")
    print(f"MSE:  {results['mse']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"MAPE: {results['mape']:.6f}%")
    print(f"{'='*70}")
    print(f"PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"Total time: {results['total_time']:.4f}s")
    print(f"Average batch time: {results['avg_batch_time']:.6f}s")
    print(f"Batches processed: {results['num_batches']}")
    print(f"**TFLOPS: {results['tflops']:.2f}**")
    print(f"{'='*70}")
    
    return results


def create_comprehensive_visualization(all_results: List[Dict], args: argparse.Namespace):
    """Create comprehensive multi-panel visualization"""
    
    metrics = [r['distance_metric'] for r in all_results]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Color scheme
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    
    # 1. Forecasting Accuracy Metrics (MAE, MSE, RMSE, MAPE)
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(metrics))
    width = 0.2
    
    mae_vals = [r['mae'] for r in all_results]
    rmse_vals = [r['rmse'] for r in all_results]
    mape_vals = [r['mape'] for r in all_results]
    
    ax1.bar(x - width, mae_vals, width, label='MAE', alpha=0.8)
    ax1.bar(x, rmse_vals, width, label='RMSE', alpha=0.8)
    ax1.bar(x + width, mape_vals, width, label='MAPE (%)', alpha=0.8)
    
    ax1.set_ylabel('Error', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Distance Metric', fontsize=11, fontweight='bold')
    ax1.set_title('Forecasting Accuracy Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Inference Speed (Total Time)
    ax2 = fig.add_subplot(gs[0, 2])
    times = [r['total_time'] for r in all_results]
    bars = ax2.barh(metrics, times, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax2.set_title('Total Inference Time', fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, time_val) in enumerate(zip(bars, times)):
        ax2.text(time_val, i, f' {time_val:.2f}s', va='center', fontsize=9)
    
    # 3. TFLOPS Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    tflops = [r['tflops'] for r in all_results]
    bars3 = ax3.bar(metrics, tflops, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('TFLOPS', fontsize=11, fontweight='bold')
    ax3.set_title('Computational Throughput', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars3, tflops):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 4. Batch Time Distribution (Box Plot)
    ax4 = fig.add_subplot(gs[1, 1])
    batch_times_data = [np.array(r['batch_times']) * 1000 for r in all_results]  # Convert to ms
    bp = ax4.boxplot(batch_times_data, labels=metrics, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_ylabel('Batch Time (ms)', fontsize=11, fontweight='bold')
    ax4.set_title('Batch Processing Time Distribution', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Speed Ranking
    ax5 = fig.add_subplot(gs[1, 2])
    sorted_by_speed = sorted(all_results, key=lambda x: x['total_time'])
    sorted_metrics = [r['distance_metric'] for r in sorted_by_speed]
    sorted_times = [r['total_time'] for r in sorted_by_speed]
    speedups = [sorted_times[0] / t for t in sorted_times]
    
    y_pos = np.arange(len(sorted_metrics))
    bars5 = ax5.barh(y_pos, speedups, color=colors, alpha=0.8, edgecolor='black')
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(sorted_metrics)
    ax5.set_xlabel('Relative Speed', fontsize=11, fontweight='bold')
    ax5.set_title('Speed Ranking (vs Fastest)', fontsize=12, fontweight='bold')
    ax5.axvline(x=1.0, color='red', linestyle='--', linewidth=2)
    ax5.grid(axis='x', alpha=0.3)
    
    for i, (bar, speedup) in enumerate(zip(bars5, speedups)):
        ax5.text(speedup, i, f' {speedup:.2f}x', va='center', fontsize=9)
    
    # 6. Accuracy Ranking
    ax6 = fig.add_subplot(gs[2, 0])
    sorted_by_mae = sorted(all_results, key=lambda x: x['mae'])
    sorted_metrics_acc = [r['distance_metric'] for r in sorted_by_mae]
    sorted_mae = [r['mae'] for r in sorted_by_mae]
    
    y_pos = np.arange(len(sorted_metrics_acc))
    bars6 = ax6.barh(y_pos, sorted_mae, color=colors, alpha=0.8, edgecolor='black')
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(sorted_metrics_acc)
    ax6.set_xlabel('MAE', fontsize=11, fontweight='bold')
    ax6.set_title('Accuracy Ranking (by MAE)', fontsize=12, fontweight='bold')
    ax6.grid(axis='x', alpha=0.3)
    
    # 7. Summary Table
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('tight')
    ax7.axis('off')
    
    table_data = []
    for r in all_results:
        table_data.append([
            r['distance_metric'],
            f"{r['mae']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['mape']:.2f}",
            f"{r['total_time']:.2f}",
            f"{r['tflops']:.2f}"
        ])
    
    table = ax7.table(
        cellText=table_data,
        colLabels=['Metric', 'MAE', 'RMSE', 'MAPE(%)', 'Time(s)', 'TFLOPS'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4A90E2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_mae_idx = min(range(len(all_results)), key=lambda i: all_results[i]['mae'])
    best_speed_idx = min(range(len(all_results)), key=lambda i: all_results[i]['total_time'])
    
    for i in range(1, len(table_data) + 1):
        if i - 1 == best_mae_idx:
            table[(i, 1)].set_facecolor('#90EE90')
            table[(i, 1)].set_text_props(weight='bold')
        if i - 1 == best_speed_idx:
            table[(i, 4)].set_facecolor('#90EE90')
            table[(i, 4)].set_text_props(weight='bold')
    
    ax7.set_title('Summary Statistics', fontsize=13, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle(
        f'GraphSSM Distance Metric Comprehensive Benchmark\n'
        f'Dataset: {args.data} | Seq: {args.seq_len} | Pred: {args.pred_len} | Model: d_model={args.d_model}',
        fontsize=15, fontweight='bold', y=0.98
    )
    
    # Save
    output_path = os.path.join(args.output_dir, f'comprehensive_benchmark_{args.data}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
    pdf_path = os.path.join(args.output_dir, f'comprehensive_benchmark_{args.data}.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")
    
    # Display in terminal if possible
    display_plot_in_terminal(output_path, results, args)
    
    plt.close()


def display_plot_in_terminal(image_path: str, all_results: List[Dict], args: argparse.Namespace):
    """Display plot in terminal using various methods"""
    
    print("\n" + "="*70)
    print("DISPLAYING PLOT IN TERMINAL")
    print("="*70)
    
    # Method 1: Try IPython display (works in Jupyter terminals)
    if HAS_IPYTHON:
        try:
            print("Attempting to display via IPython...")
            display(IPImage(filename=image_path))
            print("✓ Plot displayed via IPython")
            return
        except Exception as e:
            print(f"IPython display failed: {e}")
    
    # Method 2: Try ASCII art conversion using plotext
    try:
        import plotext as plt_term
        print("\nGenerating terminal-friendly plot with plotext...")
        create_terminal_plot(all_results, plt_term)
        return
    except ImportError:
        print("plotext not available. Install with: pip install plotext")
    except Exception as e:
        print(f"plotext display failed: {e}")
    
    # Method 3: Display file path and instructions
    print("\nPlot generated but cannot display inline in this terminal.")
    print(f"\nPlot saved to: {image_path}")
    print("\nTo view the plot, use one of these methods:")
    print("1. Download the file via RunPod's file browser")
    print("2. Use RunPod's web-based file viewer")
    print("3. Install plotext for terminal plots: pip install plotext")
    print(f"4. Use: python -c \"from PIL import Image; Image.open('{image_path}').show()\"")
    print("="*70)


def create_terminal_plot(all_results: List[Dict], plt_term):
    """Create ASCII plots for terminal display"""
    
    metrics = [r['distance_metric'] for r in all_results]
    
    # Plot 1: Inference Time Comparison
    print("\n" + "="*70)
    print("INFERENCE TIME COMPARISON (Terminal View)")
    print("="*70)
    times = [r['total_time'] for r in all_results]
    plt_term.clear_figure()
    plt_term.bar(metrics, times)
    plt_term.title("Total Inference Time by Distance Metric")
    plt_term.xlabel("Distance Metric")
    plt_term.ylabel("Time (seconds)")
    plt_term.show()
    
    # Plot 2: MAE Comparison
    print("\n" + "="*70)
    print("FORECASTING ACCURACY (MAE) - Terminal View")
    print("="*70)
    mae_vals = [r['mae'] for r in all_results]
    plt_term.clear_figure()
    plt_term.bar(metrics, mae_vals)
    plt_term.title("Mean Absolute Error by Distance Metric")
    plt_term.xlabel("Distance Metric")
    plt_term.ylabel("MAE")
    plt_term.show()
    
    # Plot 3: TFLOPS Comparison
    print("\n" + "="*70)
    print("COMPUTATIONAL THROUGHPUT - Terminal View")
    print("="*70)
    tflops = [r['tflops'] for r in all_results]
    plt_term.clear_figure()
    plt_term.bar(metrics, tflops)
    plt_term.title("TFLOPS by Distance Metric")
    plt_term.xlabel("Distance Metric")
    plt_term.ylabel("TFLOPS")
    plt_term.show()
    
    print("\n✓ Terminal plots displayed successfully!")


def print_ascii_comparison_table(all_results: List[Dict]):
    """Print a detailed ASCII table comparison"""
    
    print("\n" + "="*100)
    print("DETAILED COMPARISON TABLE")
    print("="*100)
    
    # Header
    print(f"{'Metric':<12} | {'MAE':<10} | {'RMSE':<10} | {'MAPE(%)':<10} | {'Time(s)':<10} | {'TFLOPS':<10} | {'Rank':<6}")
    print("-"*100)
    
    # Sort by speed
    sorted_results = sorted(all_results, key=lambda x: x['total_time'])
    
    for i, r in enumerate(sorted_results, 1):
        print(f"{r['distance_metric']:<12} | "
              f"{r['mae']:<10.6f} | "
              f"{r['rmse']:<10.6f} | "
              f"{r['mape']:<10.2f} | "
              f"{r['total_time']:<10.2f} | "
              f"{r['tflops']:<10.2f} | "
              f"#{i:<5}")
    
    print("="*100)
    
    # Highlight winners
    best_speed = min(all_results, key=lambda x: x['total_time'])
    best_acc = min(all_results, key=lambda x: x['mae'])
    best_tflops = max(all_results, key=lambda x: x['tflops'])
    
    print(f"\n🏆 FASTEST:         {best_speed['distance_metric']:<12} ({best_speed['total_time']:.2f}s)")
    print(f"🎯 MOST ACCURATE:  {best_acc['distance_metric']:<12} (MAE: {best_acc['mae']:.6f})")
    print(f"⚡ HIGHEST TFLOPS: {best_tflops['distance_metric']:<12} ({best_tflops['tflops']:.2f} TFLOPS)")
    print("="*100)


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Distance Metric Benchmark on ETT Dataset')
    
    # Data config
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default=os.path.expanduser("/data/eval_pipelines/datasets/ETT-small"))
    parser.add_argument("--data_path", type=str, default="ETTm1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="t")
    parser.add_argument("--checkpoints", type=str, default=os.path.expanduser("~/data/checkpoints/"))
    
    # Task config
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=24)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)
    
    # Model config
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--c_out", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_state", type=int, default=16)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    
    # Benchmark config
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3047)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--metrics", type=str, nargs='+', 
                       default=['cosine', 'euclidean', 'gaussian', 'manhattan', 'norm2'])
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./benchmark_results")
    parser.add_argument("--no_display", action="store_true")
    
    # Required for data_provider
    parser.add_argument("--task_name", type=str, default="long_term_forecast")
    parser.add_argument("--is_training", type=int, default=0)
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="GraphSSM")
    parser.add_argument("--embed", type=str, default="timeF")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print("=" * 70)
    print("GraphSSM Distance Metric Comprehensive Benchmark")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset: {args.data}")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Distance metrics: {', '.join(args.metrics)}")
    print("=" * 70)
    
    # Load test data
    print("\nLoading test data...")
    test_data, test_loader = data_provider(args, flag='test')
    print(f"Test dataset loaded: {len(test_data)} samples, {len(test_loader)} batches")
    
    # Run benchmarks
    all_results = []
    for i, metric in enumerate(args.metrics, 1):
        print(f"\n[{i}/{len(args.metrics)}] Processing: {metric}")
        try:
            results = run_benchmark_for_metric(metric, args, test_loader, device)
            all_results.append(results)
        except Exception as e:
            print(f"ERROR benchmarking {metric}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo successful benchmarks. Exiting.")
        return
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, f'benchmark_results_{args.data}.csv')
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Print ASCII comparison table
    print_ascii_comparison_table(all_results)
    
    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    best_acc = min(all_results, key=lambda x: x['mae'])
    best_speed = min(all_results, key=lambda x: x['total_time'])
    
    print(f"BEST ACCURACY: {best_acc['distance_metric']} (MAE: {best_acc['mae']:.6f})")
    print(f"FASTEST:       {best_speed['distance_metric']} (Time: {best_speed['total_time']:.2f}s)")
    
    # Create visualization
    print("\nGenerating comprehensive visualization...")
    create_comprehensive_visualization(all_results, args)
    
    print("\n✅ Benchmark complete!")
    print(f"\nAll files saved to: {args.output_dir}/")
    print(f"- PNG visualization: comprehensive_benchmark_{args.data}.png")
    print(f"- PDF visualization: comprehensive_benchmark_{args.data}.pdf")
    print(f"- CSV data: benchmark_results_{args.data}.csv")


if __name__ == "__main__":
    main()