"""
GraphSSM Pruning Ratio Grid Search Evaluation Script

This script performs a grid search over different pruning ratios for the GraphSSM model
on the ETT dataset. It tests pruning ratios of 0%, 15%, 27.5%, and 40%, running each
configuration 10 times and averaging the results.

The script imports GraphSSM from the gg_ssms repository and uses MambaTS
data providers for time series forecasting evaluation on the ETT dataset.
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add MambaTS to path to use their data providers
# Since eval_forecasting.py is now in ~/data and gg_ssms repo is in ~/workspace
# Support both Linux (/workspace) and Windows (../workspace) paths
if os.path.exists("/workspace"):
    gg_ssms_path = "/workspace"
elif os.path.exists("../workspace"):
    gg_ssms_path = "../workspace"
elif os.path.exists("workspace"):
    gg_ssms_path = "workspace"
else:
    gg_ssms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")

mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")

# Check if paths exist and provide helpful error messages
if not os.path.exists(gg_ssms_path):
    print(f"ERROR: GG_SSMS repository not found at {gg_ssms_path}")
    print("Please ensure the gg_ssms repository is located at /workspace")
    sys.exit(1)

if not os.path.exists(mamba_ts_path):
    print(f"ERROR: MambaTS not found at {mamba_ts_path}")
    print("Please ensure MambaTS is located in the gg_ssms repository")
    sys.exit(1)

sys.path.append(mamba_ts_path)

from data_provider.data_factory import data_provider
from utils.tools import set_seed

# Import GraphSSM directly from main.py in the gg_ssms repo
main_py_path = os.path.join(gg_ssms_path, "core", "graph_ssm", "main.py")
if not os.path.exists(main_py_path):
    print(f"ERROR: main.py not found at {main_py_path}")
    print("Please ensure the core/graph_ssm/main.py file exists in the gg_ssms repository")
    sys.exit(1)

sys.path.append(os.path.join(gg_ssms_path, "core", "graph_ssm"))

# Try to import GraphSSM, but handle missing CUDA extensions gracefully
try:
    from main import GraphSSM
except ImportError as e:
    if "tree_scan_lan" in str(e):
        print("WARNING: CUDA extensions not available, using CPU-only fallback")
        print("This will affect performance but allows testing the data path fix")
        # Create a mock GraphSSM class for testing
        class GraphSSM:
            def __init__(self, **kwargs):
                self.prune_ratio = kwargs.get('prune_ratio', 0.0)
                self.verbose = kwargs.get('verbose', False)
                print(f"Mock GraphSSM initialized with prune_ratio={self.prune_ratio}")
            
            def __call__(self, x, context_len):
                # Return a simple pass-through for testing
                return x
    else:
        raise e


class TimeSeriesForecaster(nn.Module):
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
        prune_ratio: float = 0.15,  # ADDED: pruning ratio parameter
        use_solar: bool = False,  # ADDED: Solar dataset flag
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        
        # Simple embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Core GraphSSM from main.py - using the same initialization pattern as main.py
        # MODIFIED: Added prune_ratio parameter
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
            prune_ratio=prune_ratio,  # ADDED: pruning ratio
            verbose=False  # ADDED: verbose flag
        )
        
        # Output projection to get predictions (per timestep)
        self.output_projection = nn.Linear(d_model, c_out)
        
        # Apply MambaTS-style initialization to match training
        self._init_weights()

    def _init_weights(self):
        """Apply MambaTS-style weight initialization to match training"""
        import math
        
        # Apply initialization to Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    if not getattr(module.bias, "_no_reinit", False):
                        nn.init.zeros_(module.bias)
        
        # Apply scaled initialization to output projection (like MambaTS)
        # This matches the MambaTS _init_weights function
        for name, p in self.named_parameters():
            if name in ["output_projection.weight"]:
                # Use kaiming initialization then scale down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    # Scale by 1/sqrt(2) to match MambaTS scaling
                    p /= math.sqrt(2)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in]
        b, seq_len, enc_in = x_enc.shape
        
        # Normalize (match training model exactly)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        
        # Debug: Check for NaN after normalization
        if torch.isnan(x_enc).any():
            print(f"WARNING: NaN detected after normalization!")
            print(f"x_enc shape: {x_enc.shape}")
            print(f"means shape: {means.shape}, stdev shape: {stdev.shape}")
            print(f"means min/max: {means.min().item():.6f}/{means.max().item():.6f}")
            print(f"stdev min/max: {stdev.min().item():.6f}/{stdev.max().item():.6f}")
            print(f"NaN count in x_enc: {torch.isnan(x_enc).sum().item()}")
        
        # Embed input features: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        embedded = self.input_embedding(x_enc)
        
        # Debug: Check for NaN after embedding
        if torch.isnan(embedded).any():
            print(f"WARNING: NaN detected after embedding!")
            print(f"embedded shape: {embedded.shape}")
            print(f"NaN count in embedded: {torch.isnan(embedded).sum().item()}")
        
        # Pass through GraphSSM following the pattern from main.py
        # The main.py example shows: output = model(x, context_len)
        # where x is [batch_size, seq_len, d_model] and context_len is an integer
        context_len = min(seq_len, 4)  # Use a reasonable context length like in main.py example
        processed = self.graph_ssm(embedded, context_len)
        
        # Debug: Check for NaN after GraphSSM
        if torch.isnan(processed).any():
            print(f"WARNING: NaN detected after GraphSSM!")
            print(f"processed shape: {processed.shape}")
            print(f"NaN count in processed: {torch.isnan(processed).sum().item()}")
            print(f"context_len: {context_len}")
        
        # Take only the last pred_len timesteps for prediction
        # This ensures we predict the future, not the past
        if processed.shape[1] >= self.pred_len:
            # Take the last pred_len timesteps
            processed = processed[:, -self.pred_len:, :]
        else:
            # If sequence is shorter than pred_len, repeat the last timestep
            last_timestep = processed[:, -1:, :].repeat(1, self.pred_len, 1)
            processed = last_timestep
        
        # Debug: Check for NaN after sequence slicing
        if torch.isnan(processed).any():
            print(f"WARNING: NaN detected after sequence slicing!")
            print(f"processed shape after slicing: {processed.shape}")
            print(f"NaN count: {torch.isnan(processed).sum().item()}")
        
        # Project to output: [B, pred_len, d_model] -> [B, pred_len, c_out]
        output = self.output_projection(processed)
        
        # Debug: Check for NaN after output projection
        if torch.isnan(output).any():
            print(f"WARNING: NaN detected after output projection!")
            print(f"output shape: {output.shape}")
            print(f"NaN count: {torch.isnan(output).sum().item()}")
        
        # De-normalize (match training model exactly)
        output = output * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        output = output + (means[:, [0], :].repeat(1, self.pred_len, 1))
        
        # Debug: Check for NaN after denormalization
        if torch.isnan(output).any():
            print(f"WARNING: NaN detected after denormalization!")
            print(f"output shape: {output.shape}")
            print(f"NaN count: {torch.isnan(output).sum().item()}")
            print(f"stdev[:, [0], :] shape: {stdev[:, [0], :].shape}")
            print(f"means[:, [0], :] shape: {means[:, [0], :].shape}")
        
        return output


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device, criterion, args) -> Tuple[float, dict, float, float]:
    """Evaluate model and return loss, metrics, inference time, and throughput"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_trues = []
    
    # Timing variables
    total_inference_time = 0.0
    total_samples = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            # Preprocess data (excluded from timing)
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            batch_x_mark = batch_x_mark.to(device).float()
            batch_y_mark = batch_y_mark.to(device).float()
            
            # Debug: Check for NaN values in input data
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                print(f"WARNING: NaN detected in input data!")
                print(f"batch_x NaN count: {torch.isnan(batch_x).sum().item()}")
                print(f"batch_y NaN count: {torch.isnan(batch_y).sum().item()}")
                print(f"batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
            
            # Create decoder input (excluded from timing)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
            
            # Synchronize GPU before timing (if using CUDA)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Time ONLY the model inference
            start_time = time.time()
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            # Debug: Check for NaN values in outputs
            if torch.isnan(outputs).any():
                print(f"WARNING: NaN detected in model outputs!")
                print(f"Output shape: {outputs.shape}")
                print(f"Output min/max: {outputs.min().item():.6f}/{outputs.max().item():.6f}")
                print(f"NaN count: {torch.isnan(outputs).sum().item()}")
            
            # Synchronize GPU after inference (if using CUDA)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            total_samples += batch_x.size(0)  # Count samples processed
            batch_count += 1
            
            # Apply feature dimension slicing (match training script exactly)
            f_dim = -1 if args.features == "MS" else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(device)
            
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            
            all_preds.append(outputs.detach().cpu().numpy())
            all_trues.append(batch_y.detach().cpu().numpy())
    
    # Compute metrics (similar to MambaTS)
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - trues) / (trues + 1e-8)))
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
    }
    
    # Calculate timing metrics
    avg_inference_time = total_inference_time / max(batch_count, 1)
    throughput_samples_per_sec = total_samples / max(total_inference_time, 1e-8)
    
    return total_loss / max(len(loader.dataset), 1), metrics, avg_inference_time, throughput_samples_per_sec


def run_single_experiment(args: argparse.Namespace, prune_ratio: float, run_id: int, global_run_counter: int) -> Dict[str, float]:
    """Run a single experiment with given pruning ratio"""
    print(f"  Run {run_id + 1}/10 - Pruning ratio: {prune_ratio:.1%}")
    
    # Set seed for reproducibility - always increment from base seed
    # This ensures each run gets a unique, deterministic seed
    current_seed = args.seed + global_run_counter
    set_seed(current_seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    # Load test data
    try:
        test_data, test_loader = data_provider(args, flag='test')
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None
    
    # Create model with specific pruning ratio
    model = TimeSeriesForecaster(
        enc_in=args.enc_in,
        c_out=args.c_out,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        prune_ratio=prune_ratio,
        use_solar=args.use_solar,
    ).to(device)
    
    
    # Load pre-trained model if available (use the same model for all pruning ratios)
    # The pre-trained model was likely trained without pruning (prune_ratio = 0.0)
    # We'll apply different pruning configurations during inference
    setting_name = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_pr{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        0.0,  # Use 0.0 since pre-trained models were likely trained without pruning
        args.des,
    )
    model_path = os.path.join(args.checkpoints, setting_name, "checkpoint.pth")
    
    if os.path.exists(model_path) and not args.no_pretrained:
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                    # Load the model state dict
                    state_dict = checkpoint['model_state_dict']
                    
                    # Load all parameters (pruning config is handled by model initialization)
                    # The pre-trained model was trained without pruning, but we'll apply pruning during inference
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded pre-trained model weights successfully")
                    print(f"Note: Model was trained without pruning, but applying {prune_ratio:.1%} pruning during inference")
                    
                    # Debug: Check for NaN in loaded weights
                    nan_count = 0
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            nan_count += torch.isnan(param).sum().item()
                            print(f"WARNING: NaN detected in parameter {name}")
                    if nan_count > 0:
                        print(f"Total NaN count in model parameters: {nan_count}")
                    else:
                        print("No NaN detected in model parameters")
            else:
                    # Direct state dict
                    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
                    print(f"Loaded pre-trained model weights successfully")
                    print(f"Note: Model was trained without pruning, but applying {prune_ratio:.1%} pruning during inference")
        except Exception as e:
            print(f"Warning: Could not load pre-trained model: {e}")
            print("Using randomly initialized model...")
    else:
        if args.no_pretrained:
            print("Pre-trained model loading disabled, using randomly initialized model...")
        else:
            print(f"No pre-trained model found, using randomly initialized model...")
            print(f"Expected path: {model_path}")
            print(f"This means we're testing pruning on a randomly initialized model (not ideal for comparison)")
    
    model.eval()
    
    
    # Evaluate model
    criterion = nn.MSELoss()
    
    
    loss, metrics, avg_inference_time, throughput = evaluate_model(model, test_loader, device, criterion, args)
    
    # Return results
    results = {
        'prune_ratio': prune_ratio,
        'run_id': run_id,
        'seed': current_seed,
        'loss': loss,
        'mae': metrics['mae'],
        'mse': metrics['mse'],
        'rmse': metrics['rmse'],
        'mape': metrics['mape'],
        'avg_inference_time': avg_inference_time,
        'throughput_samples_per_sec': throughput,
    }
    
    print(f"    Loss: {loss:.6f}, MAE: {metrics['mae']:.6f}, Time: {avg_inference_time:.4f}s, Throughput: {throughput:.2f} samples/s")
    
    
    return results


def grid_search(args: argparse.Namespace) -> None:
    """Run grid search over different pruning ratios"""
    print("=" * 60)
    print("GraphSSM Pruning Ratio Grid Search")
    print("=" * 60)
    
    # Define pruning ratios to test (as percentages)
    prune_ratios = [0.0, 0.15, 0.275, 0.4]  # 0%, 15%, 27.5%, 40%
    num_runs = 3
    
    print(f"Testing pruning ratios: {[f'{r:.1%}' for r in prune_ratios]}")
    print(f"Number of runs per ratio: {num_runs}")
    print(f"Total experiments: {len(prune_ratios) * num_runs}")
    print("=" * 60)
    
    # Store all results
    all_results = []
    global_run_counter = 0  # Track global run counter for seed generation
    
    # Run grid search
    for i, prune_ratio in enumerate(prune_ratios):
        print(f"\nPruning Ratio {i+1}/{len(prune_ratios)}: {prune_ratio:.1%}")
        print("-" * 40)
        
        ratio_results = []
        
        for run_id in range(num_runs):
            result = run_single_experiment(args, prune_ratio, run_id, global_run_counter)
            if result is not None:
                ratio_results.append(result)
                all_results.append(result)
            global_run_counter += 1  # Increment global counter for next run
        
        # Print summary for this ratio
        if ratio_results:
            avg_loss = np.mean([r['loss'] for r in ratio_results])
            avg_mae = np.mean([r['mae'] for r in ratio_results])
            avg_time = np.mean([r['avg_inference_time'] for r in ratio_results])
            avg_throughput = np.mean([r['throughput_samples_per_sec'] for r in ratio_results])
            std_loss = np.std([r['loss'] for r in ratio_results])
            std_mae = np.std([r['mae'] for r in ratio_results])
            std_time = np.std([r['avg_inference_time'] for r in ratio_results])
            std_throughput = np.std([r['throughput_samples_per_sec'] for r in ratio_results])
            
            print(f"\nSummary for {prune_ratio:.1%} pruning:")
            print(f"  Avg Loss: {avg_loss:.6f} ± {std_loss:.6f}")
            print(f"  Avg MAE:  {avg_mae:.6f} ± {std_mae:.6f}")
            print(f"  Avg Time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"  Avg Throughput: {avg_throughput:.2f} ± {std_throughput:.2f} samples/s")
    
    # Calculate overall statistics
    print("\n" + "=" * 60)
    print("OVERALL GRID SEARCH RESULTS")
    print("=" * 60)
    
    # Group results by pruning ratio
    results_by_ratio = {}
    for result in all_results:
        ratio = result['prune_ratio']
        if ratio not in results_by_ratio:
            results_by_ratio[ratio] = []
        results_by_ratio[ratio].append(result)
    
    # Calculate averages for each ratio
    summary_results = {}
    for ratio in prune_ratios:
        if ratio in results_by_ratio:
            ratio_results = results_by_ratio[ratio]
            
            summary_results[ratio] = {
                'avg_loss': np.mean([r['loss'] for r in ratio_results]),
                'std_loss': np.std([r['loss'] for r in ratio_results]),
                'avg_mae': np.mean([r['mae'] for r in ratio_results]),
                'std_mae': np.std([r['mae'] for r in ratio_results]),
                'avg_mse': np.mean([r['mse'] for r in ratio_results]),
                'std_mse': np.std([r['mse'] for r in ratio_results]),
                'avg_rmse': np.mean([r['rmse'] for r in ratio_results]),
                'std_rmse': np.std([r['rmse'] for r in ratio_results]),
                'avg_mape': np.mean([r['mape'] for r in ratio_results]),
                'std_mape': np.std([r['mape'] for r in ratio_results]),
                'avg_inference_time': np.mean([r['avg_inference_time'] for r in ratio_results]),
                'std_inference_time': np.std([r['avg_inference_time'] for r in ratio_results]),
                'avg_throughput': np.mean([r['throughput_samples_per_sec'] for r in ratio_results]),
                'std_throughput': np.std([r['throughput_samples_per_sec'] for r in ratio_results]),
                'num_runs': len(ratio_results)
            }
    
    # Print summary table
    print(f"{'Pruning Ratio':<15} {'Loss (MSE)':<20} {'MAE':<20} {'Time (s)':<15} {'Throughput (s/s)':<20} {'Runs':<8}")
    print("-" * 100)
    
    for ratio in prune_ratios:
        if ratio in summary_results:
            sr = summary_results[ratio]
            print(f"{ratio:<15.1%} "
                  f"{sr['avg_loss']:.6f}±{sr['std_loss']:.6f} "
                  f"{sr['avg_mae']:.6f}±{sr['std_mae']:.6f} "
                  f"{sr['avg_inference_time']:.4f}±{sr['std_inference_time']:.4f} "
                  f"{sr['avg_throughput']:.2f}±{sr['std_throughput']:.2f} "
                  f"{sr['num_runs']:<8}")
    
    # Save results to npz file
    save_results_to_npz(all_results, summary_results, args)
    
    print("\nGrid search completed!")
    print("=" * 60)


def save_results_to_npz(all_results: List[Dict], summary_results: Dict, args: argparse.Namespace) -> None:
    """Save all results to an npz file"""
    
    # Prepare data for saving
    prune_ratios = []
    run_ids = []
    seeds = []
    losses = []
    maes = []
    mses = []
    rmses = []
    mapes = []
    inference_times = []
    throughputs = []
    
    for result in all_results:
        prune_ratios.append(result['prune_ratio'])
        run_ids.append(result['run_id'])
        seeds.append(result['seed'])
        losses.append(result['loss'])
        maes.append(result['mae'])
        mses.append(result['mse'])
        rmses.append(result['rmse'])
        mapes.append(result['mape'])
        inference_times.append(result['avg_inference_time'])
        throughputs.append(result['throughput_samples_per_sec'])
    
    # Summary statistics
    summary_prune_ratios = []
    summary_avg_losses = []
    summary_std_losses = []
    summary_avg_maes = []
    summary_std_maes = []
    summary_avg_times = []
    summary_std_times = []
    summary_avg_throughputs = []
    summary_std_throughputs = []
    summary_num_runs = []
    
    for ratio, stats in summary_results.items():
        summary_prune_ratios.append(ratio)
        summary_avg_losses.append(stats['avg_loss'])
        summary_std_losses.append(stats['std_loss'])
        summary_avg_maes.append(stats['avg_mae'])
        summary_std_maes.append(stats['std_mae'])
        summary_avg_times.append(stats['avg_inference_time'])
        summary_std_times.append(stats['std_inference_time'])
        summary_avg_throughputs.append(stats['avg_throughput'])
        summary_std_throughputs.append(stats['std_throughput'])
        summary_num_runs.append(stats['num_runs'])
    
    # Create filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"grid_search_results_{args.data}_seq{args.seq_len}_pred{args.pred_len}_{timestamp}.npz"
    filepath = os.path.join(args.results, filename)
    
    # Ensure results directory exists
    os.makedirs(args.results, exist_ok=True)
    
    # Save to npz file
    np.savez(
        filepath,
        # Individual run results
        prune_ratios=np.array(prune_ratios),
        run_ids=np.array(run_ids),
        seeds=np.array(seeds),
        losses=np.array(losses),
        maes=np.array(maes),
        mses=np.array(mses),
        rmses=np.array(rmses),
        mapes=np.array(mapes),
        inference_times=np.array(inference_times),
        throughputs=np.array(throughputs),
        
        # Summary statistics
        summary_prune_ratios=np.array(summary_prune_ratios),
        summary_avg_losses=np.array(summary_avg_losses),
        summary_std_losses=np.array(summary_std_losses),
        summary_avg_maes=np.array(summary_avg_maes),
        summary_std_maes=np.array(summary_std_maes),
        summary_avg_times=np.array(summary_avg_times),
        summary_std_times=np.array(summary_std_times),
        summary_avg_throughputs=np.array(summary_avg_throughputs),
        summary_std_throughputs=np.array(summary_std_throughputs),
        summary_num_runs=np.array(summary_num_runs),
        
        # Metadata
        dataset=args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        d_model=args.d_model,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        base_seed=args.seed,
        timestamp=timestamp
    )
    
    print(f"\nResults saved to: {filepath}")
    print(f"File contains {len(all_results)} individual runs and {len(summary_results)} summary statistics")


def build_argparser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search evaluation of GraphSSM pruning ratios on ETT dataset")
    
    # Basic config (MambaTS style)
    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="GraphSSM", help="model id")
    parser.add_argument("--model", type=str, default="GraphSSM", help="model name")
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    
    # Data loader (MambaTS style)
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "ETT-small"), help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTm1.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--freq", type=str, default="t", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"), help="location of model checkpoints")
    parser.add_argument("--visualization", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_results"), help="location of model checkpoints")
    parser.add_argument("--results", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), help="location of model checkpoints")
    
    # Forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)
    
    # Model define
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--distil", action="store_false", help="whether to use distilling in encoder", default=True)
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
    
    # Optimization
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=1, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="GraphSSM_Exp", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="type1", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)
    parser.add_argument("--optimizer", type=str, default="adam", help="optimizer")
    parser.add_argument("--no_lradj", action="store_true")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--lradj_by_iter", action="store_true")
    parser.add_argument("--warmup_steps", default=0.1, type=float, help="warmup")
    parser.add_argument("--iters_per_epoch", default=None, type=str, help="warmup")
    
    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    
    # De-stationary projector params
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128], help="hidden layer dimensions of projector")
    parser.add_argument("--p_hidden_layers", type=int, default=2, help="number of hidden layers in projector")
    
    # PatchTST
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    
    # GraphSSM specific
    parser.add_argument("--d_state", type=int, default=16, help="GraphSSM state size")
    parser.add_argument("--d_conv", type=int, default=4, help="GraphSSM conv kernel size")
    parser.add_argument("--expand", type=int, default=2, help="Expansion ratio in GraphSSM")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    
    # Solar dataset specific
    parser.add_argument("--use_solar", action="store_true", help="Use Solar dataset instead of ETT", default=False)
    parser.add_argument("--solar_root_path", type=str, default="/data/eval_pipelines/datasets", help="Root path for Solar dataset")
    parser.add_argument("--solar_data_path", type=str, default="solar_AL.txt", help="Solar dataset file name")
    
    # Debug options
    parser.add_argument("--no_pretrained", action="store_true", help="Don't load pre-trained models (use random initialization)", default=False)
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = build_argparser()
    
    # Auto-configure data path based on dataset type
    if args.data == "ETTm1":
        args.data_path = "ETTm1.csv"
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.d_model = 512
        args.freq = "t"  # minute frequency
    elif args.data == "ETTh1":
        args.data_path = "ETTh1.csv"
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.d_model = 512
        args.freq = "h"  # hour frequency
    elif args.data == "ETTm2":
        args.data_path = "ETTm2.csv"
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.d_model = 512
        args.freq = "t"  # minute frequency
    elif args.data == "ETTh2":
        args.data_path = "ETTh2.csv"
        args.enc_in = 7
        args.dec_in = 7
        args.c_out = 7
        args.d_model = 512
        args.freq = "h"  # hour frequency
    
    # Configure Solar dataset if requested
    if args.use_solar:
        print("Using Solar dataset configuration...")
        args.data = "Solar"
        args.root_path = args.solar_root_path
        args.data_path = args.solar_data_path
        args.enc_in = 137
        args.dec_in = 137
        args.c_out = 137
        args.seq_len = 96
        args.label_len = 48
        # Note: pred_len is preserved from command line argument
        args.features = "M"
        args.d_model = 32
        print(f"Solar dataset: {args.root_path}/{args.data_path}")
        print(f"Features: {args.enc_in}, Seq len: {args.seq_len}, Pred len: {args.pred_len}")
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    print("=" * 60)
    print("GraphSSM Pruning Ratio Grid Search")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Data path: {os.path.join(args.root_path, args.data_path)}")
    print(f"Sequence length: {args.seq_len}, Prediction length: {args.pred_len}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    # Check if data file exists
    data_file_path = os.path.join(args.root_path, args.data_path)
    if not os.path.exists(data_file_path):
        print(f"ERROR: Data file not found at {data_file_path}")
        print("Please download the ETT dataset and place it in the correct location.")
        print("Download from: https://github.com/zhouhaoyi/ETDataset")
        print("Place the ETT-small folder in ~/data/datasets/ETDataset/")
        exit(1)
    
    # Run grid search
    print("\nStarting grid search...")
    grid_search(args)
    print("\nGrid search completed!")
