"""
GraphSSM TimeSeriesForecaster Training Script

This script trains TimeSeriesForecaster models for ETT and Solar datasets,
following the MambaTS training pattern but using the GraphSSM architecture.
"""

import argparse
import os
import sys
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any
from tqdm import tqdm

# Add MambaTS to path
gg_ssms_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "workspace")
mamba_ts_path = os.path.join(gg_ssms_path, "MambaTS")

if not os.path.exists(mamba_ts_path):
    print(f"ERROR: MambaTS not found at {mamba_ts_path}")
    sys.exit(1)

sys.path.append(mamba_ts_path)

from data_provider.data_factory import data_provider
from utils.tools import set_seed, EarlyStopping, adjust_learning_rate
from utils.metrics import metric

# Import GraphSSM and TimeSeriesForecaster
sys.path.append(os.path.join(gg_ssms_path, "core", "graph_ssm"))
from main import GraphSSM


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
        prune_ratio: float = 0.15,
        use_solar: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model
        
        # Simple embedding layer to project input features to d_model
        self.input_embedding = nn.Linear(enc_in, d_model)
        
        # Core GraphSSM
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
            prune_ratio=prune_ratio,
            verbose=False
        )
        
        # Output projection to get predictions
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [B, seq_len, enc_in]
        b, seq_len, enc_in = x_enc.shape
        
        # Normalize (similar to MambaTS)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x_enc /= stdev
        
        # Embed input features: [B, seq_len, enc_in] -> [B, seq_len, d_model]
        embedded = self.input_embedding(x_enc)
        
        # Pass through GraphSSM
        context_len = min(seq_len, 4)
        processed = self.graph_ssm(embedded, context_len)
        
        # Take only the last pred_len timesteps for prediction
        if processed.shape[1] >= self.pred_len:
            processed = processed[:, -self.pred_len:, :]
        else:
            last_timestep = processed[:, -1:, :].repeat(1, self.pred_len, 1)
            processed = last_timestep
        
        # Project to output: [B, pred_len, d_model] -> [B, pred_len, c_out]
        output = self.output_projection(processed)
        
        # De-normalize
        output = output * (stdev[:, [0], :].repeat(1, self.pred_len, 1))
        output = output + (means[:, [0], :].repeat(1, self.pred_len, 1))
        
        return output


class GraphSSMTrainer:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device
    
    def _build_model(self):
        model = TimeSeriesForecaster(
            enc_in=self.args.enc_in,
            c_out=self.args.c_out,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            d_model=self.args.d_model,
            d_state=self.args.d_state,
            d_conv=self.args.d_conv,
            expand=self.args.expand,
            prune_ratio=self.args.prune_ratio,
            use_solar=self.args.use_solar,
        ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        if self.args.optimizer.lower() == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        if self.args.loss in ["MSE", "L2"]:
            criterion = nn.MSELoss(reduction="mean")
        elif self.args.loss == "L1":
            criterion = nn.L1Loss(reduction="mean")
        else:
            raise NotImplementedError
        return criterion
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                
                loss = criterion(pred, true)
                total_loss.append(loss)
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            # Create progress bar for training batches
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.train_epochs}", 
                            leave=False, disable=not self.args.verbose)
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_pbar):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
                
                # Update progress bar with loss information
                train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model
    
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        self.model.eval()
        
        preds = []
        trues = []
        
        print('model:{}'.format(setting))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
        
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)
        
        # result save
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return


def build_argparser():
    parser = argparse.ArgumentParser(description="GraphSSM TimeSeriesForecaster Training")
    
    # Basic config
    parser.add_argument("--task_name", type=str, default="long_term_forecast", help="task name")
    parser.add_argument("--is_training", type=int, default=1, help="status")
    parser.add_argument("--model_id", type=str, default="GraphSSM", help="model id")
    parser.add_argument("--model", type=str, default="GraphSSM", help="model name")
    parser.add_argument("--seed", type=int, default=3047, help="random seed")
    
    # Data loader
    parser.add_argument("--data", type=str, default="ETTm1", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./data/ETT/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target feature")
    parser.add_argument("--freq", type=str, default="t", help="freq for time features encoding")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")
    parser.add_argument("--visualization", type=str, default="./test_results", help="location of model checkpoints")
    
    # Forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="subset for M4")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)
    
    # Model define
    parser.add_argument("--enc_in", type=int, default=7, help="encoder input size")
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding")
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument("--output_attention", action="store_true", help="whether to output attention in ecoder")
    
    # Optimization
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=200, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="Exp", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")
    parser.add_argument("--lradj", type=str, default="cosine", help="adjust learning rate")
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision training", default=False)
    parser.add_argument("--optimizer", type=str, default="adamw", help="optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="weight decay for AdamW optimizer")
    
    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="device ids of multile gpus")
    
    # GraphSSM specific
    parser.add_argument("--d_state", type=int, default=16, help="GraphSSM state size")
    parser.add_argument("--d_conv", type=int, default=4, help="GraphSSM conv kernel size")
    parser.add_argument("--expand", type=int, default=2, help="Expansion ratio in GraphSSM")
    parser.add_argument("--prune_ratio", type=float, default=0.15, help="Pruning ratio for GraphSSM")
    
    # Solar dataset specific
    parser.add_argument("--use_solar", action="store_true", help="Use Solar dataset instead of ETT", default=False)
    parser.add_argument("--solar_root_path", type=str, default="/data/eval_pipelines/datasets", help="Root path for Solar dataset")
    parser.add_argument("--solar_data_path", type=str, default="solar_AL.txt", help="Solar dataset file name")
    
    # Progress tracking
    parser.add_argument("--verbose", action="store_true", help="Show detailed progress bars", default=True)
    
    return parser.parse_args()


def train_dataset(args, dataset_name, dataset_config):
    """Train model for a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Training GraphSSM for {dataset_name}")
    print(f"{'='*60}")
    
    # Store the original pred_len from command line
    original_pred_len = args.pred_len
    
    # Update args with dataset-specific configuration
    for key, value in dataset_config.items():
        setattr(args, key, value)
    
    # Restore the original pred_len from command line (override dataset config)
    args.pred_len = original_pred_len
    
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
        # args.pred_len is already set from command line - don't override it
        args.features = "M"
        args.d_model = 32
        print(f"Solar dataset: {args.root_path}/{args.data_path}")
        print(f"Features: {args.enc_in}, Seq len: {args.seq_len}, Pred len: {args.pred_len}")
    
    # Create experiment setting name
    setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_pr{}_{}".format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.prune_ratio,
        args.des,
    )
    
    print(f"Experiment setting: {setting}")
    
    # Create trainer and train
    trainer = GraphSSMTrainer(args)
    trainer.train(setting)
    
    # Test
    print(f"\nTesting {dataset_name}...")
    trainer.test(setting, test=1)
    
    print(f"Training completed for {dataset_name}")
    return setting


def main():
    args = build_argparser()
    
    # Set random seed
    set_seed(args.seed)
    
    # Dataset configurations
    datasets = {
        "ETTm1": {
            "data": "ETTm1",
            "root_path": "/data/eval_pipelines/datasets/ETT-small",
            "data_path": "ETTm1.csv",
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": args.pred_len,  # Use command line pred_len
            "d_model": 512,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "ETTh1": {
            "data": "ETTh1",
            "root_path": "/data/eval_pipelines/datasets/ETT-small",
            "data_path": "ETTh1.csv",
            "enc_in": 7,
            "dec_in": 7,
            "c_out": 7,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": args.pred_len,  # Use command line pred_len
            "d_model": 512,
            "batch_size": 32,
            "learning_rate": 0.001,
        },
        "Solar": {
            "use_solar": True,
            "solar_root_path": "/data/eval_pipelines/datasets",
            "solar_data_path": "solar_AL.txt",
            "enc_in": 137,
            "dec_in": 137,
            "c_out": 137,
            "seq_len": 96,
            "label_len": 48,
            "pred_len": args.pred_len,  # Use command line pred_len
            "d_model": 32,
            "batch_size": 16,
            "learning_rate": 0.001,
        }
    }
    
    # Train all datasets with progress tracking
    trained_models = {}
    dataset_names = list(datasets.keys())
    
    # Create overall progress bar for datasets
    dataset_pbar = tqdm(dataset_names, desc="Training Datasets", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for dataset_name in dataset_pbar:
        dataset_pbar.set_description(f"Training {dataset_name}")
        try:
            setting = train_dataset(args, dataset_name, datasets[dataset_name])
            trained_models[dataset_name] = setting
            dataset_pbar.set_postfix({'status': '✓ Completed'})
        except Exception as e:
            print(f"Error training {dataset_name}: {e}")
            dataset_pbar.set_postfix({'status': '✗ Failed'})
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for dataset_name, setting in trained_models.items():
        print(f"{dataset_name}: {setting}")
    
    print(f"\nAll models saved in: ./checkpoints/")
    print(f"Test results saved in: ./test_results/")


if __name__ == "__main__":
    main()
