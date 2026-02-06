# [EACL SRW 2026] Token Pruning for Improving Graph-Generating State Space Model Performance

## Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- Conda package manager

### Installation

Create the environment and install dependencies:

```bash
conda create -y -n gg_ssms python=3.11
conda activate gg_ssms
conda install -y pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install -y nvidia::cuda-toolkit
```

Install the custom CUDA kernels:

```bash
cd core/convolutional_graph_ssm/third-party/TreeScan/
pip install -v -e .
cd $(git rev-parse --show-toplevel)

cd core/graph_ssm/third-party/TreeScanLan/
pip install -v -e .
cd $(git rev-parse --show-toplevel)
```

Install remaining Python dependencies:

```bash
pip install -r MambaTS/requirements.txt
```

## Training

To train a single model (e.g., ETTm1 with prediction length 96):

```bash
python train_graphssm.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id GraphSSM \
    --model GraphSSM \
    --data ETTm1 \
    --root_path ./eval_pipelines/datasets/ETT-small/ \
    --data_path ETTm1.csv \
    --seq_len 96 \
    --pred_len 96 \
    --d_model 512 \
    --prune_ratio 0.15 \
    --train_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --loss MSE \
    --lradj cosine \
    --use_amp \
    --use_gpu True
```

To train all datasets (ETTm1, ETTh1, Solar) across prediction lengths 96, 192, and 336:

```bash
bash train_all_datasets.sh
```

Checkpoints are saved to `./checkpoints/` and test results to `./test_results/`.

## Inference / Evaluation

To benchmark distance metrics on a trained model:

```bash
python ET_Eval2.py \
    --data ETTm1 \
    --seq_len 96 \
    --pred_len 96 \
    --metrics cosine euclidean gaussian manhattan norm2 \
    --output_dir ./benchmark_results
```

This evaluates each distance metric and produces comparison plots (PNG/PDF) and a results CSV in the specified output directory.

## Datasets

The following datasets are supported:

- **ETTm1** -- Electricity Transformer (minutely), 7 features
- **ETTh1** -- Electricity Transformer (hourly), 7 features
- **SolarAV** -- Solar power generation, 137 features

Dataset files are located under `eval_pipelines/datasets/`.

## Acknowledgments

The core GraphSSM architecture and training framework in this repository are sourced from the Graph-Generating State Space Models project by Nikola Zubic and Davide Scaramuzza at the University of Zurich. Their work was published at CVPR 2025.

- Paper: [GG-SSMs: Graph-Generating State Space Models](https://arxiv.org/abs/2412.12423)
- Original repository: [uzh-rpg/gg_ssms](https://github.com/uzh-rpg/gg_ssms)
