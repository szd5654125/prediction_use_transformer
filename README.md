BTC-Transformer · Minute‑Level Bitcoin Trend Forecast with Transformer
TL;DR: This repository offers a full end‑to‑end workflow — from raw Binance K‑line CSV → feature engineering → automatic labelling → Transformer training & hyper‑parameter tuning → inference & visualisation — with CPU/GPU acceleration at every stage.

✨ Highlights
Time2Vector temporal embedding — SineActivation first applies a linear map and then concatenates multi‑frequency sine components to expand the original features to hidden_dim, effectively injecting periodic patterns.

Pure encoder architecture — only nn.TransformerEncoder is used; no decoder is required for training or inference, which simplifies parallelism and deployment.

Optuna hyper‑parameter search — bitcoin_price_prediction_optuna.py explores a 20+ dimensional space with pruning, dynamically balancing CPU/GPU workloads.

Multi‑process feature engineering — add_finta_feature_parallel utilises 100+ processes to compute Finta technical indicators, fully saturating the CPU.

Self‑supervised trend labels — detect_trend(_optimized) generates the binary trend_returns label based on draw‑down thresholds, eliminating manual labelling.

📂 Repository Layout
├── bitcoin_price_prediction.py          # Baseline training script (single‑node)
├── bitcoin_price_prediction_optuna.py   # Hyper‑parameter search + retrain + visualisation
├── model.py                             # BTC_Transformer & Time2Vector
├── evaluation.py                        # Validation / test loops
├── data_process.py                      # Data preprocessing & batch sampling
├── set_target.py                        # Generate trend_returns label
├── reform.py                            # Notebook → .py converter
└── cuda.py                              # GPU environment check
⚙️ Environment
Component	Version
Python	≥ 3.10
PyTorch	≥ 2.2, CUDA 11.8
pandas / numpy / matplotlib / seaborn	latest
finta	latest
optuna	≥ 3.6
numba / scikit-learn	latest

Installation
conda create -n btc-transformer python=3.10 pytorch cudatoolkit=11.8 -c pytorch -c conda-forge
conda activate btc-transformer
pip install -r requirements.txt
Example requirements.txt:

pandas
numpy
matplotlib
seaborn
scikit-learn
finta
optuna
numba
torchsummary
tqdm
📑 Data Preparation
Download BTCUSDT-1m or BTCUSDT-3m K‑line CSV from Binance (official) or Kaggle.

Place files under input/btcusdt/ with names like BTCUSDT-1m-2024-12.csv.

Running any training script will automatically:

compute 25+ Finta indicators;

call detect_trend / detect_trend_optimized to create trend_returns;

split train / validation / test in an 8 : 1 : 1 ratio.

🚀 Quick Start
1. Baseline training
python bitcoin_price_prediction.py --epochs 50 --device cuda:0
The script prints training/validation loss curves and saves convergence plots.

2. Hyper‑parameter search
python bitcoin_price_prediction_optuna.py --trials 200 --n_jobs 32
18 dimensions are searched, including Transformer depth, hidden size, learning rate, etc.

CPU/GPU resources are scheduled automatically for maximum throughput.

The best model is stored as best_model_final.pt, with hyper‑parameters in best_params.json.

📏 Evaluation & Inference
from model import BTC_Transformer
from bitcoin_price_prediction_optuna import define_model
import torch, json

best_params = json.load(open("best_params.json"))
model, _ = define_model(best_params, torch.device("cuda:0"))
model.load_state_dict(torch.load("best_model_final.pt"))
model.eval()
See estimate_BTC for a sample prediction pipeline that returns de‑normalised ground‑truth and forecast series ready for back‑testing.

📊 Example Results (3‑Minute, 2022‑01 → 09)
Metric	Baseline	Optuna Best
Test Loss (CE)	0.51	0.34
↑ Directional Accuracy	61 %	69 %

Hardware: single RTX 4090; results for reference only.

🛠️ Customisation & Extensions
Switch to another coin — simply replace the CSV; indicators and labelling remain unchanged.

Regression task — disable the classifier, enable the generator, and swap the loss to nn.L1Loss().

Multi‑scale windows — sample bptt_src / bptt_tgt inside the Optuna objective for stronger generalisation.

Integrate WandB — a one‑liner to track experiments (see TODO).

📝 TODO
 Integrate Informer or PatchTST to reduce latency

 Use GPU‑Numba/CuPy to accelerate label generation

 WandB experiment tracking & feature‑importance analysis

 Lightweight back‑test engine for profit evaluation
