# packages import
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import optuna
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import copy
import sklearn.preprocessing as pp
import multiprocessing
import threading
import os

from data_process import train_validation_test_split, normalize_data, betchify, get_batch, add_finta_feature_parallel
from model import BTC_Transformer
from evaluation import evaluate
from set_target import detect_trend_optimized


CONFIG = {
    # 数据路径与特征
    "data_file": "input/btcusdt/BTCUSDT-3m-2022-01_to_03.csv",
    "extra_features": [
        'SMA', 'TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI',
        'ADX', 'STOCHRSI', 'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI', 'STC'
    ],
    "both_columns_features": ["DMI", "EBBP", "BASP", "BASPN"],

    # 数据划分
    "val_percentage": 0.1,
    "test_percentage": 0.1,

    # 模型训练参数
    "epochs": 50,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "step_size": 1,
    "overlap": 1,
    "random_start": True,

    # 超参数搜索空间
    "lr_range": (1e-4, 1e-1),
    "gamma_range": (0.7, 0.97, 0.05),
    "clip_range": (0.25, 1.0, 0.25),
    "bptt_src_range": (10, 100, 10),
    "bptt_tgt_range": (6, 28, 2),
    "optimizers": ["SGD", "Adam", "AdamW"],
    "scalers": ["standard", "minmax"],
    "activations": ["relu", "gelu"],
    "nhead_candidates": [2, 4, 8, 12, 16],
    "encoder_layer_range": (2, 8, 2),
    "hidden_dim_range": (48, 208, 16),
    "feedforward_dim_range": (128, 512, 128),
    "dropout_range": (0.0, 0.5, 0.1),

    # 显卡/CPU 并发控制
    "num_gpus": torch.cuda.device_count(),
    "num_cpus": max(multiprocessing.cpu_count(), 1),

    # 其他
    "default_device": "cuda:0",
    "model_save_path": "best_model_final.pt",
    "predicted_column": "trend_returns",
    "plot_data_process": True,
    "font": {'family': 'Arial', 'weight': 'normal', 'size': 16},
}
CONFIG["max_cpu_jobs"] = CONFIG["num_cpus"] / 3  # 另外 2/3暂时给回测任务
CONFIG["max_gpu_jobs"] = CONFIG["num_gpus"] * 5  # 每个 GPU 最多运行 10 个任务
CONFIG["total_jobs"] = CONFIG["max_cpu_jobs"] + CONFIG["max_gpu_jobs"]

cpu_lock = threading.Lock()  # 线程锁，防止 CPU 任务计数竞争
cpu_active_tasks = 0  # 当前 CPU 任务计数

# font configuration
font = {'family': 'Arial', 'weight': 'normal', 'size': 16}

plt.rc('font', **font)

grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

data = pd.read_csv(os.path.join(grandparent_dir, CONFIG["data_file"]))
# plot data processing statistics
plot_data_process = True
data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

# create data with all wanted features per minute
data_min = data.copy()

data_min = detect_trend_optimized(data_min)
# 使用finta添加特征
data_min = add_finta_feature_parallel(data_min, CONFIG["extra_features"], CONFIG["both_columns_features"])

# 删除所有包含NA的行
data_min = data_min.dropna().reset_index(drop=True)
if plot_data_process:
    print("删除NA后数据行数：", data_min.shape[0])
# 由于已经删除了所有NA行，所以第一行一定是完整的
first_complete_index = 0
start_hour = first_complete_index // 60
start_minute = first_complete_index % 60

# 去掉时间，不再需要的列以及相似的列
data_min.drop(['open_time'], inplace=True, axis=1)
data_min.drop(['close_time'], inplace=True, axis=1)
data_min.drop(['quote_volume'], inplace=True, axis=1)
data_min.drop(['taker_buy_quote_volume'], inplace=True, axis=1)
data_min.drop(['ignore'], inplace=True, axis=1)
in_features = data_min.shape[1]
print(f"data_min shape: {data_min.shape}")

# show info of data - now there are no Nans
if plot_data_process:
    data_min.info()

# split the data
train_df, val_df, test_df = train_validation_test_split(data_min, CONFIG['val_percentage'], CONFIG['test_percentage'])

print(np.shape(train_df))
if val_df is not None:
    print(np.shape(val_df))
print(np.shape(test_df))

# plot train, validation and test separation
if plot_data_process:
    train_time = np.arange(np.size(train_df, 0))
    fig = plt.figure(figsize=(16, 8))
    st = fig.suptitle("Data Separation", fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time - Minutes From (UTC+8): 2021-01-01 {:02d}:{:02d}:00".format(start_hour, start_minute))
    ax.set_ylabel("Closing Price [USD]")
    ax.set_title("Closing Price Through Time")
    ax.plot(train_time, train_df['close'], label='Training data')
    if val_df is not None:
        val_time = np.arange(np.size(train_df, 0), np.size(train_df, 0) + np.size(val_df, 0))
        ax.plot(val_time, val_df['close'], label='Validation data')
        test_time = np.arange(np.size(train_df, 0) + np.size(val_df, 0),
                              np.size(train_df, 0) + np.size(val_df, 0) + np.size(test_df, 0))
    else:
        test_time = np.arange(np.size(train_df, 0), np.size(train_df, 0) + np.size(test_df, 0))
    ax.plot(test_time, test_df['close'], label='Test data')
    ax.grid()
    ax.legend(loc="best", fontsize=12)
    plt.show()


def compute_min_hidden_dim(in_features: int, min_linear_features: int = 1, nhead_candidates=(2, 4, 8, 12, 16)):
    base = min(nhead_candidates) * 2
    H = ((in_features + base - 1) // base) * base
    while True:
        periodic = ((H - in_features) // 10) * 4 + 2
        linear = H - in_features - periodic
        if linear >= min_linear_features:
            return H
        H += base


# declaration of the define_model class for optuna
def define_model(params_or_trial, device):
    def get_param(key, default=None, suggest_fn=None):
        if isinstance(params_or_trial, dict):
            return params_or_trial.get(key, default)
        else:
            return suggest_fn(key)

    enc_start, enc_end, enc_step = CONFIG["encoder_layer_range"]
    num_encoder_layers = get_param("encoder_layers", suggest_fn=lambda k: params_or_trial.suggest_int(k, enc_start, enc_end, step=enc_step))
    in_features = data_min.shape[1]
    hid_start_raw, hid_end, hid_step = CONFIG["hidden_dim_range"]
    min_hidden_dim = compute_min_hidden_dim(in_features=data_min.shape[1], min_linear_features=1,
                                            nhead_candidates=CONFIG["nhead_candidates"])
    hid_start = max(hid_start_raw, min_hidden_dim)
    hidden_dim = get_param("hidden_dim", suggest_fn=lambda k: params_or_trial.suggest_int(k, hid_start, hid_end,
                                                                                          step=hid_step))
    nhead = get_param("nhead", suggest_fn=lambda k: params_or_trial.suggest_categorical(k, CONFIG["nhead_candidates"]))
    if hidden_dim % nhead != 0 or (hidden_dim // nhead) % 2 != 0:
        raise optuna.TrialPruned()
    ff_start, ff_end, ff_step = CONFIG["feedforward_dim_range"]
    dim_feedforward = get_param("dim_feedforward", suggest_fn=lambda k: params_or_trial.suggest_int(k, ff_start, ff_end,
                                                                                                    step=ff_step))
    drop_start, drop_end, drop_step = CONFIG["dropout_range"]
    dropout = get_param("dropout", suggest_fn=lambda k: params_or_trial.suggest_float(k, drop_start, drop_end,
                                                                                      step=drop_step))
    activations = CONFIG["activations"]
    activation = get_param("activation", suggest_fn=lambda k: params_or_trial.suggest_categorical(k, activations))
    periodic_features = int((((hidden_dim - in_features) // 10) * 4) + 2)
    return BTC_Transformer(
        num_encoder_layers=num_encoder_layers, in_features=in_features,
        periodic_features=periodic_features, hidden_dim=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward,
        dropout=dropout, activation=activation, num_classes=2).to(device), in_features


# declaration of the objective class for optuna
def objective(trial):
    """动态监测 CPU 负载，智能分配任务"""
    global cpu_active_tasks
    # 检查当前 CPU 活跃任务数
    with cpu_lock:
        if cpu_active_tasks < CONFIG["max_cpu_jobs"]:
            cpu_active_tasks += 1  # 增加 CPU 任务计数
            device = torch.device("cpu")
        elif CONFIG["num_gpus"] > 0:
            device = torch.device(f"cuda:{trial.number % CONFIG['num_gpus']}")
        else:
            device = torch.device("cpu")  # 如果没有 GPU，所有任务都在 CPU

    print(f"Running trial {trial.number} on {device}")
    # define the parameters
    criterion = nn.CrossEntropyLoss()  # nn.L1Loss(): 绝对值损失 nn.MSELoss():平方损失 nn.CrossEntropyLoss:交叉熵损失
    best_val_loss = float('inf')
    in_features = data_min.shape[1]
    num_features = in_features
    bptt_src_low, bptt_src_high, bptt_src_step = CONFIG["bptt_src_range"]
    bptt_src = trial.suggest_int("bptt_src", bptt_src_low, bptt_src_high, step=bptt_src_step)
    bptt_tgt_low, bptt_tgt_high, bptt_tgt_step = CONFIG["bptt_tgt_range"]
    bptt_tgt = trial.suggest_int("bptt_tgt", bptt_tgt_low, bptt_tgt_high, step=bptt_tgt_step)
    lr_low, lr_high = CONFIG["lr_range"]
    lr = trial.suggest_float("lr", low=lr_low, high=lr_high, log=True)
    optimizer_name = trial.suggest_categorical("optimizer_name", CONFIG["optimizers"])
    scaler_name = trial.suggest_categorical("scaler_name", CONFIG["scalers"])
    gamma_low, gamma_high, gamma_step = CONFIG["gamma_range"]
    gamma = trial.suggest_float("gamma", gamma_low, gamma_high, step=gamma_step)
    clip_low, clip_high, clip_step = CONFIG["clip_range"]
    clip_param = trial.suggest_float("clip_param", clip_low, clip_high, step=clip_step)
    random_start_point = "True"  # trial.suggest_categorical("random_start_point", ["True", "False"])  取消固定起点减少时间
    # define the model
    model, in_features = define_model(trial, device)
    model.to(device)  # 让模型在正确的设备上运行
    # define the optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    # define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG['step_size'], gamma=gamma)
    # define the scaler
    if scaler_name == 'standard':
        scaler = pp.StandardScaler()
    elif scaler_name == 'minmax':
        scaler = pp.MinMaxScaler()
    else:
        raise ValueError(f'invalid scaler_name as {scaler_name}')
    # create the relevant data
    train = train_df.iloc[:, :num_features]
    if val_df is not None:
        val = val_df.iloc[:, :num_features]
    else:
        val = val_df
    test = test_df.iloc[:, :num_features]
    train, val, test, scaler = normalize_data(train, val, test, scaler)
    train_data = betchify(train, CONFIG['train_batch_size'], device).float()
    if val is not None:
        val_data = betchify(val, CONFIG['eval_batch_size'], device).float()
    for epoch in range(1, CONFIG['epochs'] + 1):
        # epoch initialization
        model.train()
        total_loss = 0.
        epoch_loss = 0.
        # start point of the data
        if random_start_point:
            start_point = np.random.randint(bptt_src)
        else:
            start_point = 0
        num_batches = (len(train_data) - start_point) // bptt_src
        log_interval = max(1, round(num_batches // 3 / 10) * 10)
        # look-ahead mask for the target
        for batch, i in enumerate(range(start_point, train_data.size(1) - 1, bptt_src)):
            # forward
            source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, CONFIG['overlap'])
            src_len = source.size(1)  # 时间维
            tgt_len = targets.size(1)
            src_mask = torch.zeros(src_len, src_len, dtype=torch.bool, device=device)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), 1)
            # output = model(source, targets, src_mask, tgt_mask)
            output = model(source, src_mask)
            # 用于分类任务
            target_index = train_df.columns.get_loc(CONFIG["predicted_column"])
            targets = (targets > 0.5).long()  # 先转换为0和1构成的数列
            targets = targets[:, -1, target_index]
            loss = criterion(output, targets)
            # backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param)
            # step
            optimizer.step()
            # record bacth statistics
            total_loss += loss.item()
            epoch_loss += len(source) * loss.item()
            # print statistics every log_interval
            if (batch % log_interval == 0) and batch > 0:
                lr = scheduler.get_last_lr()[0]
                cur_loss = total_loss / log_interval
                total_loss = 0
        # evaluate on validation and save best model
        if val is not None:
            val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, CONFIG['overlap'], criterion,
                                CONFIG["predicted_column"], device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            # report results of optuna trial
            trial.report(val_loss, epoch)
            # cut trial if we get bad results
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        # scheduler step
        scheduler.step()

    # 任务完成后减少 CPU 任务计数
    if device.type == "cpu":
        with cpu_lock:
            cpu_active_tasks -= 1  # 任务完成，减少 CPU 任务数
    return val_loss


def retrain_model(best_params, device="cuda:0", save_path="best_model_final.pt"):
    print(f"使用最佳参数重新训练模型，设备: {device}")
    # 构建模型
    model, _ = define_model(best_params, torch.device(device))
    model.to(device)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params["optimizer_name"])(model.parameters(), lr=best_params["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, CONFIG['step_size'], gamma=best_params["gamma"])
    bptt_src = best_params["bptt_src"]
    bptt_tgt = best_params["bptt_tgt"]
    clip_param = best_params["clip_param"]
    # 归一化数据
    if best_params["scaler_name"] == 'standard':
        scaler = pp.StandardScaler()
    else:
        scaler = pp.MinMaxScaler()

    train = train_df.iloc[:, :in_features]
    val = val_df.iloc[:, :in_features] if val_df is not None else None
    test = test_df.iloc[:, :in_features]

    train, val, test, scaler = normalize_data(train, val, test, scaler)

    train_data = betchify(train, 32, torch.device(device)).float()
    val_data = betchify(val, 32, torch.device(device)).float() if val is not None else None

    best_val_loss = float('inf')
    best_model = None

    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0.
        start_point = np.random.randint(bptt_src)
        for batch, i in enumerate(range(start_point, train_data.size(1) - 1, bptt_src)):
            source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, CONFIG['overlap'])
            src_len = source.size(1)  # 时间维
            tgt_len = targets.size(1)
            src_mask = torch.zeros(src_len, src_len, dtype=torch.bool, device=device)
            tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=device), 1)
            # output = model(source, targets, src_mask, tgt_mask)
            output = model(source, src_mask)
            target_index = train_df.columns.get_loc(CONFIG["predicted_column"])
            targets = (targets > 0.5).long()
            targets = targets[:, -1, target_index]
            loss = criterion(output, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param)
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()
        if val_data is not None:
            val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, CONFIG['overlap'], criterion,
                                CONFIG["predicted_column"], torch.device(device))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")
    # 保存模型
    save_path = "best_model_final.pt"
    torch.save(best_model.state_dict(), save_path)
    print(f"✅ 最佳模型已保存至: {save_path}")
    return best_model, scaler


def visualize_test_predictions(model, test_df, scaler, bptt_src, bptt_tgt, device="cuda:0"):
    model.eval()
    model.to(device)
    # 只保留输入特征
    train = train_df.iloc[:, :in_features]
    val = val_df.iloc[:, :in_features] if val_df is not None else None
    test = test_df.iloc[:, :in_features]
    train, val, test, scaler = normalize_data(train, val, test, scaler)
    test_batches = betchify(test, batch_size=32, device=torch.device(device)).float()
    predictions = []
    targets = []
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    tgt_mask = torch.triu(torch.ones((bptt_tgt, bptt_tgt), dtype=torch.bool), diagonal=1).to(device)
    sequence_length = test_batches.shape[1]
    for batch_idx in range(test_batches.shape[0]):
        for i in range(0, sequence_length - bptt_src - bptt_tgt, bptt_src):
            src, tgt = get_batch(test_batches, i, bptt_src, bptt_tgt, CONFIG['overlap'])
            # [time, feature] -> 加 batch 维度，变成 [1, time, feature]
            src = src.to(device)
            tgt = tgt.to(device)

            with torch.no_grad():
                src_len = src.size(1)
                src_mask = torch.zeros(src_len, src_len, dtype=torch.bool, device=device)
                output = model(src, src_mask=src_mask)
                pred = output[:, -1]  # shape: [1]
                predictions.append(pred.cpu().numpy())
                binary_target = (tgt[:, -1, 0] > 0.5).float()
                targets.append(binary_target.cpu().numpy())
    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)
    # 可视化
    plt.figure(figsize=(15, 6))
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    plt.plot(predictions, label="Predicted Prob (Up)", color='blue')
    plt.plot(targets, label="Actual Trend", color='red', alpha=0.6)
    plt.title("Test Set Predictions vs Ground Truth")
    plt.xlabel("Batch Index")
    plt.ylabel("Probability / Binary Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_predictions.png")
    # plt.show()  # 非交互式无法展示图片


# 设置要预测的列
sampler = optuna.samplers.TPESampler()
# 三块gpu最多运行40个任务，cpu最多128个，两个设备当前任务比值是1:2
study = optuna.create_study(study_name="BTC_Transformer", direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=100, n_jobs=CONFIG["total_jobs"])  # 并行数乘二是因为一个gpu可以运行多个任务
# study.optimize(objective, n_trials=200)  # 先尝试一个任务
# 打印获得的最佳参数结果
pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("Number of finished trials: ", len(study.trials))
print("Number of pruned trials: ", len(pruned_trials))
print("Number of complete trials: ", len(complete_trials))
print("Best trial: ")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
best_params = study.best_trial.params
with open("best_params.json", "w") as f:
    json.dump(best_params, f)
best_model_from_retrain, scaler = retrain_model(best_params, device="cuda:0", save_path="best_model_final.pt")

with open("best_params.json", "r") as f:
    best_params = json.load(f)
best_model, _ = define_model(best_params, device="cuda:0")
best_model.load_state_dict(torch.load("best_model_final.pt"))
best_model.to("cuda:0")
if best_params["scaler_name"] == 'standard':
    scaler = pp.StandardScaler()
else:
    scaler = pp.MinMaxScaler()
train = train_df.iloc[:, :in_features]
scaler = scaler.fit(train)

visualize_test_predictions(model=best_model, test_df=test_df, scaler=scaler, bptt_src=best_params["bptt_src"],
                           bptt_tgt=best_params["bptt_tgt"], device="cuda:0")
