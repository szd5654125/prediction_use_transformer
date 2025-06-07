# packages import
# Numpy
import numpy as np

# Pandas
import pandas as pd 

# Pytorch
import torch
import torch.nn as nn

# Optuna
import optuna
import torch.optim as optim

# Others
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


num_gpus = torch.cuda.device_count()
num_cpus = max(multiprocessing.cpu_count(), 1)

# 限制 CPU 任务数量，防止 CPU 任务堆积
max_cpu_jobs = num_cpus / 3  # 限制 CPU 任务一半给回测
max_gpu_jobs = num_gpus * 10  # 每个 GPU 最多运行 10 个任务
total_jobs = max_cpu_jobs + max_gpu_jobs  # 总任务数

cpu_lock = threading.Lock()  # 线程锁，防止 CPU 任务计数竞争
cpu_active_tasks = 0  # 当前 CPU 任务计数

# font configuration
font = {'family': 'Arial', 'weight': 'normal', 'size': 16}

plt.rc('font', **font)

grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

data = pd.read_csv(os.path.join(grandparent_dir, "input", "btcusdt", "BTCUSDT-1m-2024-10.csv"))
# plot data processing statistics
plot_data_process = True
data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

# create data with all wanted features per minute
data_min = data.copy()

extra_features = ['SMA', 'TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI',
                  'ADX',
                  'STOCHRSI', 'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI',
                  'STC']
both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]
data_min = detect_trend_optimized(data_min)
# 使用finta添加特征
data_min = add_finta_feature_parallel(data_min, extra_features, both_columns_features)

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
# data_min.drop(['ignore'], inplace=True, axis=1)
in_features = data_min.shape[1]
print(f"data_min shape: {data_min.shape}")

# show info of data - now there are no Nans
if plot_data_process:
    data_min.info()

# split the data
val_percentage = 0.1
test_percentage = 0.1
train_df, val_df, test_df = train_validation_test_split(data_min, val_percentage, test_percentage)

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


# declaration of the define_model class for optuna
def define_model(trial, device):
    num_encoder_layers = trial.suggest_int("encoder_layers", 4, 8, step=4)
    num_decoder_layers = num_encoder_layers
    in_features = data_min.shape[1]
    # out_features = trial.suggest_int("out_features", 36, 64, step=4)
    # nhead = int(out_features / 4)
    hidden_dim = trial.suggest_int("hidden_dim", 48, 72, step=4)
    nhead = max(2, int(hidden_dim / 4)) # ✅ 确保 `nhead` 为偶数
    dim_feedforward = trial.suggest_int("dim_feedforward", 128, 512, step=128)
    dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
    activation = trial.suggest_categorical("activation", ["relu", "gelu"])
    # periodic_features = int((((out_features - in_features) // 10) * 4) + 2)
    periodic_features = int((((hidden_dim - in_features) // 10) * 4) + 2)

    return BTC_Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        in_features=in_features,
        periodic_features=periodic_features,
        # out_features=out_features,
        # 用于分类任务
        hidden_dim=hidden_dim,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        activation=activation,
        num_classes=2
    ).to(device), in_features


# declaration of the objective class for optuna
def objective(trial):
    """动态监测 CPU 负载，智能分配任务"""
    global cpu_active_tasks
    # 检查当前 CPU 活跃任务数
    with cpu_lock:
        if cpu_active_tasks < max_cpu_jobs:
            cpu_active_tasks += 1  # 增加 CPU 任务计数
            device = torch.device("cpu")
        elif num_gpus > 0:
            device = torch.device(f"cuda:{trial.number % num_gpus}")
        else:
            device = torch.device("cpu")  # 如果没有 GPU，所有任务都在 CPU

    print(f"Running trial {trial.number} on {device}")
    # split the data
    val_percentage = 0.1
    test_percentage = 0.1
    train_, val_, test_ = train_validation_test_split(data_min, val_percentage, test_percentage)
    # define the parameters
    overlap = 1
    criterion = nn.CrossEntropyLoss()  # nn.L1Loss(): 绝对值损失 nn.MSELoss():平方损失 nn.CrossEntropyLoss:交叉熵损失
    best_val_loss = float('inf')
    best_model = None
    in_features = data_min.shape[1]
    num_features = in_features
    step_size = 1
    epochs = 50
    train_batch_size = 32
    eval_batch_size = 32
    bptt_src = trial.suggest_int("bptt_src", 10, 60, step=10)
    bptt_tgt = trial.suggest_int("bptt_tgt", 2, 18, step=2)

    lr = trial.suggest_float("lr", low=1e-4, high=1e-1, log=True)

    optimizer_name = trial.suggest_categorical("optimizer_name", ["SGD", "Adam", "AdamW"])

    scaler_name = trial.suggest_categorical("scaler_name", ["standard", "minmax"])

    gamma = trial.suggest_float("gamma", 0.7, 0.99, step=0.05)

    clip_param = trial.suggest_float("clip_param", 0.25, 1, step=0.25)

    random_start_point = "True"  # trial.suggest_categorical("random_start_point", ["True", "False"])  取消固定起点减少时间

    # define the model
    model, in_features = define_model(trial, device)
    model.to(device)  # 让模型在正确的设备上运行

    # define the optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)

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

    train_data = betchify(train, train_batch_size, device).float()
    if val is not None:
        val_data = betchify(val, eval_batch_size, device).float()
    test_data = betchify(test, eval_batch_size, device).float()

    for epoch in range(1, epochs + 1):
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
        # masks for the model
        src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)  # zeros mask for the source (no mask)
        # look-ahead mask for the target
        # tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)
        tgt_mask = torch.triu(torch.ones((bptt_tgt, bptt_tgt), dtype=torch.bool), diagonal=1).to(device)
        # for batch, i in enumerate(range(start_point, train_data.size(0) - 1, bptt_src)):
        for batch, i in enumerate(range(start_point, train_data.size(1) - 1, bptt_src)):
            # forward
            source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, overlap)
            src_batch_size = source.size(0)
            tgt_batch_size = targets.size(0)
            if tgt_batch_size != bptt_tgt or src_batch_size != bptt_src:  # only on last batch
                src_mask = src_mask[:src_batch_size, :src_batch_size]
                tgt_mask = tgt_mask[:tgt_batch_size, :tgt_batch_size]
            output = model(source, targets, src_mask, tgt_mask)
            # loss = criterion(output[:-1,:,predicted_feature], targets[1:,:,predicted_feature])
            # 用于分类任务
            targets = (targets > 0.5).long()  # 先转换为0和1构成的数列
            '''targets = targets[-1, :, 0]
            output = output.view(-1, output.size(-1))'''
            targets = targets[:, -1, 0]

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
            val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

            # report results of optuna trial
            trial.report(val_loss, epoch)

            # cut trial if we get bad results
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # scheduler step
        scheduler.step()

    if val is None:
        best_model = copy.deepcopy(model)

    # 任务完成后减少 CPU 任务计数
    if device.type == "cpu":
        with cpu_lock:
            cpu_active_tasks -= 1  # 任务完成，减少 CPU 任务数

    return val_loss


def retrain_model(best_params, device="cuda:0", save_path="best_model_final.pt"):
    print(f"使用最佳参数重新训练模型，设备: {device}")

    # 构建模型
    trial = optuna.trial.FrozenTrial(
        number=0, state=optuna.trial.TrialState.COMPLETE, value=None,
        params=best_params, distributions=study.best_trial.distributions, user_attrs={}, system_attrs={},
        intermediate_values={}, datetime_start=None, datetime_complete=None
    )
    model, _ = define_model(trial, torch.device(device))
    model.to(device)

    # 设置训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, best_params["optimizer_name"])(model.parameters(), lr=best_params["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=best_params["gamma"])
    epochs = 50
    overlap = 1
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

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.
        start_point = np.random.randint(bptt_src)
        num_batches = (len(train_data) - start_point) // bptt_src

        src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
        # tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)
        tgt_mask = torch.triu(torch.ones((bptt_tgt, bptt_tgt), dtype=torch.bool), diagonal=1).to(device)

        # for batch, i in enumerate(range(start_point, train_data.size(0) - 1, bptt_src)):
        for batch, i in enumerate(range(start_point, train_data.size(1) - 1, bptt_src)):
            source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, overlap)
            output = model(source, targets, src_mask, tgt_mask)

            targets = (targets > 0.5).long()
            '''targets = targets[-1, :, 0]
            output = output.view(-1, output.size(-1))'''
            targets = targets[:, -1, 0]

            loss = criterion(output, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_param)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if val_data is not None:
            val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature,
                                torch.device(device))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)
            print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}")

    # 保存模型
    torch.save(best_model.state_dict(), save_path)
    print(f"✅ 最佳模型已保存至: {save_path}")

    return best_model


def visualize_test_predictions(model, test_df, scaler, predicted_feature, bptt_src, bptt_tgt, device="cuda:0"):
    model.eval()
    model.to(device)

    # 只保留输入特征
    test_data_raw = test_df.iloc[:, :in_features]
    true_labels_raw = test_df.iloc[:, predicted_feature]

    # 归一化
    test_scaled = scaler.transform(test_data_raw)
    test_scaled_tensor = torch.tensor(test_scaled, dtype=torch.float32).to(device)

    # 生成批次数据
    test_batches = betchify(pd.DataFrame(test_scaled, columns=test_data_raw.columns), batch_size=32, device=device).float()

    predictions = []
    targets = []

    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    # tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)
    tgt_mask = torch.triu(torch.ones((bptt_tgt, bptt_tgt), dtype=torch.bool), diagonal=1).to(device)

    for i in range(0, len(test_batches) - bptt_src - bptt_tgt, bptt_src):
        src, tgt = get_batch(test_batches, i, bptt_src, bptt_tgt, overlap=1)
        with torch.no_grad():
            output = model(src, tgt, src_mask, tgt_mask)
            # pred = output[-1, :, 1]  # 输出第二列，假设是1表示上涨的概率
            pred = output[:, -1, 1]
            predictions.append(pred.cpu().numpy())

            # 目标
            binary_target = (tgt[-1, :, 0] > 0.5).float()
            targets.append(binary_target.cpu().numpy())

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    # 可视化
    plt.figure(figsize=(15, 6))
    plt.plot(predictions, label="Predicted Prob (Up)", color='blue')
    plt.plot(targets, label="Actual Trend", color='red', alpha=0.6)
    plt.title("Test Set Predictions vs Ground Truth")
    plt.xlabel("Batch Index")
    plt.ylabel("Probability / Binary Label")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 设置要预测的列
predicted_feature = train_df.columns.get_loc('trend_returns')

sampler = optuna.samplers.TPESampler()
# 三块gpu最多运行40个任务，cpu最多128个，两个设备当前任务比值是1:2
study = optuna.create_study(study_name="BTC_Transformer", direction="minimize", sampler=sampler)
# study.optimize(objective, n_trials=200, n_jobs=total_jobs)  # 并行数乘二是因为一个gpu可以运行多个任务
study.optimize(objective, n_trials=200)  # 先尝试一个任务
best_params = study.best_trial.params
model, _ = define_model(study.best_trial, 'gpu')
best_model = retrain_model(best_params, device="cuda:0", save_path="best_model_final.pt")
if best_params["scaler_name"] == 'standard':
    scaler = pp.StandardScaler()
else:
    scaler = pp.MinMaxScaler()
visualize_test_predictions(model=best_model, test_df=test_df, scaler=scaler, predicted_feature=predicted_feature,
                           bptt_src=best_params["bptt_src"], bptt_tgt=best_params["bptt_tgt"], device="cuda:0")

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

# check which parameter is the most effective
optuna.visualization.plot_param_importances(study)
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["encoder_layers", "hidden_dim"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["out_features", "dim_feedforward"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["dim_feedforward", "dropout"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["dropout", "activation"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["activation", "bptt_src"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["bptt_src", "bptt_tgt"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["bptt_tgt", "clip_param"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["clip_param", "lr"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["lr", "optimizer_name"])
# Visualizing the Search Space
optuna.visualization.plot_contour(study, params=["optimizer_name", "gamma"])