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
import plotly

# Others
import matplotlib.pyplot as plt
import copy
import sklearn.preprocessing as pp
import multiprocessing
import datetime
import difflib

from data_process import train_validation_test_split, normalize_data, betchify, get_batch, add_finta_feature
from model import BTC_Transformer
from evaluation import evaluate
from set_target import detect_trend


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_gpus = torch.cuda.device_count()
num_cpus = max(multiprocessing.cpu_count() - 1, 1)
# 计算 Optuna 的最大并行任务数
n_parallel_trials = num_gpus + num_cpus
n_parallel_trials = 1
# font configuration
font = {'family': 'Arial', 'weight': 'normal', 'size': 16}

plt.rc('font', **font)


# load the data
data = pd.read_csv("../input/btcusdt/BTCUSDT-1m-2024-12.csv")
# plot data processing statistics
plot_data_process = True

# create data with all wanted features per minute
data_min = data.copy()

extra_features = ['SMA','TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX',
                  'STOCHRSI', 'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI',
                  'STC']
both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]

data_min = detect_trend(data_min)
# 使用finta添加特征
data_min = add_finta_feature(data_min, extra_features, both_columns_features)
# 获取当前时间
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
# 生成文件名并保存 CSV
file_name = f"trend_data_{current_time}.csv"
data_min.to_csv(file_name, index=False, encoding='utf-8')
print(f"文件已保存为: {file_name}")

if plot_data_process:
    print("原始数据行数：", data_min.shape[0])
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
data_min.drop(['ignore'], inplace=True, axis=1)
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
    if val_df is not None:
        val_time = np.arange(np.size(train_df, 0), np.size(train_df, 0) + np.size(val_df, 0))
        test_time = np.arange(np.size(train_df, 0) + np.size(val_df, 0), np.size(train_df, 0) + np.size(val_df, 0)
                              + np.size(test_df, 0))
    else:
        test_time = np.arange(np.size(train_df, 0), np.size(train_df, 0) + np.size(test_df, 0))
    fig = plt.figure(figsize=(16, 8))
    st = fig.suptitle("Data Separation", fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time - Minutes From (UTC+8): 2021-01-01 {:02d}:{:02d}:00".format(start_hour, start_minute))     
    ax.set_ylabel("Closing Price [USD]")            
    ax.set_title("Closing Price Through Time")
    ax.plot(train_time, train_df['close'], label='Training data')
    if val_df is not None:
        ax.plot(val_time, val_df['close'], label='Validation data')
    ax.plot(test_time, test_df['close'], label='Test data')
    ax.grid()
    ax.legend(loc="best", fontsize=12)
    plt.show()


# declaration of the define_model class for optuna
def define_model(trial):
    num_encoder_layers = trial.suggest_int("encoder_layers", 4, 8, step=4)
    num_decoder_layers = num_encoder_layers
    in_features = 40
    # out_features = trial.suggest_int("out_features", 36, 64, step=4)
    # nhead = int(out_features / 4)
    hidden_dim = trial.suggest_int("hidden_dim", 44, 72, step=4)
    nhead = int(hidden_dim / 4)
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
                            num_classes = 2
                            ).to(device), in_features

# %%
# declaration of the objective class for optuna
def objective(trial):
    # 分配计算设备
    if num_gpus > 0:
        gpu_id = trial.number % num_gpus  # 轮询方式分配 GPU
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    print(f"Trial {trial.number} running on {device}")

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
    model, in_features = define_model(trial)
    model.to(device)  # 让模型在正确的设备上运行
    
    # define the optimizer
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # define the scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=gamma)
    
    # define the scaler
    if scaler_name == 'standard':
        scaler = pp.StandardScaler()
    if scaler_name == 'minmax':
        scaler = pp.MinMaxScaler()
        
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
        log_interval =  max(1, round(num_batches // 3 / 10) * 10)
        # masks for the model 
        src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device) # zeros mask for the source (no mask)
        tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device) # look-ahead mask for the target
        # print(train_data[:10, :, 9])
        for batch, i in enumerate(range(start_point, train_data.size(0) - 1, bptt_src)):
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
            targets = targets[-1, :, 0]
            output = output.view(-1, output.size(-1))
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

    return val_loss

# 设置要预测的列
predicted_feature = train_df.columns.get_loc('trend_returns')

sampler = optuna.samplers.TPESampler()

study = optuna.create_study(study_name="BTC_Transformer", direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=50, n_jobs=n_parallel_trials)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("Number of finished trials: ",len(study.trials))
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
optuna.visualization.plot_contour(study, params=["encoder_layers", "out_features"])

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

