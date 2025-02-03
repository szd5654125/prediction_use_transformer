# packages import
# Numpy
import numpy as np

# Pandas
import pandas as pd

# Pytorch
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

# Seaborn
import seaborn as sns

# Finta
from finta import TA

# Others
import matplotlib.pyplot as plt
import time
import math
import copy
import sklearn.preprocessing as pp

from data_process import train_validation_test_split, normalize_data, betchify, get_batch, add_finta_feature
from model import BTC_Transformer
from evaluation import evaluate

# %%
# plot data processing statistics
plot_data_process = True

# choose the Hyper-Parameters
# data Hyper-Parameters
num_features = 34  # maximum 34
scaler_name = 'minmax'  # choose from ['standard','minmax']
train_batch_size = 32
eval_batch_size = 32
epochs = 50
bptt_src = 10
bptt_tgt = 2
overlap = 1

# model Hyper-Parameters
num_encoder_layers = 4
num_decoder_layers = 4
periodic_features = 10
out_features = 60  # must be greater or equal to (num_features + periodic_features)
nhead = 15  # must divide out_features
dim_feedforward = 384
dropout = 0.0
activation = 'gelu'  # choose from ['relu','gelu']

# training Hyper-Parameters
random_start_point = False
clip_param = 0.75
lr = 0.5
gamma = 0.95
step_size = 1.0

# %%
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# %%
# font configuration
font = {'family': 'Arial', 'weight': 'normal', 'size': 14}

plt.rc('font', **font)

# %%
# load the data
data = pd.read_csv("../input/btcusdt/okex_btcusdt_kline_1m.csv")
columns_dict = {'t': 'Unix_timestamp',
                'o': 'Opening_price',
                'h': 'Highest_price',
                'l': 'Lowest_price',
                'c': 'Closing_price',
                'v': 'Volume_of_transactions'
                }
data = data.rename(columns=columns_dict)


# show info of data
if plot_data_process:
    data.info()
if plot_data_process:
    data.head()
# print last 5 lines of data
if plot_data_process:
    data.tail()


# sort datapoints by timestamp
data = data.sort_values('t', ignore_index=True)


# converts format from unix to UTC+8
data['t'] = pd.to_datetime(data['t'], unit='ms') + pd.Timedelta('08:00:00')
data = data.rename(columns={'t': 'Timestamp'})


def plot_correlation(data, title, figsize=(10, 5)):
    """Plots correlations between all columns in data

    Args:
        data: DataFrame, holds the data
        title: string, desired title of the plot
        figsize: tuple, width and height of the figure in inches
    """
    f, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data.corr() * 100, annot=True, linewidths=5, fmt='.2f', ax=ax)
    plt.title(title)
    plt.xticks(rotation=60)
    plt.show()


# %%
def plot_scaled_features(data, features_to_plot, scaler=None, xlabel="Time"):
    """Plots desired features in data after scaling

    Args:
        data: DataFrame, holds the data
        xlabel: string, desired x label of the plot
        scaler: function, scaling function
        features_to_plot: list of strings, columns names of data to plot
    """


    for feature in features_to_plot:
        if feature not in data_min.columns:
            print(f"Feature '{feature}' is missing in data_min.")
        elif data_min[feature].isnull().all():
            print(f"Feature '{feature}' has no valid data (all NaN).")



    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Features' Values [a.u]")
    ax.set_title("Features' Values Through Time")
    ax.grid()
    for column in data:
        if column not in features_to_plot:
            continue
        feature = data[column].values
        if scaler is not None:
            feature_normalized = scaler.fit_transform(feature.reshape(-1, 1))
        else:
            feature_normalized = feature
        ax.plot(feature_normalized, label=column)
    ax.legend()
    plt.show()


# %%
# check correlation between features in minutes form
if plot_data_process:
    plot_correlation(data, title="Correlations Between Features in Minutes")


# Finta needs specific columns' names to work
data_finta = pd.DataFrame()
data_finta['open'] = data['o']
data_finta['high'] = data['h']
data_finta['low'] = data['l']
data_finta['close'] = data['c']
data_finta['volume'] = data['v']

# create data with all wanted features per minute
data_min = data.copy()
extra_features = ['TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI', 'ADX',
                  'STOCHRSI', 'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP', 'BASP', 'BASPN', 'WTO', 'SQZMI', 'VFI',
                  'STC']
both_columns_features = ["DMI", "EBBP", "BASP", "BASPN"]
add_finta_feature(data_min, data_finta, extra_features, both_columns_features)


# check correlation between all features including new ones in minutes form
if plot_data_process:
    plot_correlation(data_min,
                     title="Correlations Between Extended Features in Minutes",
                     figsize=(46, 23))


# show info of data
if plot_data_process:
    data_min.info()


# find the maximum index containing NaN
last_nan_indices = data_min.apply(lambda col: col.last_valid_index() if col.isna().any() else -1)
if plot_data_process:
    print("Last index containing NaN in each feature:")
    print(last_nan_indices)
max_index = last_nan_indices.max()
if plot_data_process:
    print("\nLast index containing NaN in all data:", max_index)


# drop the rows up to the last Nans in data of minutes - 131 minutes
start_index = max_index + 1
data_min = data_min.iloc[start_index:, :]
data_min = data_min.reset_index(drop=True)
start_hour = start_index // 60
start_minute = start_index % 60
# Drop Non-numeric columns
data_min.drop(['Timestamp'], inplace=True, axis=1)

# reorder columns by importance:
new_columns_order = ['Closing_price', 'Volume_of_transactions', 'Opening_price', 'Highest_price', 'Lowest_price',
                     'TRIX', 'VWAP', 'MACD', 'EV_MACD', 'MOM', 'RSI', 'IFT_RSI', 'TR', 'ATR', 'BBWIDTH', 'DMI_1',
                     'DMI_2', 'ADX', 'STOCHRSI', 'MI', 'CHAIKIN', 'VZO', 'PZO', 'EFI', 'EBBP_1', 'EBBP_2', 'BASP_1',
                     'BASP_2', 'BASPN_1', 'BASPN_2', 'WTO', 'SQZMI', 'VFI', 'STC']
data_min = data_min[new_columns_order]
# show info of data - now there are no Nans
if plot_data_process:
    data_min.info()

# %%
# show all features
pd.options.display.max_columns = data_min.shape[1]
# print first 5 lines of data
if plot_data_process:
    data_min.head()


# print last 5 lines of data
if plot_data_process:
    data_min.tail()


# plot chosen features per minute
if plot_data_process:
    plot_scaled_features(data_min,
                         features_to_plot=['Closing_price', 'Volume_of_transactions', 'TRIX', 'ATR'],
                         scaler=pp.StandardScaler(),
                         xlabel="Time - Minutes From (UTC+8): 2021-01-01 {:02d}:{:02d}:00".format(start_hour,
                                                                                                  start_minute)
                         )


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
        test_time = np.arange(np.size(train_df, 0) + np.size(val_df, 0),
                              np.size(train_df, 0) + np.size(val_df, 0) + np.size(test_df, 0))
    else:
        test_time = np.arange(np.size(train_df, 0), np.size(train_df, 0) + np.size(test_df, 0))
    fig = plt.figure(figsize=(16, 8))
    st = fig.suptitle("Data Separation", fontsize=20)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Time - Minutes From (UTC+8): 2021-01-01 {:02d}:{:02d}:00".format(start_hour, start_minute))
    ax.set_ylabel("Closing Price [USD]")
    ax.set_title("Closing Price Through Time")
    ax.plot(train_time, train_df['Closing_price'], label='Training data')
    if val_df is not None:
        ax.plot(val_time, val_df['Closing_price'], label='Validation data')
    ax.plot(test_time, test_df['Closing_price'], label='Test data')
    ax.grid()
    ax.legend(loc="best", fontsize=12)
    plt.show()


# Hyper-Parameters interpretation
predicted_feature = train_df.columns.get_loc('Closing_price')
if scaler_name == 'standard':
    scaler = pp.StandardScaler()
if scaler_name == 'minmax':
    scaler = pp.MinMaxScaler()
in_features = num_features
criterion = nn.MSELoss()  # 可以试试绝对值误差

# define the model
model = BTC_Transformer(num_encoder_layers=num_encoder_layers,
                        num_decoder_layers=num_decoder_layers,
                        in_features=in_features,
                        periodic_features=periodic_features,
                        out_features=out_features,
                        nhead=nhead,
                        dim_feedforward=dim_feedforward,
                        dropout=dropout,
                        activation=activation
                        ).to(device)

# optimization
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

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

# %%
# print the model
model_stats = summary(
    model=model,
    input_data=[(bptt_src, in_features), (bptt_tgt, in_features)],
    dtypes=[torch.float, torch.float],
    batch_dim=1,
    col_names=("input_size", "output_size", "num_params"),
    device=device,
    depth=4,
    verbose=0
)
print(str(model_stats))


# train the model
# prepare vars to hold best model and training statistics
best_val_loss = float('inf')
best_model = None
train_loss = []
valid_loss = []

for epoch in range(1, epochs + 1):
    # epoch initialization
    epoch_start_time = time.time()
    model.train()
    total_loss = 0.
    epoch_loss = 0.
    start_time = time.time()

    # start point of the data
    if random_start_point:
        start_point = np.random.randint(bptt_src)
    else:
        start_point = 0

    num_batches = (len(train_data) - start_point) // bptt_src
    log_interval = round(num_batches // 3 / 10) * 10

    # masks for the model 
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)  # zeros mask for the source (no mask)
    tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)  # look-ahead mask for the target

    for batch, i in enumerate(range(start_point, train_data.size(0) - 1, bptt_src)):
        # forward
        source, targets = get_batch(train_data, i, bptt_src, bptt_tgt, overlap)
        src_batch_size = source.size(0)
        tgt_batch_size = targets.size(0)
        if tgt_batch_size != bptt_tgt or src_batch_size != bptt_src:  # only on last batch
            src_mask = src_mask[:src_batch_size, :src_batch_size]
            tgt_mask = tgt_mask[:tgt_batch_size, :tgt_batch_size]
        output = model(source, targets, src_mask, tgt_mask)
        loss = criterion(output[:-1, :, predicted_feature], targets[1:, :, predicted_feature])

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
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.6f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.6f} ')
            total_loss = 0
            start_time = time.time()

    # record epoch statistics
    train_loss.append(epoch_loss / (len(train_data) - 1))

    # evaluate on validation and save best model
    if val is not None:
        val_loss = evaluate(model, val_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
        elapsed = time.time() - epoch_start_time
        print('-' * 77)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.6f} ')
        print('-' * 77)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        valid_loss.append(np.mean(val_loss))

    # scheduler step
    scheduler.step()

if val is None:
    best_model = copy.deepcopy(model)

# %%
# plot loss functions
epochs_list = np.arange(1, epochs + 1)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.set_title("Loss vs. Epochs")
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.plot(epochs_list, train_loss, label='Training Loss')
if val is not None:
    ax.plot(epochs_list, valid_loss, label='Validation Loss')
ax.legend()
plt.show()

# %%
# calculate test loss
test_loss = evaluate(best_model, test_data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device)
test_ppl = math.exp(test_loss)
print('-' * 77)
print(f'test loss {test_loss:5.6f}')
print('-' * 77)


def greedy_decode(model, src, bptt_src, pred_len, overlap):
    """use the model to create a pridection of pred_len out of src
    
    Args:
        model: nn.Module, the model you want to run the data in
        src: Tensor, shape [N, 1, E]
        bptt_src: int, size of back propagation through time, sequence length of source
        pred_len: int, desired prediction length
        overlap: int, number of overlapping elements between source and target
        
    Returns:
        greedy_output: tesnor, shape [pred_len, 1, E], holds the predictions based on src

    """
    model.eval()
    if overlap == 0:
        start_point = src[-1:, :, :]
        src = src[:-1, :, :]
    else:
        start_point = src[-overlap:, :, :]
    bptt_src = src.size(0)
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    memory = model.encode(src, src_mask)
    targets = start_point
    for i in range(pred_len):
        tgt_mask = model.transformer.generate_square_subsequent_mask(targets.size(0)).to(device)
        prediction = model.decode(targets, memory, tgt_mask)
        prediction = model.generator(prediction)
        targets = torch.cat([start_point, prediction[:i + 1]], dim=0)

    greedy_output = prediction[:pred_len, :, :]
    greedy_output = greedy_output.detach()
    return greedy_output


def estimate_BTC(best_model, test, num_features, bptt_src, bptt_tgt, overlap, predicted_feature, scaler, use_real=True,
                 early_stop=1):
    """use the best model to create a pridection on the test up to early_stop percentage of it
    
    Args:
        best_model: nn.Module, the model you want to run the data in
        test: Tensor, shape [N, E]
        num_features: int, number of features used to train the model
        bptt_src: int, size of back propagation through time, sequence length of source
        bptt_tgt: int, size of back propagation through time, sequence length of target
        overlap: int, number of overlapping elements between source and target
        predicted_feature: int, index of the feature you want to evaluate in [0,E-1]
        scaler: function, scaler fitted to train set
        use_real: bool, when True use the real data for future predictions, when False use the previous prediction
        early_stop: float, percentage of test you want to predict (0,1]
        

    Returns:
        feature_unnormalized: ndarray, holds the real values of the chosen feature in the test in original scale
        feature_prediction_unnormalized: ndarray, holds the predicted values of the chosen feature starting from bptt_src in original scale
        inference_bptt_src: int, starting point of the predictions
    """
    inference_batch_size = 1
    inference_bptt_src = bptt_src + (overlap == 0)
    pred_len = min(bptt_tgt - overlap, bptt_tgt - 1)
    test_data = betchify(test[:, :num_features], inference_batch_size).float()
    num_iter = (test_data.size(0) - bptt_src) // pred_len
    inference_data = test_data[:inference_bptt_src, :, :]
    for i in range(num_iter):
        prediction = greedy_decode(best_model, inference_data, bptt_src, pred_len, overlap)
        if use_real:
            inference_data = test_data[i * pred_len: i * pred_len + inference_bptt_src, :, :]
        else:
            inference_data = torch.cat([inference_data, prediction], dim=0)[pred_len:]
        if i == 0:
            predictions = prediction
        else:
            predictions = torch.cat([predictions, prediction], dim=0)
        if i > num_iter * early_stop:
            break

    feature_unnormalized = scaler.inverse_transform(torch.transpose(test_data, 0, 1).reshape(-1, num_features).cpu())[:,
                           predicted_feature]
    feature_prediction_unnormalized = scaler.inverse_transform(
        torch.transpose(predictions, 0, 1).reshape(-1, num_features).cpu())[:, predicted_feature]

    return feature_unnormalized, feature_prediction_unnormalized, inference_bptt_src


# predict over the test set
early_stop = 1
feature_real, feature_prediction, pred_start = estimate_BTC(best_model, test, num_features, bptt_src, bptt_tgt,
                                                            overlap, predicted_feature, scaler, use_real=True,
                                                            early_stop=early_stop)
real_time = np.arange(0, np.size(feature_real))
prediction_time = np.arange(pred_start, pred_start + np.size(feature_prediction))

# %%
# plot predictions with real price
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Time [Minutes]")
ax.set_ylabel("Closing Price [USD]")
ax.set_title("Closing price through time")
ax.grid()
ax.plot(real_time, feature_real, label='Real Values')
ax.plot(prediction_time, feature_prediction, label='Predicted Values')
ax.set_xlim(0, np.size(prediction_time))
ax.set_ylim(min(min(feature_prediction), min(feature_real[:np.size(prediction_time)])),
            max(max(feature_prediction), max(feature_real[:np.size(prediction_time)])))
ax.legend()
plt.show()

# %%
# plot predictions with real price - zoom in to point of interest
desired_range = [42550, 42650]
low = desired_range[0]
high = desired_range[1]
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("Time [Minutes]")
ax.set_ylabel("Closing Price [USD]")
ax.set_title("Closing price through time - Zoom In")
ax.grid()
ax.plot(real_time, feature_real, label='Real Values')
ax.plot(prediction_time, feature_prediction, label='Predicted Values')
ax.set_xlim(low, high)
ax.set_ylim(min(min(feature_prediction[low:high]), min(feature_real[low:high])),
            max(max(feature_prediction[low:high]), max(feature_real[low:high])))
ax.legend()
plt.show()
