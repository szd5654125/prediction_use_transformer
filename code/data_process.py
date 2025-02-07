import torch
import sklearn.preprocessing as pp
from finta import TA
import datetime as time
from concurrent.futures import ProcessPoolExecutor, as_completed


def train_validation_test_split(data, val_percentage, test_percentage):
    """Splits the data into train, validation and test

    Args:
        data: Tensor, shape [N, E]
        val_percentage: float, validation percentage from the data [0,1]
        test_percentage: float, test percentage from the data (0,1]
    Returns:
        train: Tensor, shape [N - N * (val_percentage + test_percentage), E]
        val: Tensor, shape [N * val_percentage, E] or None if val_percentage = 0
        test: Tensor, shape [N * test_percentage, E]
    """
    data_length = len(data)

    val_length = int(data_length * val_percentage)
    test_length = int(data_length * test_percentage)
    train_length = data_length - val_length - test_length

    train = data[:train_length]
    if val_length == 0:
        val = None
    else:
        val = data[train_length:train_length + val_length]
    test = data[train_length + val_length:]

    return train, val, test


def normalize_data(train, val, test, scaler=pp.StandardScaler()):
    """Scale the data: train, val and test according to train
    Args:
        train: Tensor, shape [N_train, E]
        val: Tensor, shape [N_val, E] (supports val=None)
        test: Tensor, shape [N_test, E]
        scaler: function, scaling function
    Returns:
        train: Tensor, shape [N_train, E]
        val: Tensor, shape [N_val, E]
        test: Tensor, shape [N_test, E]
        fitted_scaler: function, scaler fitted to train
    """
    fitted_scaler = scaler.fit(train)
    train = torch.tensor(fitted_scaler.transform(train))
    if val is not None:
        val = torch.tensor(fitted_scaler.transform(val))

    test = torch.tensor(fitted_scaler.transform(test))

    return train, val, test, fitted_scaler


def betchify(data, batch_size, device):
    """Divides the data into batch_size separate sequences,
    removing extra elements that wouldn't cleanly fit.
    Args:
        data: Tensor, shape [N, E]
        batch_size: int, batch size
    Returns:
        Tensor of shape [N // batch_size, batch_size, E]
    """
    seq_len = data.size(0) // batch_size
    data = data[:seq_len * batch_size, :]
    data = data.view(batch_size, seq_len, -1)
    data = torch.transpose(data, 0, 1).contiguous()
    return data.to(device)


def get_batch(data, i, bptt_src, bptt_tgt, overlap):
    """Divides data to source and target, from offset i
    Args:
        source: Tensor, shape [N, batch_size, E]
        i: int, offset for the source
        bptt_src: int, size of back propagation through time, sequence length of source
        bptt_tgt: int, size of back propagation through time, sequence length of target
        overlap: int, number of overlapping elements between source and target
    Returns:
        source: Tensor, shape [bptt_src, batch_size, E]
        target: Tensor, shape [bptt_tgt, batch_size, E]
    """
    src_seq_len = min(bptt_src, len(data) - i - 1)
    target_seq_len = min(bptt_tgt, len(data) - i - src_seq_len + overlap)
    source = data[i: i + src_seq_len]
    target = data[i + src_seq_len - overlap: i + src_seq_len + target_seq_len - overlap]

    return source, target


def add_finta_feature(data, feature_names, both_columns_features):
    """Adds new fanta features to data by their feature_name in feature_names

    Args:
        data: DataFrame, where the feature will be added
        data_finta: DataFrame, columns' names are: 'open', 'high', 'low', 'close' and 'volume'(optinal)
                    from which the new feature will be calculated
        feature_names: list of strings, names of the new features you want to add
        both_columns_features: list of strings, names of the new features you want to add both of their outputs
    """
    for feature_name in feature_names:
        feature_func = getattr(TA, feature_name)
        finta_feature = feature_func(data)
        if finta_feature.isna().all().all():
            print(f"Feature '{feature_name}' is empty after calculation.")
        if finta_feature.ndim > 1:
            if feature_name in both_columns_features:
                data["{}_1".format(feature_name)] = finta_feature.iloc[:, 0]
                data["{}_2".format(feature_name)] = finta_feature.iloc[:, 1]
            else:
                data[feature_name] = finta_feature.iloc[:, 0]
        else:
            data[feature_name] = finta_feature

    return data


def compute_finta_feature(feature_name, data, both_columns_features):
    """计算单个 Finta 指标，并返回结果"""
    print(f"[START] 开始计算 {feature_name} 指标...")

    try:
        feature_func = getattr(TA, feature_name)
        finta_feature = feature_func(data)

        if finta_feature.isna().all().all():
            print(f"[WARNING] Feature '{feature_name}' is empty after calculation.")
            return None

        if finta_feature.ndim > 1:
            if feature_name in both_columns_features:
                result = {
                    f"{feature_name}_1": finta_feature.iloc[:, 0],
                    f"{feature_name}_2": finta_feature.iloc[:, 1]
                }
            else:
                result = {feature_name: finta_feature.iloc[:, 0]}
        else:
            result = {feature_name: finta_feature}
        return result

    except Exception as e:
        print(f"[ERROR] 计算 {feature_name} 时出错: {e}")
        return None


def add_finta_feature_parallel(data, feature_names, both_columns_features, num_processes=128):
    """多进程计算 Finta 指标"""
    results = []

    print(f"\n[INFO] 开始并行计算 {len(feature_names)} 个指标，使用 {num_processes} 进程...")

    # 使用进程池进行并行计算
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        future_to_feature = {
            executor.submit(compute_finta_feature, feature, data, both_columns_features): feature
            for feature in feature_names
        }

        for future in as_completed(future_to_feature):
            feature_name = future_to_feature[future]  # 获取当前计算的指标名
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[RESULT] {feature_name} 计算完成，已合并到数据集")
            except Exception as e:
                print(f"[ERROR] {feature_name} 计算失败: {e}")

    # 合并结果
    for result in results:
        for col_name, values in result.items():
            data[col_name] = values

    print(f"\n[INFO] 全部指标计算完成，共添加 {len(results)} 个指标。\n")
    return data