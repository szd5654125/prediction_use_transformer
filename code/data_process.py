import torch
import sklearn.preprocessing as pp
from finta import TA


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
