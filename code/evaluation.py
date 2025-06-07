import torch
from data_process import get_batch


'''def evaluate(model, data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    """run the data through the model in eval mode and calculate the average loss in the given feature

    Args:
        model: nn.Module, the model you want to run the data in
        data: Tensor, shape [N, batch_size, E]
        bptt_src: int, size of back propagation through time, sequence length of source
        bptt_tgt: int, size of back propagation through time, sequence length of target
        overlap: int, number of overlapping elements between source and target
        criterion: nn.module, the loss function
        predicted_feature: int, index of the feature you want to evaluate in [0,E-1]


    Returns:
        mean_loss: float, average loss recieved over all data on the chosen feature
    """
    model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)  # zeros mask for the source (no mask)
    tgt_mask = model.transformer.generate_square_subsequent_mask(bptt_tgt).to(device)  # look-ahead mask for the target
    with torch.no_grad():
        for i in range(0, data.size(0) - 1, bptt_src):
            source, targets = get_batch(data, i, bptt_src, bptt_tgt, overlap)
            src_batch_size = source.size(0)
            tgt_batch_size = targets.size(0)
            if tgt_batch_size != bptt_tgt or src_batch_size != bptt_src:  # only on last batch
                src_mask = src_mask[:src_batch_size, :src_batch_size]
                tgt_mask = tgt_mask[:tgt_batch_size, :tgt_batch_size]
            output = model(source, targets, src_mask, tgt_mask)
            # loss = criterion(output[:-1, :, predicted_feature], targets[1:, :, predicted_feature])
            # 用于分类任务
            targets = (targets > 0.5).long()  # 先转换为0和1构成的数列
            targets = targets[-1, :, 0]
            output = output.view(-1, output.size(-1))
            loss = criterion(output, targets)
            total_loss += len(source) * loss.item()
    mean_loss = total_loss / (len(data) - 1)
    return mean_loss'''


def evaluate(model, data, bptt_src, bptt_tgt, overlap, criterion, predicted_feature, device):
    model.eval()
    total_loss = 0.
    src_mask = torch.zeros((bptt_src, bptt_src), dtype=torch.bool).to(device)
    tgt_mask = model.decoder.layers[0].self_attn.bias.bool().new_ones((bptt_tgt, bptt_tgt)).tril().logical_not().to(
        device)

    with torch.no_grad():
        for i in range(0, data.size(1) - 1, bptt_src):  # iterate over time dim
            source, targets = get_batch(data, i, bptt_src, bptt_tgt, overlap)

            if source.size(1) != bptt_src or targets.size(1) != bptt_tgt:
                src_mask = torch.zeros((source.size(1), source.size(1)), dtype=torch.bool).to(device)
                tgt_mask = torch.triu(torch.ones((targets.size(1), targets.size(1)), dtype=torch.bool), diagonal=1).to(
                    device)

            output = model(source, targets, src_mask, tgt_mask)

            targets = (targets > 0.5).long()
            targets = targets[:, -1, 0]
            output = output.view(-1, output.size(-1))
            loss = criterion(output, targets)
            total_loss += source.size(0) * loss.item()

    mean_loss = total_loss / (data.size(0) * (data.size(1) - 1))
    return mean_loss
