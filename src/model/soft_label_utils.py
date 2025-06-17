import torch

def embed_rate_to_softlabel(rate: torch.Tensor, num_classes=11) -> torch.Tensor:
    """
    rate: Tensor of shape [B], values in 0~1
    return: Tensor of shape [B, num_classes], soft-labels
    """
    rate = rate * 10.0  # scale to 0~10
    base = torch.floor(rate).long()
    frac = rate - base
    B = rate.shape[0]
    soft_labels = torch.zeros((B, num_classes), device=rate.device)

    for i in range(B):
        b = base[i].item()
        f = frac[i].item()
        if b < num_classes - 1:
            soft_labels[i, b] = 1 - f
            soft_labels[i, b + 1] = f
        else:
            soft_labels[i, num_classes - 1] = 1.0
    return soft_labels
