import torch


def recall_at_k(sim_matrix: torch.Tensor, k: int):
    if sim_matrix.size(1) < k:
        return None
    B = sim_matrix.size(0)
    labels = torch.arange(B, device=sim_matrix.device)
    topk = sim_matrix.topk(k, dim=1).indices
    correct = (topk == labels.unsqueeze(1)).any(dim=1)
    recall = correct.float().mean()
    return recall


def uniformity(features, max_samples=2048):
    if features.size(0) > max_samples:
        indices = torch.randperm(features.size(0), device=features.device)[:max_samples]
        features = features[indices]
    sq_dists = torch.pdist(features, p=2).pow(2)
    return sq_dists.mul(-2).exp().mean().log()
