"""Stage-B mini-set construction for ST slides.

ST mini-sets are spatially-coherent patches drawn from precomputed per-slide
targets; each carries the frozen encoder embeddings plus a pose-free Gram
geometry target. These are the training examples for the Stage-C generator +
diffusion refiner.
"""

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from .targets import STTargets, factor_from_gram


def _sample_local_patch(dists: torch.Tensor, idx_no_self: torch.Tensor,
                        all_idx: torch.Tensor, center_idx: int, n: int,
                        pool_mult: float, stochastic_tau: float) -> torch.Tensor:
    """Draw an n-point patch: center + stochastically sampled near neighbors.

    Neighbors are drawn without replacement from a distance-ordered candidate
    pool with softmax weights, so patches are locally coherent but varied.
    """
    if idx_no_self.numel() == 0:
        return all_idx[:n]

    sort_order = torch.argsort(dists)
    sorted_neighbors = idx_no_self[sort_order]
    sorted_dists = dists[sort_order]

    n_needed = n - 1
    K_pool = min(sorted_neighbors.numel(), int(pool_mult * n))
    K_pool = max(K_pool, n_needed)
    pool = sorted_neighbors[:K_pool]
    pool_d = sorted_dists[:K_pool]

    if pool.numel() <= n_needed:
        neighbors = pool
    else:
        weights = torch.softmax(-pool_d / stochastic_tau, dim=0)
        sampled = torch.multinomial(weights, n_needed, replacement=False)
        neighbors = pool[sampled]

    indices = torch.cat([torch.tensor([center_idx], dtype=torch.long), neighbors])

    if indices.numel() < n:
        missing = n - indices.numel()
        extra_pool = all_idx[~torch.isin(all_idx, indices)]
        if extra_pool.numel() > 0:
            perm = torch.randperm(extra_pool.numel())
            indices = torch.cat([indices, extra_pool[perm[: min(missing, extra_pool.numel())]]])
    return indices


class STSetDataset(Dataset):
    def __init__(
        self,
        targets_dict: Dict[int, STTargets],
        encoder: torch.nn.Module,
        st_gene_expr_dict: Dict[int, torch.Tensor],
        n_min: int = 96,
        n_max: int = 384,
        D_latent: int = 32,
        num_samples: int = 10000,
        device: str = "cuda",
        pool_mult: float = 2.0,
        stochastic_tau: float = 1.0,
    ):
        self.targets_dict = targets_dict
        self.st_gene_expr_dict = st_gene_expr_dict
        self.n_min = n_min
        self.n_max = n_max
        self.D_latent = D_latent
        self.num_samples = num_samples
        self.device = device
        self.pool_mult = pool_mult
        self.stochastic_tau = stochastic_tau
        self.slide_ids = list(targets_dict.keys())

        # Precompute frozen encoder embeddings for every slide.
        self.Z_dict = {}
        encoder.eval()
        with torch.no_grad():
            for slide_id, st_expr in st_gene_expr_dict.items():
                self.Z_dict[slide_id] = encoder(st_expr.to(device)).cpu()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        slide_id = np.random.choice(self.slide_ids)
        targets = self.targets_dict[slide_id]

        m = targets.y_hat.shape[0]
        n = min(np.random.randint(self.n_min, self.n_max + 1), m)

        # Local patch in ST distance space.
        center_idx = np.random.randint(0, m)
        all_idx = torch.arange(m)
        mask_self = all_idx != center_idx
        indices = _sample_local_patch(
            targets.D[center_idx][mask_self], all_idx[mask_self], all_idx,
            center_idx, n, self.pool_mult, self.stochastic_tau,
        )
        indices = indices[torch.randperm(indices.numel())]  # avoid positional bias
        n_total = indices.numel()

        # Spatial kNN remapped into local mini-set indexing (-1 = outside set).
        global_to_local = torch.full((m,), -1, dtype=torch.long)
        global_to_local[indices] = torch.arange(n_total, dtype=torch.long)
        knn_spatial_local = global_to_local[targets.knn_spatial[indices]]

        Z_set = self.Z_dict[slide_id][indices]
        y_hat_subset = targets.y_hat[indices]

        # Pose-free Gram target from the centered patch coordinates.
        y_hat_centered = y_hat_subset - y_hat_subset.mean(dim=0, keepdim=True)
        G_subset = y_hat_centered @ y_hat_centered.t()
        V_target = factor_from_gram(G_subset, self.D_latent)

        return {
            "Z_set": Z_set,
            "V_target": V_target,
            "G_target": G_subset,
            "knn_spatial": knn_spatial_local,
            "spot_indices": indices,
            "n": n_total,
        }


def collate_minisets(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate variable-size ST mini-sets with padding and a validity mask."""
    device = batch[0]["Z_set"].device
    n_max = max(item["n"] for item in batch)
    B = len(batch)
    h_dim = batch[0]["Z_set"].shape[1]
    D_latent = batch[0]["V_target"].shape[1]
    knn_k = batch[0]["knn_spatial"].shape[1]

    Z = torch.zeros(B, n_max, h_dim, device=device)
    V = torch.zeros(B, n_max, D_latent, device=device)
    G = torch.zeros(B, n_max, n_max, device=device)
    mask = torch.zeros(B, n_max, dtype=torch.bool, device=device)
    n_batch = torch.zeros(B, dtype=torch.long)
    knn_spatial = torch.full((B, n_max, knn_k), -1, dtype=torch.long, device=device)
    spot_indices = torch.full((B, n_max), -1, dtype=torch.int64, device=device)

    for i, item in enumerate(batch):
        n = item["n"]
        Z[i, :n] = item["Z_set"]
        V[i, :n] = item["V_target"]
        G[i, :n, :n] = item["G_target"]
        mask[i, :n] = True
        n_batch[i] = n
        knn_spatial[i, :n, :] = item["knn_spatial"]
        spot_indices[i, :n] = item["spot_indices"].long()

    return {
        "Z_set": Z, "V_target": V, "G_target": G, "mask": mask, "n": n_batch,
        "knn_spatial": knn_spatial, "spot_indices": spot_indices,
    }
