"""Stage-B geometric target precomputation.

For each ST slide we precompute a small set of pose-free geometric quantities
from the (already pose-normalized) spatial coordinates:

    - y_hat        : the coordinates themselves (unchanged)
    - D            : graph-geodesic pairwise distances on a kNN graph, used to
                     draw spatially-coherent mini-sets
    - knn_spatial  : nearest-neighbor indices from the raw coordinates, used as
                     the neighborhood structure target during Stage-C training

These are computed once per slide and reused across all mini-sets drawn from
that slide. This module also holds the PSD Gram factorization helper used to
build per-mini-set geometry targets in minisets.py.
"""

from dataclasses import dataclass
from typing import Dict

import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


def compute_geodesic_distances(coords: torch.Tensor, k: int = 15, device: str = "cuda") -> torch.Tensor:
    """Graph-geodesic distances via shortest-path on a symmetric kNN graph."""
    coords_np = coords.cpu().numpy()
    knn_graph = kneighbors_graph(
        coords_np, n_neighbors=k, mode="distance", metric="euclidean", include_self=False
    )
    knn_graph = (knn_graph + knn_graph.T) / 2.0

    D_geo = shortest_path(knn_graph, method="auto", directed=False)
    D_geo = torch.from_numpy(D_geo).float().to(device)

    # Bridge disconnected components with a large-but-finite distance.
    finite_max = D_geo[D_geo != float("inf")].max().item()
    D_geo = torch.nan_to_num(D_geo, nan=0.0, posinf=finite_max * 2)
    return D_geo


def safe_eigh(A: torch.Tensor, return_vecs: bool = True, regularize: float = 1e-6):
    """Symmetric eigendecomposition with a small diagonal regularizer."""
    if A.dtype in (torch.float16, torch.bfloat16):
        A = A.float()

    if A.dim() == 2:
        eye = torch.eye(A.size(0), device=A.device, dtype=A.dtype)
        A_reg = A + regularize * eye
    elif A.dim() == 3:
        B, N, _ = A.shape
        eye = torch.eye(N, device=A.device, dtype=A.dtype).unsqueeze(0).expand(B, -1, -1)
        A_reg = A + regularize * eye
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {A.shape}")

    if return_vecs:
        return torch.linalg.eigh(A_reg)
    return torch.linalg.eigvalsh(A_reg), None


def factor_from_gram(G: torch.Tensor, D_latent: int) -> torch.Tensor:
    """Factor a PSD Gram matrix into V with G ~= V V^T in a canonical gauge.

    Eigenvectors are only defined up to sign; the sign of each column is fixed
    by forcing its max-magnitude entry positive so that V is stable across calls.
    The factor is padded to D_latent columns when the Gram matrix is low-rank.
    """
    eigvals, eigvecs = safe_eigh(G, return_vecs=True)
    eigvals = eigvals.flip(0).clamp(min=0)  # descending, non-negative
    eigvecs = eigvecs.flip(1)

    D = min(D_latent, eigvals.shape[0])
    eigvals_top = eigvals[:D]
    eigvecs_top = eigvecs[:, :D]

    if D > 0 and G.shape[0] > 0:
        abs_max_idx = eigvecs_top.abs().argmax(dim=0)  # (D,)
        col_indices = torch.arange(D, device=G.device)
        sign = torch.sign(eigvecs_top[abs_max_idx, col_indices])
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        eigvecs_top = eigvecs_top * sign.view(1, D)

    V = eigvecs_top @ torch.diag(torch.sqrt(eigvals_top))

    if D < D_latent:
        pad = torch.zeros(V.shape[0], D_latent - D, device=V.device, dtype=V.dtype)
        V = torch.cat([V, pad], dim=1)
    return V


@dataclass
class STTargets:
    """Precomputed geometric quantities for a single ST slide."""
    y_hat: torch.Tensor        # (n, d) pose-normalized coordinates
    D: torch.Tensor            # (n, n) distance matrix used for locality sampling
    knn_spatial: torch.Tensor  # (n, k) spatial-neighbor indices
    k: int                     # kNN parameter


class STStageBPrecomputer:
    """Precompute pose-free geometric quantities for each ST slide.

    Coordinates are expected to arrive already pose-normalized, so no additional
    centering / whitening is applied here.
    """

    def __init__(self, k: int = 20, device: str = "cuda"):
        self.k = k
        self.device = device

    def precompute(
        self,
        slides: Dict[int, torch.Tensor],
        use_geodesic_targets: bool = True,
        geodesic_k: int = 15,
    ) -> Dict[int, STTargets]:
        """Precompute per-slide targets (always recomputed, never cached).

        Args:
            slides: mapping slide_id -> (m, d) pre-normalized spatial coordinates.
            use_geodesic_targets: derive D from graph geodesics (default) rather
                than raw Euclidean distances.
            geodesic_k: kNN parameter for the geodesic graph.
        """
        targets_dict = {}
        for slide_id, st_coords in slides.items():
            print(f"Computing targets for slide {slide_id} (n={st_coords.shape[0]})")
            y_hat = st_coords
            n_spots = y_hat.shape[0]

            if use_geodesic_targets:
                D = compute_geodesic_distances(y_hat, k=geodesic_k, device=self.device)
            else:
                D = torch.cdist(y_hat, y_hat)

            # Spatial kNN from coordinates (self-excluded).
            D_spatial = torch.cdist(y_hat, y_hat)
            D_spatial_noself = D_spatial + torch.eye(n_spots, device=y_hat.device) * 1e10
            _, knn_spatial = torch.topk(D_spatial_noself, k=self.k, dim=1, largest=False)

            targets_dict[slide_id] = STTargets(
                y_hat=y_hat.cpu(),
                D=D.cpu(),
                knn_spatial=knn_spatial.cpu(),
                k=self.k,
            )
            print(f"  Targets computed for slide {slide_id}")
        return targets_dict
