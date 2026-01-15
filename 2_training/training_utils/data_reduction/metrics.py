import torch
import numpy as np


def compute_pairwise_jaccard_similarity(s0, s1):
    # quantify how much new set s1 changed from old set s0
    if isinstance(s0, list):
        s0 = set(s0)
    if isinstance(s1, list):
        s1 = set(s1)

    intersection = len(s0 & s1)
    union = len(s0 | s1)
    jaccard = intersection / union

    return jaccard

def _singleton_metrics(A, b, cols_idx):
    """
    A: (d, n)
    b: (d,)
    cols_idx: 1D LongTensor of size (k,)
    Returns:
        cos: (k,) cosine between each atom a_j and b
        rr : (k,) residual ratio ||b - a_j * alpha|| / ||b||
    """
    a = A[:, cols_idx]                                     # (d, k)
    nb = b.norm() + 1e-12
    na = a.norm(dim=0) + 1e-12                             # (k,)
    cos = (a.t() @ b) / (na * nb)                          # (k,)

    numer = (a.t() @ b)                                    # (k,)
    denom = (a.t() * a.t()).sum(dim=1) + 1e-12             # (k,) = ||a_j||^2
    alpha = numer / denom                                  # (k,)

    r = b.unsqueeze(1) - a * alpha.unsqueeze(0)            # (d, k)
    rr = r.norm(dim=0) / nb                                # (k,)
    return cos.detach().cpu(), rr.detach().cpu()

@torch.no_grad()
def omp_diagnostics(A, b, reg, k, positive=False):
    """
    Compute OMP-related diagnostics similar to your old implementation.

    Args:
        A: (d, n) design matrix (atoms are columns)
        b: (d,) target vector
        reg: (n,) OMP coefficients (your solver output)
        k: int, nominal budget (top-k)
        positive: bool, same semantics you use for OMP

    Returns:
        diagnostics: dict
    """
    device = A.device
    eps = 1e-12
    n = A.shape[1]

    # projection distribution; p = A^T b
    p = (A.t() @ b).detach() # (n,)
    proj = p if positive else p.abs() 

    # top/bottom-k atoms by projection magnitude
    kk = min(int(k), int(n))
    sort_idx = torch.argsort(proj, descending=True)
    topk_idx = sort_idx[:kk]
    bottomk_idx = sort_idx[-kk:] if kk > 0 else sort_idx[:0]

    topk_cos, topk_res_ratio = _singleton_metrics(A, b, topk_idx)
    bottomk_cos, bottomk_res_ratio = _singleton_metrics(A, b, bottomk_idx)

    # chosen support from reg (nonzeros)
    support_idx = torch.nonzero(reg).view(-1)
    nnz = int(support_idx.numel())

    # subset reconstruction metrics (Ax vs b)
    nb = float(b.norm() + eps)
    if nnz > 0:
        coeffs = reg[support_idx]                       # (nnz,)
        Ax = A[:, support_idx] @ coeffs                 # (d,)
        nax = float(Ax.norm() + eps)
        residual_ratio = float((b - Ax).norm() / nb)
        cos_sim = float((b @ Ax) / (nb * nax)) if nax > 0 else 0.0
    else:
        residual_ratio = 1.0
        cos_sim = 0.0

    # some useful scalar summaries of the projection distribution
    proj_cpu = proj.detach().float().cpu()
    diagnostics = {
        # --- keep key names from your old code where possible ---
        "residual_ratio": residual_ratio,
        "cos_sim": cos_sim,

        "topk_cos": topk_cos.tolist(),                       # (k,)
        "topk_res_ratio": topk_res_ratio.tolist(),           # (k,)
        "bottomk_cos": bottomk_cos.tolist(),                 # (k,)
        "bottomk_res_ratio": bottomk_res_ratio.tolist(),     # (k,)

        "proj_vals": proj_cpu.tolist(),
        "proj_mean": float(proj_cpu.mean().item()) if proj_cpu.numel() else 0.0,
        "proj_std": float(proj_cpu.std(unbiased=False).item()) if proj_cpu.numel() else 0.0,
        "proj_max": float(proj_cpu.max().item()) if proj_cpu.numel() else 0.0,
        "proj_min": float(proj_cpu.min().item()) if proj_cpu.numel() else 0.0,

        "n_atoms": int(n),
        "k": int(k),
        "reg_nnz": nnz,
    }

    return diagnostics