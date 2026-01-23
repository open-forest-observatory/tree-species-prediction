import torch
import numpy as np

from training_utils.ctx import vram_ctx

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

def _singleton_metrics(A, b, cols_idx, eps=1e-8):
    """
    Finds quantifies each batches contribution to the target gradient
    Note: these are vectorized operations to find indv batch contributions to the full gradient,
    so while comments may indicate 'scalar', these fns all operate on vectors, it's just scalars if it were done iteratively.

    Args: 
        A: (d, n); gradient of loss wrt classifier head params (i.e. no base/pretrained model params)
        b: (d,); target gradient; row sum of gradient matrix
        cols_idx: 1D LongTensor of size (k,) (selected batches)
    Returns:
        cos: (k,) cosine between each atom a_j and b
        rr : (k,) residual ratio ||b - a_j * alpha|| / ||b||
    """
    a = A[:, cols_idx]                              # (d, k); single column/atom from gradient matrix; 
    nb = b.norm() + eps                             # (k,); scalar norm gradient sum 
    sqna = (a.T @ a) + eps                          # (k,); squared norm of atom
    na = sqna.sqrt()                                # (k,); scalar norm of singleton
    inner_prod = (a.T @ b)                          # (k,); inner product of atom and tgt; raw directional alignment
    cos = inner_prod / (na * nb)                    # (k,); cosine similarity between atom and target; norm inner product

    # compute squared residual norm using the identity: ||b - alpha a_j||^2 = ||b||^2 - 2 alpha (a_j^T b) + alpha^2 ||a_j||^2
    # this avoids explicitly forming the (d, k) residual matrix
    alpha = (a.T @ b)                               # (k,); projection coeff of singleton gradient onto target
    bTb = (b @ b) + eps
    r2 = bTb - 2.0 * alpha * inner_prod + (alpha * alpha) * sqna
    rr = r2.clamp_min(0.0).sqrt() / nb
    return cos.detach().cpu(), rr.detach().cpu()

@torch.no_grad()
def omp_diagnostics(A, b, reg, k, positive=False, epoch=None):
    """
    Compute OMP-related diagnostics; adapted from previous project.

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
    with vram_ctx('DR-omp-metrics-projectiondist', epoch=epoch):
        p = (A.t() @ b).detach() # (n,)
        proj = p if positive else p.abs() 

    # top/bottom-k atoms by projection magnitude
    with vram_ctx('DR-omp-metrics-projectiondistsorted', epoch=epoch):
        kk = min(int(k), int(n))
        _, topk_idx = torch.topk(proj, kk, largest=True)        # most informative batch idxs
        _, bottomk_idx = torch.topk(proj, kk, largest=False)    # least informative batch idxs

    with vram_ctx('DR-omp-metrics-_singleton_metrics-topk', epoch=epoch):
        topk_cos, topk_res_ratio = _singleton_metrics(A, b, topk_idx) # how informative are each of the worst samples indv

    with vram_ctx('DR-omp-metrics-_singleton_metrics-bottomk', epoch=epoch):
        bottomk_cos, bottomk_res_ratio = _singleton_metrics(A, b, bottomk_idx) # how informative are each of the worst samples indv

    # chosen support from reg (nonzeros)
    support_idx = torch.nonzero(reg).view(-1)
    nnz = int(support_idx.numel())

    # subset reconstruction metrics (Ax vs b)
    with vram_ctx('DR-omp-metrics-subsetreconstruction', epoch=epoch):
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
    with vram_ctx('DR-omp-metrics-diagdict', epoch=epoch):
        diagnostics = {
            "residual_ratio": residual_ratio,
            "cos_sim": cos_sim,

            "topk_cos": topk_cos.tolist(),               
            "topk_res_ratio": topk_res_ratio.tolist(),   
            "bottomk_cos": bottomk_cos.tolist(),         
            "bottomk_res_ratio": bottomk_res_ratio.tolist(), 

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