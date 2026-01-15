import numpy as np
np.seterr(all='raise') 

import torch
from numpy.linalg import cond
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls
from tqdm import tqdm

def omp_select(grads_matrix, grad_sum, k, device="cuda", lam=0.5, track_diagnostics=False, positive=False):
    """
    Runs OMP solver to pick up to 'k' subbatches that best 
    approximate grad_sum in grads_matrix.
    grads_matrix: (N, param_dim)
    grad_sum: (param_dim,)
    k: int, max number of samples to select
    """
    # OMP expects shape (d, n) => param_dim x N
    A = grads_matrix.transpose(0, 1).to(device) # shape: (d, N)
    b = grad_sum.to(device) # shape: (d,)

    # Call solver
    x_soln, support, per_iter_resid = OrthogonalMP_REG_CUDA(
        A,
        b,
        tol=1e-4,
        nnz=k, 
        positive=positive,
        lam=lam, # Tikhonov regularization
        device=device
    ) # x_soln is a vector of chosen columns with weighted importance; (N,)

    support_tensor = torch.tensor(support, device=device, dtype=torch.long)
    if support_tensor.numel() > k:
        _, order = torch.topk(x_soln.abs()[support_tensor], k=k)
        support_tensor = support_tensor[order]

    sel_rows = support_tensor.detach().cpu().tolist()
    sel_coeffs = x_soln[support_tensor].detach().cpu()  # keep raw OMP weights (ridge reg already applied)

    if track_diagnostics:
        # visualizing the first OMP iteration's projection distribution
        # if flat -> lots of redundant points and near ties -> low jaccard similarity
        # not actually a histogram since large num of subbatches, but interpreted similarly
        p = (A.t() @ b).detach()
        proj = p if positive else p.abs() # force positive if not alr for ranking purposes

        # singleton metrics -> see how well each subbatch does to approx gradient
        # look at top-k and bottom-k atoms to see how well they approx

        sort_idx = torch.argsort(proj, descending=True)
        topk_idx  = sort_idx[:k]
        bottomk_idx = sort_idx[-k:]

        topk_cos, topk_res_ratio = singleton_metrics(A, b, topk_idx)
        bottomk_cos, bottomk_res_ratio = singleton_metrics(A, b, bottomk_idx)

        # get cosine similarity of subset with full gradient
        Ax = A[:, idx_t] @ sel_coeffs_dev
        nb = float(b.norm() + 1e-12)
        nax = float(Ax.norm() + 1e-12)
        rr = float((b - Ax).norm() / nb)
        cos = float((b @ Ax) / (nb * nax)) if nax > 0 else 0.0

        diagnostics = {
            "residual_ratio": rr,
            "cosine_align": cos,
            "iters": len(support),
            "per_iter_residual": [float(v) for v in per_iter_resid],
            
            # raw projections & indices for plotting
            "proj_vals": proj.detach().cpu().numpy().tolist(),
            "positive_mode": bool(positive),
            "topk_indices": topk_idx.detach().cpu().tolist(),
            "bottomk_indices": bottomk_idx.detach().cpu().tolist(),
            
            # singleton quality arrays
            "topk_cos": topk_cos.numpy().tolist(),
            "topk_rr": topk_res_ratio.numpy().tolist(),
            "botk_cos": bottomk_cos.numpy().tolist(),
            "botk_rr": bottomk_res_ratio.numpy().tolist(),
        }

        return sel_rows, sel_coeffs, diagnostics

    else:
        return sel_rows, sel_coeffs

def singleton_metrics(A, b, cols_idx):
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

# from cords project: https://github.com/decile-team/cords/blob/main/cords/selectionstrategies/helpers/omp_solvers.py
# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_CUDA(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    
    OMP is a greedy approach that finds a sparse solution to the system Ax=b
    iteratively selects the best column (sample/subbatch), to reduce the residual between A x and b -> min(b-Ax).
    picks the data points that best approximate the “full gradient.”
    
    Args:
      A: design matrix of size (d, N)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n

    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    per_iter_resid = [] # tracking residual

    for i in range(nnz):
        rratio = (resid.norm().item() / normb)
        per_iter_resid.append(rratio)
        if rratio < tol:
            break

        projections = AT @ resid # (n,)
        idx = torch.argmax(projections if positive else projections.abs())
        if idx in indices:
            break
        indices.append(idx)

        if len(indices) == 1:
            a = A[:, idx] # (d,)
            A_i = a.unsqueeze(0) # (1,d)
            x_i = projections[idx] / torch.dot(a, a)
            x_i = x_i.view(1,1)
        else:
            A_i = torch.cat((A_i, A[:, idx].unsqueeze(0)), dim=0) # (k, d)
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            
            if positive:
                while min(x_i) < 0.0:
                    argmin = torch.argmin(x_i)
                    #indices = indices[:argmin] + indices[argmin + 1:]
                    del indices[argmin.item()]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]), dim=0)
                    if argmin.item() == A_i.shape[0]:
                        break
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        if argmin.item() == A_i.shape[0]:
            break

        resid = b - (A_i.t() @ x_i).view(-1) # A_i.T.dot(x_i)

    x_i = x_i.view(-1)
    for i, idx in enumerate(indices):
        try:
            x[idx] += x_i[i]
        except IndexError:
            x[idx] += x_i
    
    return x, indices, per_iter_resid

def OrthogonalMP_REG_Parallel_V1(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []

    argmin = torch.tensor([-1])
    omp_pbar = tqdm(
        range(nnz),
        total=nnz,
        desc=f"OMP (k={nnz} batches)",
        leave=True,
        dynamic_ncols=True,
    )
    for i in omp_pbar:
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)

        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))

        if index not in indices:
            indices.append(index)

        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:
                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
    x_i = x_i.view(-1)
    for i, index in enumerate(indices):
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    return x