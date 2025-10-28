import numpy as np

np.seterr(all="raise")

import torch
from numpy.linalg import cond, inv, norm
from scipy import sparse as sp
from scipy.linalg import lstsq, solve
from scipy.optimize import nnls


def omp_select(grads_matrix, grad_sum, budget, device="cuda"):
    """
    Runs OMP solver to pick up to 'budget' subbatches that best
    approximate grad_sum in grads_matrix.
    grads_matrix: (N, param_dim)
    grad_sum: (param_dim,)
    budget: int, max number of samples to select
    """
    # OMP expects shape (d, n) => param_dim x N
    A = grads_matrix.transpose(0, 1).to(device)  # shape: (d, N)
    b = grad_sum.to(device)  # shape: (d,)

    # Call your solver
    x_solution = OrthogonalMP_REG_CUDA(
        A,
        b,
        tol=1e-4,
        nnz=budget,
        positive=False,
        lam=1,  # Tikhonov regularization
        device=device,
    )
    # x_solution is shape (N,). Nonzero (or largest absolute) entries => chosen columns.

    # use the top-k largest absolute values:
    _, selected_subbatch_idxs = torch.topk(torch.abs(x_solution), k=budget)
    selected_subbatch_idxs = selected_subbatch_idxs.cpu().tolist()
    return selected_subbatch_idxs


# from cords project: https://github.com/decile-team/cords/blob/main/cords/selectionstrategies/helpers/omp_solvers.py
# NOTE: Standard Algorithm, e.g. Tropp, ``Greed is Good: Algorithmic Results for Sparse Approximation," IEEE Trans. Info. Theory, 2004.
def OrthogonalMP_REG_CUDA(
    A, b, tol=1e-4, nnz=None, positive=False, lam=1, device="cpu"
):
    """approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit

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
    """
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat(
                (A_i, A[:, index].view(1, -1)), dim=0
            )  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(
                A_i.shape[0], device=device
            )
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:

                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1 :]
                    A_i = torch.cat(
                        (A_i[:argmin], A_i[argmin + 1 :]), dim=0
                    )  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    if argmin.item() == A_i.shape[0]:
                        break
                    # print(argmin.item(),A_i.shape[0],index.item())
                    temp = torch.matmul(
                        A_i, torch.transpose(A_i, 0, 1)
                    ) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(
                        temp, torch.matmul(A_i, b).view(-1, 1)
                    )
        if argmin.item() == A_i.shape[0]:
            break
        # print(b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape,\
        #  torch.matmul(torch.transpose(A_i, 0, 1), x_i).shape)
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(
            -1
        )  # A_i.T.dot(x_i)
        # print("REsID",resid.shape)

    x_i = x_i.view(-1)
    # print(x_i.shape)
    # print(len(indices))
    for i, index in enumerate(indices):
        # print(i,index,end="\t")
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    # print(x[indices])
    return x
