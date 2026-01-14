import math
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import _bootstrap
from training_utils.omp import OrthogonalMP_REG_Parallel_V1
from configs.data_reduction_config import dr_config


def _unpack_batch(batch, device):
    # your collate gives (imgs, labels, metas)
    imgs, labels, metas = batch
    return imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True), metas


class GradMatchPBSelector:
    def __init__(self, model, criterion, device, lam=0.0, eps=1e-4):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.lam = lam
        self.eps = eps

    def _batch_grad_head(self, imgs, labels):
        """
        Batch gradient atom = grad(loss_mean, head_params) flattened to (d,)
        """
        self.model.eval()
        head_params = self.model.head_parameters()
        if not head_params:
            raise RuntimeError("No trainable classifier_head params found for selection.")

        logits = self.model(imgs)
        loss = self.criterion(logits, labels)  # mean CE by default

        grads = torch.autograd.grad(
            loss,
            head_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        g = torch.cat([gi.reshape(-1) for gi in grads], dim=0).detach()

        # IMPORTANT: keep OMP numerically stable
        return g.float()

    def _is_subset_epoch(self, epoch, total_epochs):
        return (
            (epoch <= total_epochs) and  # if we haven't trained for the full spec'd num epochs and,
            (epoch >= dr_config.num_warm_start_epochs) and  # if we have trained enough on the full data and,
            (epoch % dr_config.epoch_selection_interval == 0)  # we are at a selection interval epoch,
        )

    def select_perbatch(self, selection_loader, subset_ratio, positive=True):
        """This returns chosen_subset_indices relative to selection_loader.dataset (which is the train split)

        Returns:
            chosen_subset_indices: list[int] (indices w.r.t. selection_loader.dataset)
            sample_weight_map: dict[int -> float]  (global_idx -> gamma)
            diag: dict
        """
        assert 0 < subset_ratio <= 1.0

        # batch_sampler yields the dataset indices for each batch (deterministic because shuffle=False)
        batch_wise_indices = list(selection_loader.batch_sampler)
        n_batches = len(batch_wise_indices)
        k_batches = max(1, int(math.ceil(subset_ratio * n_batches)))

        # --- build grad atoms (one per batch) ---
        grad_list = []
        grad_pbar = tqdm(
            selection_loader,
            desc=f"GradMatchPB: computing {n_batches} batch-grads",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in grad_pbar:
            imgs, labels, _metas = _unpack_batch(batch, self.device)
            g_b = self._batch_grad_head(imgs, labels)
            grad_list.append(g_b)

        grads_per_batch = torch.stack(grad_list, dim=0)          # (n_batches, d)
        A = grads_per_batch.transpose(0, 1).contiguous()         # (d, n_batches)
        b = grads_per_batch.sum(dim=0).contiguous()              # (d,)

        # --- OMP on GPU (your function) ---
        omp_pbar = tqdm(
            total=1,
            desc=f"GradMatchPB: OMP (k={k_batches}/{n_batches})",
            leave=True,
            dynamic_ncols=True,
        )
        reg = OrthogonalMP_REG_Parallel_V1(
            A, b,
            nnz=k_batches,
            positive=positive,
            lam=self.lam,
            tol=self.eps,
            device=str(self.device),
        )  # (n_batches,)
        omp_pbar.update(1)
        omp_pbar.close()

        chosen_batch_ids = torch.nonzero(reg).view(-1).tolist()

        # selection_loader.dataset is a Subset(sel_cp, train_subset.indices)
        sel_subset = selection_loader.dataset
        base_ds = sel_subset.dataset  # TreeDataset copy
        subset_indices = sel_subset.indices  # maps subset idx -> original tree_dset idx

        chosen_subset_indices = []          # indices into selection subset (used to build new Subset)
        sample_weight_map = {}              # global_idx -> weight

        # --- expand chosen batches -> chosen samples + gammas ---
        expand_pbar = tqdm(
            chosen_batch_ids,
            desc=f"GradMatchPB: expanding {len(chosen_batch_ids)} batches -> samples",
            leave=True,
            dynamic_ncols=True,
        )
        for bid in expand_pbar:
            gamma = float(reg[bid].item())
            subset_batch_idxs = batch_wise_indices[bid]  # indices into sel_subset (0..len(sel_subset)-1)

            chosen_subset_indices.extend(subset_batch_idxs)

            # convert subset idx -> global_idx via meta
            for sub_i in subset_batch_idxs:
                orig_i = subset_indices[sub_i]                  # index into original tree_dset ordering
                gidx = int(base_ds.meta[orig_i]["global_idx"])  # stored in meta
                sample_weight_map[gidx] = gamma

        diag = {
            "n_batches_total": n_batches,
            "k_batches": k_batches,
            "n_points_selected": len(chosen_subset_indices),
            "reg_nnz": len(chosen_batch_ids),
        }
        return chosen_subset_indices, sample_weight_map, diag